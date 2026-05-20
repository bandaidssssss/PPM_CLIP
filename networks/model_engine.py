from loralib import layers as lora_layers
from loralib.utils import mark_only_lora_as_trainable, apply_lora
import torch.nn as nn
import numpy as np
import clip
from networks.DCT_score import DCTPatches,DCTPatches_random
from networks.Bayes import  Context_Prompting,Orthogonal_Loss,TextEncoder_Bayes_V4
import torch
import pdb
INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



class PPM_clip(nn.Module):
    def __init__(self,args):
        super(PPM_clip, self).__init__()

        self.args = args
        if not args.gpu or args.gpu == '-1':
            self.device = torch.device('cpu')
        else:
            gpu_id = args.gpu.split(',')[0] 
            self.device = torch.device(f'cuda:{gpu_id}')

        if "16" in args.backbone:
            patch_size = 16
        elif "14" in args.backbone:
            patch_size = 14
        elif "32" in args.backbone:
            patch_size = 32
        else:
            patch_size = 14

        self.Dctpatch = DCTPatches( 
            window_size=patch_size, 
            stride=patch_size, 
            grade_N=6, 
            num_select_rate=args.num_select_rate
        )

        self.clip_model, _ = clip.load(self.args.backbone, device=self.device)
        self.clip_model = self.clip_model.float()
        self.clip_model.eval()

        self.args.vision_width = self.clip_model.visual.transformer.width
        self.args.text_width = self.clip_model.transformer.width
        self.args.embed_dim = self.clip_model.text_projection.shape[1]

        self.loss_ort_function=Orthogonal_Loss()


        list_lora_layers = apply_lora(self.args, self.clip_model)
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(self.device)
        self.indices = INDEX_POSITIONS_VISION[self.args.backbone][self.args.lora_position]
       

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.pro_image=None


        self.class_mapping=nn.Linear(self.args.text_width,self.args.text_width)
        self.context_prompting = Context_Prompting(self.args)
        self.text_encoder_Bayes = TextEncoder_Bayes_V4(self.clip_model,self.args)
        self.temperature_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def patch_contrastive_loss(self,image,topk_indices,bottomk_indices, margin=1.0):
        """
        top_emb: Tensor of shape (N, K, D), embeddings of Top-k patches.
        bottom_emb: Tensor of shape (N, M, D), embeddings of Bottom-k patches.
        margin: Margin for the contrastive loss.
        """
        # image: (bs,256,D)
        image=image[1:,:,:]  # 排除class_embedding
        image=image.permute(1,0,2)
        D = image.size(-1)
        top_emb=torch.gather(image,dim=1,index=topk_indices.unsqueeze(-1).expand(-1,-1,D))
        bottom_emb=torch.gather(image,dim=1,index=bottomk_indices.unsqueeze(-1).expand(-1,-1,D))
        N, K, D = top_emb.shape
        M = bottom_emb.shape[1]
        device = image.device

        anchor_emb = top_emb[:, 0, :]  # (N,1,D)

        # 合并Top和Bottom嵌入
        all_emb = torch.cat([top_emb, bottom_emb], dim=1)  # (N, K+M, D)

        diff = all_emb - anchor_emb.unsqueeze(1)
        d_squared = torch.sum(diff**2, dim=-1)  # (N, K+M)
        labels = torch.zeros(N, K+M, device=device)
        labels[:, 1:K] = 1  # 排除anchor自身，仅标记其他Top-k为正样本
        # 生成mask排除anchor自身（索引0位置）
        mask = torch.ones(N, K+M, dtype=torch.bool, device=device)
        mask[:, 0] = False

        # 计算对比损失项
        pos_loss = labels * d_squared  # 正样本损失
        neg_loss = (1 - labels) * torch.clamp(margin - d_squared, min=0)  # 负样本损失

        # 合并损失并应用mask
        total_loss_per_sample = (pos_loss + neg_loss) * mask
        total_loss = total_loss_per_sample.sum(dim=1).mean()  # 按样本平均

        return total_loss
    

    def encode_image_lora(self, x: torch.Tensor,mode='train'):
        x = x.float()
        if mode=='train':
            topk_indices, bottomk_indices = self.Dctpatch(x)

        visual = self.clip_model.visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        contrastive_loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(visual.transformer.resblocks):
            x = block(x)
            if mode=='train' and i in self.indices:
                contrastive_loss += self.patch_contrastive_loss(x, topk_indices, bottomk_indices, margin=1.0)
        if len(self.indices) > 0:
            contrastive_loss = contrastive_loss / len(self.indices)
        x = x.permute(1, 0, 2)
        x = visual.ln_post(x[:, 0, :])  # 取第一个token作为全局特征
        x=x@visual.proj
        return x,contrastive_loss

    def forward(self, image,labels,mode='test'):
        image_features,contrastive_loss=self.encode_image_lora(image.type(self.dtype),mode=mode)
        # image_features=self.clip_model.encode_image(image.type(self.dtype))
        text_features,loss_dist=self.context_prompting(self.text_encoder_Bayes,image_features,mode=mode) #bs,num_prompt*2,width
        text_embeddings_mapping = self.class_mapping(text_features)
        text_embeddings_mapping = text_embeddings_mapping / text_embeddings_mapping.norm(dim = -1, keepdim = True)
        image_embeddings_mapping = image_features
        image_embeddings_mapping = image_embeddings_mapping / image_embeddings_mapping.norm(dim=-1, keepdim = True)
        pro_img = self.temperature_image.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)
        self.pro_image=pro_img
        if mode=='train' :
            pro_img = pro_img.squeeze(2)
            index_prompt = torch.randint(0,self.args.prompt_num, (1,1))[0]
            pro_img = torch.cat([pro_img[:,index_prompt], pro_img[:,index_prompt + self.args.prompt_num]], dim = 1)

            ort_function=self.loss_ort_function(text_features,self.args)
            kl_total = loss_dist[1] + loss_dist[2] + loss_dist[3] + loss_dist[4]
            losses = {
                'rec': loss_dist[0],  # 重构损失
                'kl': kl_total,       # KL散度总和
                'contrastive': contrastive_loss, # 对比损失
                'ort':ort_function
            }
            return pro_img,losses
        else:
            N = self.args.sample_num*self.args.prompt_num
            pro_img = pro_img.squeeze(2)  # (bs, 2N)

            pos = pro_img[:, :N]
            neg = pro_img[:, N:]
            pairs = torch.stack([pos, neg], dim=2)  # (bs, N, 2)
            probs = torch.softmax(pairs, dim=2)  # (bs, N, 2)
            pos_score = probs[:, :, 0].mean(dim=1)  # 对 N 个 prompt 的预测概率取平均
            neg_score = probs[:, :, 1].mean(dim=1) 
            # 拼成 (bs, 2)
            outputs = torch.stack([pos_score, neg_score], dim=1)  # (bs, 2)
            return outputs,torch.tensor(0.0, device=image.device)

