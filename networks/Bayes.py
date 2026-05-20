import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import pdb

from networks.PFL import PlanarPFL, PlanarPFL_learnable





class Orthogonal_Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Orthogonal_Loss, self).__init__()
        self.epsilon = epsilon
    
    def compute_orthogonal_loss(self, embeddings):
        B, L, C = embeddings.shape
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        cosine_sim_matrix = torch.einsum('blc,bkc->blk', embeddings_norm, embeddings_norm)
        cosine_sim_squared = cosine_sim_matrix ** 2
        eye_mask = torch.eye(L, device=embeddings.device).unsqueeze(0)
        cosine_sim_squared = cosine_sim_squared * (1 - eye_mask)
        cosine_loss = cosine_sim_squared.mean()
        return cosine_loss
    def forward(self, embeddings, args):
        Loss_noraml_text = self.compute_orthogonal_loss(embeddings[:, 0:args.prompt_num,:])
        Loss_abnormal_text = self.compute_orthogonal_loss(embeddings[:,args.prompt_num:,:])
        orthogonal_loss = Loss_noraml_text + Loss_abnormal_text
        return orthogonal_loss
def log_normal_dist(x, mean, logvar, dim):
    log_norm = -0.5 * (logvar + (x - mean) * (x - mean) * logvar.exp().reciprocal()) 

    return torch.sum(log_norm, dim)
    
def binary_loss_function(x_recon, x, z_mu, z_var, z_0, z_k, log_det_jacobians, z_size, cuda, beta=1, summ = True, log_vamp_zk = None, if_rec = True):
    """
    z_mu: mean of z_0
    z_var: variance of z_0
    z_0: first stochastic(随机) latent variable 
    z_k: last stochastic latent variable
    log_det_jacobians: log det jacobian
    beta: beta for annealing according to Equation 20
    log_vamp_zk: default None but log_p(zk) if VampPrior used
    the function returns: Free Energy Bound (ELBO), reconstruction loss, kl
    """
    batch_size = x.size(0) 
    logvar=torch.zeros(batch_size, z_size).to(x.device)    
        
    # calculate log_p(zk) under standard Gaussian unless log_p(zk) under VampPrior given
    if log_vamp_zk is None:
        log_p_zk = log_normal_dist(z_k, mean=0, logvar=logvar, dim=1) # ln p(z_k) = N(0,I) log_normal_dist高斯分布的对数概率密度函数
    else:
        log_p_zk = log_vamp_zk
    
                 
    log_q_z0 = log_normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)

    log_p_zk = log_p_zk + 1e-8
    log_q_z0 = log_q_z0 + 1e-8
    
    if (summ == True):  ## Computes the binary loss function with summing over batch dimension 
        
        #Reconstruction loss: Binary cross entropy
        reconstruction_loss = nn.MSELoss(reduction='sum')

        if if_rec:
            log_p_xz = reconstruction_loss(x_recon, x)  #-log_p(x|z_k)
        else:
            log_p_xz = 0
        log_p_xz = log_p_xz
        kl = torch.sum(log_q_z0 - log_p_zk) - torch.sum(log_det_jacobians) #sum over batches
        #elbo = elbo / batch_size
        log_p_xz = log_p_xz / batch_size
        kl = kl / batch_size

        elbo = 0
        
        return elbo, log_p_xz, kl #elbo变分下界
    
    else:              ## Computes the binary loss function without summing over batch dimension (used during testing) 
        if len(log_det_jacobians.size()) > 1:
            log_det_jacobians = log_det_jacobians.view(log_det_jacobians.size(0), -1).sum(-1)

        reconstruction_loss = nn.BCELoss(reduction='none')
        log_p_xz = reconstruction_loss(x_recon.view(batch_size, -1), x.view(batch_size, -1))  #-log_p(x|z_k)
        log_p_xz = torch.sum(log_p_xz, dim=1)
        
        #Equation (20)
        elbo = log_q_z0 - log_p_zk - log_det_jacobians + log_p_xz 

        return elbo, log_p_xz, (log_q_z0 - log_p_zk - log_det_jacobians)
    
class InferenceBlock(nn.Module):
    def __init__(self, input_units, d_theta, output_units):
        """
        :param d_theta: dimensionality of the intermediate hidden layers.
        :param output_units: dimensionality of the output.
        :return: batch of outputs.
        """
        super(InferenceBlock, self).__init__()
        self.module = nn.Sequential(
            #nn.Linear(input_units, output_units, bias=True),
            nn.Linear(input_units, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, d_theta, bias=True),
            nn.Softplus(),
            nn.Linear(d_theta, output_units, bias=True),
        )

    def forward(self, inps):
        out = self.module(inps)
        return out
class Encoder(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Encoder, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)
        self.weight_log_variance = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps) #输入维度1024
        weight_log_variance = self.weight_log_variance(inps)
        return weight_mean, torch.exp(0.5 * weight_log_variance)
    
class Decoder(nn.Module):
    def __init__(self, input_units=400, d_theta=400, output_units=400):
        super(Decoder, self).__init__()
        self.output_units = output_units
        self.weight_mean = InferenceBlock(input_units, d_theta, output_units)

    def forward(self, inps):
        weight_mean = self.weight_mean(inps)
        return weight_mean


class TextEncoder_Bayes_V2(nn.Module):

    def __init__(self, clip_model, args):
        super().__init__()
        self.clip_model = clip_model
        self.context_length = clip_model.context_length
        self.prompt_private_len=args.prompt_private_len
        self.prompt_num= args.prompt_num 
        self.sample_num=args.sample_num 
        self.prompt_share_len=args.prompt_share_len

    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    def _build_pseudo_tokens(self, visual_feature_len, batch_size, device):
        # 获取嵌入维度
        embed_dim = self.clip_model.ln_final.weight.shape[0]
        
        # 构建伪文本序列：[SOS] + 视觉特征位置的填充 + [EOS] + 剩余填充
        sos = self.clip_model.token_embedding(torch.tensor(49406, device=device)).unsqueeze(0)  # (1, embed_dim)
        eos = self.clip_model.token_embedding(torch.tensor(49407, device=device)).unsqueeze(0)  # (1, embed_dim)
        visual_placeholders = torch.zeros(visual_feature_len, embed_dim, device=device)        # (visual_feature_len, embed_dim)

        tokens = [sos, visual_placeholders, eos]

        # 如果总长度不足 context_length，添加 padding
        current_len = 1 + visual_feature_len + 1  # SOS + visual + EOS
        if current_len < self.context_length:
            padding_len = self.context_length - current_len
            padding = torch.zeros(padding_len, embed_dim, device=device)  # (padding_len, embed_dim)
            tokens.append(padding)

        # 拼接所有部分   
        pseudo_tokens = torch.cat(tokens, dim=0)  # (context_length, embed_dim)
        return pseudo_tokens.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, context_length, embed_dim)

    def forward(self, visual_feature, prompt_bias_list, mode):  
        prompt_bias_private = prompt_bias_list[0]
        prompt_bias_class = prompt_bias_list[1]
        prompt_bias_share= prompt_bias_list[2]
        # 计算批大小和视觉特征长度
        device = visual_feature.device
        dtype = visual_feature.dtype
        embed_dim = visual_feature.shape[-1]
        visual_feature_len = visual_feature.shape[1]
        eos_index = 1 + visual_feature_len  # EOS位置索引
        if mode == "train":
            batch_size=prompt_bias_private.shape[0]
            sos = self.clip_model.token_embedding(torch.tensor([[49406]], device=device)).expand(batch_size*self.prompt_num, -1, -1)
            eos = self.clip_model.token_embedding(torch.tensor([[49407]], device=device)).expand(batch_size*self.prompt_num, -1, -1)

            visual_feature_new = visual_feature.repeat(batch_size,1,1).clone() # shape (prompt_num,prompt_context_len+prompt_state_len,text_width 768)
            # 应用prompt_bias调整视觉特征
            visual_feature_new[:,:self.prompt_share_len,:]+=prompt_bias_share.reshape(1,1,-1)
            private_bias_expanded = prompt_bias_private.unsqueeze(1).expand(-1, self.prompt_num, -1)
            private_bias_large = private_bias_expanded.reshape(batch_size*self.prompt_num, 1, -1)
            visual_feature_new[:, self.prompt_share_len:self.prompt_private_len+self.prompt_share_len, :] += private_bias_large

            visual_feature_new[:, self.prompt_private_len+self.prompt_share_len:, :] += prompt_bias_class.reshape(1, 1, -1)

            x_new = torch.cat([sos, visual_feature_new, eos], dim=1) 
            current_len = x_new.shape[1]
            padding_len = self.context_length - current_len
            if padding_len > 0:
                padding = torch.zeros(x_new.shape[0], padding_len, embed_dim, device=device, dtype=dtype)
                x_new = torch.cat([x_new, padding], dim=1)

            #文本编码
            x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)
            x_new = x_new.permute(1, 0, 2)  # NLD -> LND
            x_new= self.clip_model.transformer(x_new)
            x_new = x_new.permute(1, 0, 2)  # LND -> NLD
            x_new = self.clip_model.ln_final(x_new).type(self.dtype)
            # 提取EOS位置的特征
            x_new = x_new[:, eos_index, :] @ self.clip_model.text_projection
            x_new_array = x_new.view(batch_size,self.prompt_num,-1)
        else:  # test模式
            batch_size=prompt_bias_private.shape[0]//self.sample_num
            total_squence=batch_size*self.prompt_num*self.sample_num
            
            sos = self.clip_model.token_embedding(torch.tensor([[49406]], device=device)).expand(total_squence, -1, -1)
            eos = self.clip_model.token_embedding(torch.tensor([[49407]], device=device)).expand(total_squence, -1, -1)


            visual_feature_large = visual_feature.repeat(batch_size * self.sample_num, 1, 1)
            visual_feature_new = visual_feature_large.clone() 
            
            share_bias_aligned = prompt_bias_share.repeat(batch_size, 1)
            share_bias_expanded = share_bias_aligned.unsqueeze(1).expand(-1, self.prompt_num, -1)
            share_bias_large = share_bias_expanded.reshape(-1, 1, share_bias_aligned.shape[-1])

            visual_feature_new[:,:self.prompt_share_len,:]+=share_bias_large

            private_bias_large = prompt_bias_private.unsqueeze(1).expand(-1,self.prompt_num, -1)
            private_bias_large = private_bias_large.reshape(-1,1,prompt_bias_private.shape[-1])
            visual_feature_new[:, self.prompt_share_len:self.prompt_private_len+self.prompt_share_len, :] += private_bias_large

            class_bias_aligned = prompt_bias_class.repeat(batch_size, 1)
            class_bias_expanded = class_bias_aligned.unsqueeze(1).expand(-1, self.prompt_num, -1)
            class_bias_large = class_bias_expanded.reshape(-1, 1, class_bias_aligned.shape[-1])
            visual_feature_new[:, self.prompt_private_len+self.prompt_share_len:, :] += class_bias_large

            x_new = torch.cat([sos, visual_feature_new, eos], dim=1) 
            current_len = x_new.shape[1]
            padding_len = self.context_length - current_len
            if padding_len > 0:
                padding = torch.zeros(x_new.shape[0], padding_len, embed_dim, device=device, dtype=dtype)
                x_new = torch.cat([x_new, padding], dim=1)

            #文本编码
            x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)
            x_new = x_new.permute(1, 0, 2)  # NLD -> LND
            x_new= self.clip_model.transformer(x_new)
            x_new = x_new.permute(1, 0, 2)  # LND -> NLD
            x_new = self.clip_model.ln_final(x_new).type(self.dtype)
            # 提取EOS位置的特征
            x_new = x_new[:, eos_index, :] @ self.clip_model.text_projection
            x_new_array = x_new.view(batch_size,self.sample_num*self.prompt_num,-1)
        return x_new_array


class TextEncoder_Bayes_V4(nn.Module):

    def __init__(self, clip_model, args):
        super().__init__()
        self.clip_model = clip_model
        self.num_G_S_tokens = args.prompt_private_len+args.prompt_share_len
        self.prompt_share_len=args.prompt_share_len
        self.context_length = clip_model.context_length
        self.prompt_num= args.prompt_num 
        self.sample_num=args.sample_num 
        self.last_visual_feature_new = None

    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    def _build_pseudo_tokens(self, visual_feature_len, batch_size, device):

        embed_dim = self.clip_model.ln_final.weight.shape[0]

        sos = self.clip_model.token_embedding(torch.tensor(49406, device=device)).unsqueeze(0)  # (1, embed_dim)
        eos = self.clip_model.token_embedding(torch.tensor(49407, device=device)).unsqueeze(0)  # (1, embed_dim)
        visual_placeholders = torch.zeros(visual_feature_len, embed_dim, device=device)        # (visual_feature_len, embed_dim)

        tokens = [sos, visual_placeholders, eos]

        current_len = 1 + visual_feature_len + 1  # SOS + visual + EOS
        if current_len < self.context_length:
            padding_len = self.context_length - current_len
            padding = torch.zeros(padding_len, embed_dim, device=device)  # (padding_len, embed_dim)
            tokens.append(padding)

  
        pseudo_tokens = torch.cat(tokens, dim=0)  # (context_length, embed_dim)
        return pseudo_tokens.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, context_length, embed_dim)

    def forward(self, visual_feature, prompt_bias_list, mode): 
        prompt_bias_private = prompt_bias_list[0]
        prompt_bias_class = prompt_bias_list[1]
        prompt_bias_share= prompt_bias_list[2]
        if mode == "train":
            batch_size = prompt_bias_private.shape[0]
        else:
            batch_size = prompt_bias_private.shape[0] // self.sample_num
        visual_feature_len = visual_feature.shape[1]
        x_pseudo = self._build_pseudo_tokens(visual_feature_len, batch_size, visual_feature.device).type(self.dtype)
        pos_y = 1
        eos_index = 1 + visual_feature_len 
        if mode == "train":
            x_new_temp = torch.zeros((batch_size, self.prompt_num,self.context_length, self.clip_model.text_projection.shape[1]),
                                     dtype=self.dtype, device=visual_feature.device)
            for i in range(batch_size):
                visual_feature_new = visual_feature.clone() 
                visual_feature_new[:,:self.prompt_share_len,:]+=prompt_bias_share.reshape(1,1,-1)
                visual_feature_new[:, self.prompt_share_len:self.num_G_S_tokens, :] += prompt_bias_private[i].reshape(1, 1, -1) # prompt_bias.shape(bs,embed_dim)
                visual_feature_new[:, self.num_G_S_tokens:, :] += prompt_bias_class.reshape(1, 1, -1)

                x_temp = x_pseudo[i].unsqueeze(0).expand(visual_feature.shape[0], -1, -1)
                x_new = torch.cat([
                    x_temp[:, :pos_y, :],              
                    visual_feature_new,                
                    x_temp[:, pos_y + visual_feature_len:, :] 
                ], dim=1)
                x_new_temp[i] = x_new
        else: 
            x_new_temp = torch.zeros((batch_size, self.prompt_num* self.sample_num,self.context_length,
                                       self.clip_model.text_projection.shape[1]),
                                      dtype=self.dtype, device=visual_feature.device)
            for i in range(batch_size):
                text_feature_list = []
                for j in range(prompt_bias_class.shape[0]):
                    visual_feature_new = visual_feature.clone() #shape (prompt_num,prompt_context_len+ prompt_state_len,text_width 768)

                    visual_feature_new[:,:self.prompt_share_len,:]+=prompt_bias_share[j].reshape(1,1,-1)
                    visual_feature_new[:, self.prompt_share_len:self.num_G_S_tokens, :] += prompt_bias_private[i*self.sample_num+j].reshape(1, 1, -1) # prompt_bias.shape(bs * sample_num,embed_dim)
                    visual_feature_new[:, self.num_G_S_tokens:, :] += prompt_bias_class[j].reshape(1, 1, -1) #prompt_bias_state.shape(sample_num,embed_dim)

                    x_temp = x_pseudo[i].unsqueeze(0).expand(visual_feature.shape[0], -1, -1)
                    x_new = torch.cat([
                        x_temp[:, :pos_y, :],
                        visual_feature_new,
                        x_temp[:, pos_y + visual_feature_len:, :]
                    ], dim=1)
                    text_feature_list.append(x_new)
                x_new = torch.cat(text_feature_list, dim=0)
                x_new_temp[i] = x_new
        
        x_new_temp=x_new_temp.reshape(-1,self.context_length,self.clip_model.text_projection.shape[1])
        self.last_visual_feature_new=x_new_temp

        x_new_temp = x_new_temp + self.clip_model.positional_embedding.type(self.dtype)
        x_new_temp = x_new_temp.permute(1, 0, 2)
        x_new_temp= self.clip_model.transformer(x_new_temp)
        x_new_temp = x_new_temp.permute(1, 0, 2)
        x_new_temp = self.clip_model.ln_final(x_new_temp).type(self.dtype)

        x_new_temp = x_new_temp[:, eos_index, :] @ self.clip_model.text_projection
        x_new_array = x_new_temp.reshape(batch_size,-1,self.clip_model.text_projection.shape[1])
        return x_new_array
    

class Context_Prompting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_width = args.vision_width   
        self.text_width = args.text_width  
        self.embed_dim = args.embed_dim     

       
        self.prompt_private = nn.Parameter(torch.randn(args.prompt_num, args.prompt_private_len, self.text_width))  # Constructing the learnable private vectors in the original prompt repository.
        self.prompt_real_class = nn.Parameter(torch.randn(args.prompt_num, args.prompt_class_len, self.text_width)) # Constructing the learnable real class vectors in the original prompt repository.
        self.prompt_fake_class = nn.Parameter(torch.randn(args.prompt_num, args.prompt_class_len, self.text_width)) # Constructing the learnable fake class vectors in the original prompt repository.
        self.prompt_share=nn.Parameter(torch.randn(args.prompt_num,args.prompt_share_len, self.text_width))  # Constructing the learnable share vectors in the original prompt repository.

        self.temperature_pixel = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        

        self.private_encoder = Encoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  

        self.private_decoder = Decoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  

        self.class_encoder = Encoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )

        self.class_decoder = Decoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )  

        self.share_encoder = Encoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )

        self.share_decoder = Decoder(
            input_units= self.embed_dim, d_theta= self.vision_width // 2, output_units= self.embed_dim 
        )
        self.PFL_share= PlanarPFL_learnable(self.share_encoder, self.share_decoder, args) # For image-generate distribution
        self.PFL_private = PlanarPFL(self.private_encoder, self.private_decoder, args) # For image-specify distribution
        self.PFL_real = PlanarPFL_learnable(self.class_encoder, self.class_decoder, args) # For image-class distribution
        self.PFL_fake = PlanarPFL_learnable(self.class_encoder, self.class_decoder, args)  # For image-class distribution



        self.class_mapping = nn.Linear(self.text_width, self.text_width)
        self.image_mapping = nn.Linear(self.text_width, self.text_width)
        self._initialize_weights()

        
        nn.init.trunc_normal_(self.prompt_private, mean=0, std=0.02)
        nn.init.trunc_normal_(self.prompt_real_class, mean=0.5, std=0.02)
        nn.init.trunc_normal_(self.prompt_fake_class, mean=-0.5, std=0.02)
        nn.init.trunc_normal_(self.prompt_share, mean=0, std=0.02)


        self.zk_real = None
        self.zk_fake = None
        self.zk_private=None
        self.zk_share=None

    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    



    def forward(self, mode_text_encode, global_img_feature,mode = "train"):
        beta = 0
        global_img = global_img_feature

        
        x_mean_private, z_mu_private, z_var_private, log_det_jacobians_private, z0_private, zk_private = self.PFL_private(global_img.clone(), mode = mode)
        x_mean_real, z_mu_real, z_var_real, log_det_jacobians_real, z0_real, zk_real = self.PFL_real(global_img.clone(), mode = mode)
        x_mean_fake, z_mu_fake, z_var_fake, log_det_jacobians_fake, z0_fake, zk_fake = self.PFL_fake(global_img.clone(), mode = mode)
        x_mean_share, z_mu_share, z_var_share, log_det_jacobians_share, z0_share, zk_share = self.PFL_share(global_img.clone(), mode = mode)
        


        if self.zk_real is None and self.zk_fake is None and mode != "train":  # 
            self.zk_real = zk_real
            self.zk_fake = zk_fake
        if mode != "train":
            zk_real = self.zk_real
            zk_fake = self.zk_fake

        # self.zk_private=zk_private
        # self.zk_share=zk_share
        # self.zk_real = zk_real
        # self.zk_fake = zk_fake
        
        


        #Compute the loss 
        if mode == "train":
            loss, rec_share, kl_share = binary_loss_function(x_mean_share, global_img, z_mu_share, z_var_share, z0_share, zk_share, log_det_jacobians_share,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = False)
            loss, rec, kl = binary_loss_function(x_mean_private, global_img, z_mu_private, z_var_private, z0_private, zk_private, log_det_jacobians_private,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = True)
            
            loss, rec_n, kl_n = binary_loss_function(x_mean_real, global_img, z_mu_real, z_var_real, z0_real, zk_real, log_det_jacobians_real,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = False)

            loss, rec_a, kl_a = binary_loss_function(x_mean_fake, global_img, z_mu_fake, z_var_fake, z0_fake, zk_fake, log_det_jacobians_fake,
                                                self.args.embed_dim, True, beta = beta, log_vamp_zk = None, if_rec = False)

        else:
            kl_share = 0
            loss = 0
            rec=0
            kl=0
            kl_n  = 0
            kl_a = 0
        context_real= torch.cat([self.prompt_share,self.prompt_private , self.prompt_real_class], dim = 1)
        context_fake=torch.cat([self.prompt_share,self.prompt_private , self.prompt_fake_class], dim = 1)
        real_embeddings = mode_text_encode( context_real, [zk_private,zk_real,zk_share], mode = mode) 
        fake_embeddings = mode_text_encode(context_fake, [zk_private,zk_fake,zk_share], mode = mode) 

        text_embeddings = torch.cat([real_embeddings, fake_embeddings], dim =1)
        text_embeddings = text_embeddings / text_embeddings.norm(dim = -1,keepdim = True)
        return text_embeddings, [rec, kl ,kl_n , kl_a, kl_share]
        # return text_embeddings,[rec,kl]
    

