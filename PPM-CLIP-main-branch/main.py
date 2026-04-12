import os
import argparse
from loguru import logger
import sys
import csv
from sklearn.metrics import accuracy_score, average_precision_score
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import random
import datetime
from networks.model_engine import PPM_clip
from networks.uncertainty_loss import UncertaintyLoss
import pdb
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EarlyStopping:
    def __init__(
        self,
        patience=3,
        delta=0
    ):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.delta = delta

    def __call__(self, score, model, args,time_str):
        save_path = os.path.join(
        "./checkpoints",
        f"{args.dataset}_PPM_CLIP_{time_str}{args.ckpt_path}"
        )

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score < self.best_score - self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping.")
        else:
            self.best_score = score
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


# ===============================================================================================
# ===============================================================================================
class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument('--num_workers', type=int,
                            default=5, help='method name')
        parser.add_argument('--normalize', type=str, default="clip", )
        parser.add_argument('--loadSize', type=int, default=256, )
        parser.add_argument('--epochs', type=int, default=100, )
        parser.add_argument('--ckpt_path', type=str, default=".pt")
        parser.add_argument('--dataset', type=str, default="genimage", help='try')
        parser.add_argument(
            '--gpu', type=str, default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
        )

        parser.add_argument("--backbone", type=str, default='ViT-L/14', help="lora backbone")

        #clip_lora
        parser.add_argument("--lora_encoder", type=str, default='vision', help="lora encoder")
        parser.add_argument("--lora_position", type=str, default='half-up', help="lora position")
        parser.add_argument("--lora_alpha", type=float, default=0.5, help="lora alpha")
        parser.add_argument("--lora_r", type=int, default=4, help="lora r")
        parser.add_argument("--lora_params", type=str, default='qkv', help="lora params")
        parser.add_argument("--lora_dropout_rate", type=float, default=0.0, help="lora dropout rate")

        #text prompt params
        parser.add_argument("--prompt_share_len", type=int, default=3, help="prompt share len")
        parser.add_argument("--prompt_private_len", type=int, default=7, help="prompt private len")
        parser.add_argument("--prompt_class_len",type=int,default=10,help="prompt class len")
        parser.add_argument("--prompt_num", type=int, default=2, help="prompt repository number")
        parser.add_argument("--num_flows", type=int, default=10, help="num flows in prompt flow module") 
        parser.add_argument("--sample_num", type=int, default=10, help="sample num in test")
        parser.add_argument("--embed_dim", type=int, default=768, help="embed dim")
        parser.add_argument("--vision_width", type=int, default=1024, help="vision width of clip ViT-L/14")
        parser.add_argument("--text_width", type=int, default=768, help="text width of clip ViT-L/14")


        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--optim_type", type=str, default='Adam')
        parser.add_argument("--lr", type=float, default=0.0001)

        parser.add_argument('--num_select_rate', type=float, default=0.5, help='num select rate for DCT_patches')
        parser.add_argument("--lambda_rec",type=float, default=0.5, help="lambda for distillation loss")
        parser.add_argument("--lambda_kl",type=float, default=0.001, help="lambda for distillation loss")
        parser.add_argument("--lambda_contra",type=float, default=0.5, help="lambda for distillation loss")
        parser.add_argument("--lambda_ort",type=float, default=1.0, help="lambda for distillation loss")
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(self.parser)
        # get the basic options
        if self.parser is None:
            raise ValueError("Parser not initialized")
        opt, unknown = self.parser.parse_known_args()
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        logger.info(message)

    def parse(self, print_options=True):
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        opt = self.gather_options()
        self.opt = opt
        if print_options:
            self.print_options(opt)
        return self.opt


args = BaseOptions().parse()
device = torch.device('cuda:{}'.format(
    args.gpu[0])) if args.gpu else torch.device('cpu')
# ===============================================================================================
datasets = {
    "genimage": {
        "train_root": os.path.join("/data/ys/wxy/deepfake/datasets/Genimages_SD-V1.4", "train"),
        "val_root": os.path.join("/data/ys/wxy/deepfake/datasets/Genimages_SD-V1.4", "val"),
        "test_root": "/data/ys/wxy/deepfake/datasets/GenImage",
        "vals": [
           'Midjourney', 'stable_diffusion_v_1_4',  'stable_diffusion_v_1_5', 'ADM', 'glide','wukong','VQDM','BigGAN']
    },
    "ojha": {
        "train_root":  os.path.join("/data/ys/wxy/AIGC-Detector/datasets/ForenSynths_4classtrain_val_test", "train"),
        "val_root":  os.path.join("/data/ys/wxy/AIGC-Detector/datasets/ForenSynths_4classtrain_val_test", "val"),
        "test_root": "/data/ys/wxy/AIGC-Detector/datasets/UniversalFakeDetect_test",
        "vals": ['dalle', 'glide_100_10', 'glide_100_27', 'glide_50_27', 'guided',
                 'ldm_100', 'ldm_200', 'ldm_200_cfg']
    },
}
train_root = datasets[args.dataset]["train_root"]
val_root = datasets[args.dataset]["val_root"]
test_root = datasets[args.dataset]["test_root"]
vals = datasets[args.dataset]["vals"]


class ForenSynths(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.data = []

        from concurrent.futures import ThreadPoolExecutor

        def process_path(root, filename):
            file_path = os.path.join(root, filename)
            if '0_real' in file_path:
                return (file_path, 0)
            if '1_fake' in file_path:
                return (file_path, 1)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for root, _, files in os.walk(self.root_dir):
                for filename in files:
                    futures.append(executor.submit(
                        process_path, root, filename))

            for future in futures:
                result = future.result()
                if result is not None:
                    self.data.append(result)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # 返回一个默认图像
            return torch.zeros(3, 224, 224), label


# ===============================================================================================
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}


def judge_img(img):
    img_width, img_height = img.size
    if (img_width < args.loadSize or img_height < args.loadSize):
        img = torchvision.transforms.Resize((args.loadSize, args.loadSize), interpolation=InterpolationMode.BILINEAR)(
            img)
    return img
def jpeg_compression(img, quality=50):
    """模拟JPEG压缩伪影"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def train_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        transforms.RandomCrop(224),
        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))
        # ], p=0.5),
        # transforms.RandomApply([
        #     transforms.Lambda(lambda img: jpeg_compression(img, quality=random.randint(30, 100)))
        # ], p=0.5),

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


def val_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        transforms.CenterCrop(224),

        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))
        # ], p=0.5),
        # transforms.RandomApply([
        #     transforms.Lambda(lambda img: jpeg_compression(img, quality=random.randint(30, 100)))
        # ], p=0.5),
        
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


def test_augment():
    transform_list = [
        transforms.Lambda(judge_img),
        transforms.CenterCrop(224),

        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))
        # ], p=0.5),
        # transforms.RandomApply([
        #     transforms.Lambda(lambda img: jpeg_compression(img, quality=random.randint(30, 100)))
        # ], p=0.5),
        
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN[args.normalize], std=STD[args.normalize]),
    ]
    return transforms.Compose(transform_list)


# ===============================================================================================
def set_random_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def validate(model, data_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            outputs, _ = model(in_tens,label,mode='test')
            y_pred.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


# ===============================================================================================

def create_dataloaders():
    dl_train = torch.utils.data.DataLoader(
        ForenSynths(train_root, train_augment()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # 使用pin_memory加速数据传输
        persistent_workers=True,  # 保持worker进程存活
        prefetch_factor=2  # 预加载因子
    )
    dl_val = torch.utils.data.DataLoader(
        ForenSynths(val_root, train_augment()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    dl_test = {}
    for v_id, val in enumerate(vals):
        test_dir = os.path.join(test_root, val)
        dl_test[val] = torch.utils.data.DataLoader(
            ForenSynths(test_dir, test_augment()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    return dl_train, dl_val, dl_test




def get_optimizer(model, optim_type, lr):
    if optim_type == 'Adam':
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999))
    elif optim_type == 'SGD':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999),
                                      weight_decay=0.0)
    else:
        raise ValueError(f"Invalid optimizer: {optim_type}")
    return optimizer




def test_epoch(model, dl_test,time_str):
    datasets = []
    accs = []
    aps = []
    r_accs = []
    f_accs = []
    data_csv = []
    # 添加参数信息
    data_csv.append(['Parameters'])
    for k, v in sorted(vars(args).items()):
        data_csv.append([k, v])
    data_csv.append([])  # 添加空行分隔
    # 添加性能指标
    data_csv.append(['Dataset', 'Accuracy', 'AP', 'r_acc', 'f_acc'])

    with torch.no_grad():
        for _, (test_type, loader) in enumerate(dl_test.items()):
            acc, ap, r_acc, f_acc = validate(model, loader)[:4]
            data_csv.append([test_type, acc * 100, ap *
                            100, r_acc * 100, f_acc * 100])
            datasets.append(test_type)
            accs.append(acc)
            aps.append(ap)
            r_accs.append(r_acc)
            f_accs.append(f_acc)

            logger.info("( {:12}) acc: {:.4f}; ap: {:.4f};  r_acc: {:.4f}, f_acc: {:.4f}".format(
                test_type, acc * 100, ap * 100, r_acc * 100, f_acc * 100))

    mean_acc = np.array(accs).mean() * 100
    mean_ap = np.array(aps).mean() * 100
    logger.info("({:10}) acc: {:.1f}; ap: {:.1f}".format(
        'Mean', mean_acc, mean_ap))
    data_csv.append(['MEAN', mean_acc, mean_ap])
    # 使用异步IO写入CSV
    import asyncio

    async def write_csv():
        with open(f'{os.path.join("./results", f"{args.dataset}_PPM_CLIP_{time_str}.csv")}', 'a',
                  newline='') as file_:
            writer = csv.writer(file_, delimiter=',')
            writer.writerows(data_csv)

    asyncio.run(write_csv())
    return mean_acc / 100  # 返回归一化的准确率


def train():
    # ======================================================================
    set_random_seed()
    init_logger()
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    writer = SummaryWriter(f'runs/PPM_CLIP_{args.dataset}_{time_str}')

    # ======================================================================
    model = PPM_clip(args)
    model = model.to(device)

    # task_names = ['classification', 'reconstruction', 'kl', 'contrastive','ort']
    # loss_balancer = UncertaintyLoss(task_names=task_names).to(device)

    # 使用梯度累积
    accumulation_steps = 2 
    global_step =0 # 累积2个批次的梯度

    logger.info("learnable:")
    count = 1
    total_params = 0
    trainable_params = 0
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f'{count}:{name}')
    #         count = count + 1
    # for name, param in loss_balancer.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f'{count}:{name}')
    #         count = count + 1

    # ======================================================================
    dl_train, dl_val, dl_test = create_dataloaders()
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': args.lr},
    #     {'params': loss_balancer.parameters(), 'lr': 1e-3} 
    # ])
    optimizer = get_optimizer(model, args.optim_type, args.lr)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose='True'
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    # optimizer, 
    # T_max=args.epochs, # 总 epoch 数
    # eta_min=1e-6       # 最小学习率
    # )

    # ======================================================================
    early_stopping = EarlyStopping()

    # test_acc = test_epoch(model, dl_test,time_str)


    # ======================================================================
    for epoch in range(1, args.epochs + 1):
        # 训练阶段


        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (inputs, labels) in enumerate(dl_train):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # 前向传播
            outputs, losses_dict = model(inputs, labels,mode='train')
            classification_loss =  nn.CrossEntropyLoss()(outputs, labels.long())
            # losses=losses_dict['kl']+losses_dict['contrastive']
            # total_loss=((1-args.lambda_contra)*classification_loss+args.lambda_contra*losses)/accumulation_steps
            total_loss = (classification_loss + 
                  args.lambda_rec * losses_dict['rec'] + 
                  args.lambda_kl * losses_dict['kl'] + 
                  args.lambda_contra * losses_dict['contrastive']+
                  args.lambda_ort*losses_dict['ort']) 

            # raw_losses = {
            #     'classification': classification_loss*2.0,
            #     'reconstruction': losses_dict['rec']/150.0,
            #     'kl': losses_dict['kl']/25000.0,
            #     'contrastive': losses_dict['contrastive']/750000.0,
            #     'ort':losses_dict['ort']*5000.0
            # }
            
            # 3. 使用 loss_balancer 计算最终的总损失
            # total_loss = loss_balancer(raw_losses)
            total_loss = total_loss / accumulation_steps
            # ---------------------------------------------------------
            # 反向传播
            total_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # ------ 监控与日志 (强烈建议) ------
                writer.add_scalar('Loss/classification', classification_loss.item(), global_step)
                writer.add_scalar('Loss/Raw_rec', losses_dict['rec'].item(), global_step)
                writer.add_scalar('Loss/Raw_kl', losses_dict['kl'].item(), global_step)
                writer.add_scalar('Loss/Raw_contrastive', losses_dict['contrastive'].item(), global_step)
                writer.add_scalar('Loss/Raw_ort',losses_dict['ort'].item(),global_step)
                

                # writer.add_scalar('Loss/Scaled_classification', raw_losses['classification'].item(), global_step)
                # writer.add_scalar('Loss/Scaled_reconstruction', raw_losses['reconstruction'].item(), global_step)
                # writer.add_scalar('Loss/Scaled_kl', raw_losses['kl'].item(), global_step)
                # writer.add_scalar('Loss/Scaled_contrastive', raw_losses['contrastive'].item(), global_step)
                # writer.add_scalar('Loss/Scaled_ort', raw_losses['ort'].item(), global_step)
                # # 监控学到的权重
                # current_weights = loss_balancer.get_weights()
                # for name, weight_val in current_weights.items():
                #     writer.add_scalar(f'Weights/{name}', weight_val, global_step)
                global_step += 1

            running_loss += total_loss.item() * inputs.size(0) * accumulation_steps

        train_loss = running_loss / len(dl_train.dataset)
        logger.info(f"epoch【{epoch}】--> epoch_loss= {train_loss}")
        # current_weights = loss_balancer.get_weights()
        # logger.info(f"Current learned weights: {current_weights}")

        # 验证阶段
        model.eval()
        val_acc = validate(model, dl_val)[0]
        logger.info(f"epoch【{epoch}】--> val_acc = {100 * val_acc:.2f}%")

        # 更新学习率
        scheduler.step(val_acc)

        # scheduler.step()

        if val_acc >= 0.90:
            test_acc = test_epoch(model, dl_test,time_str)
            logger.info(f"epoch【{epoch}】--> test_acc = {100 * test_acc:.2f}%")
            early_stopping(test_acc, model, args,time_str)
            if early_stopping.early_stop:
                writer.close()
                return
    writer.close()
    return


if __name__ == "__main__":
    train()