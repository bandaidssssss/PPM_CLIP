import os
import sys
import csv
import random
import datetime

from loguru import logger
import torch
from torch import nn
import numpy as np

from networks.model_engine import PPM_clip
import pdb
from torch.utils.tensorboard import SummaryWriter
from options import BaseOptions
from data_loading import create_dataloaders
from validation import validate

torch.cuda.empty_cache()


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


args = BaseOptions().parse()

if not args.gpu or args.gpu == "-1":
    device = torch.device("cpu")
else:
    gpu_id = args.gpu.split(",")[0]
    device = torch.device(f"cuda:{gpu_id}")


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
            acc, ap, r_acc, f_acc = validate(model, loader, device)[:4]
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


    # 使用梯度累积
    accumulation_steps = 2 
    global_step =0 # 累积2个批次的梯度

    logger.info("learnable:")
    count = 1
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f'{count}:{name}')
            count = count + 1

    # ======================================================================
    dl_train, dl_val, dl_test = create_dataloaders(args)
    optimizer = get_optimizer(model, args.optim_type, args.lr)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose='True'
    )

    # ======================================================================
    early_stopping = EarlyStopping()


    # ======================================================================
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (inputs, labels) in enumerate(dl_train):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            outputs, losses_dict = model(inputs, labels,mode='train')
            classification_loss =  nn.CrossEntropyLoss()(outputs, labels.long())
            total_loss = (classification_loss + 
                  args.lambda_rec * losses_dict['rec'] + 
                  args.lambda_kl * losses_dict['kl'] + 
                  args.lambda_contra * losses_dict['contrastive']+
                  args.lambda_ort*losses_dict['ort']) 
            

            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                writer.add_scalar('Loss/classification', classification_loss.item(), global_step)
                writer.add_scalar('Loss/Raw_rec', losses_dict['rec'].item(), global_step)
                writer.add_scalar('Loss/Raw_kl', losses_dict['kl'].item(), global_step)
                writer.add_scalar('Loss/Raw_contrastive', losses_dict['contrastive'].item(), global_step)
                writer.add_scalar('Loss/Raw_ort',losses_dict['ort'].item(),global_step)
                

                global_step += 1

            running_loss += total_loss.item() * inputs.size(0) * accumulation_steps

        train_loss = running_loss / len(dl_train.dataset)
        logger.info(f"epoch【{epoch}】--> epoch_loss= {train_loss}")

        model.eval()
        val_acc = validate(model, dl_val, device)[0]
        logger.info(f"epoch【{epoch}】--> val_acc = {100 * val_acc:.2f}%")

        scheduler.step(val_acc)
        
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
