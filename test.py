import asyncio
import csv
import datetime
import os
import random
import sys

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

from data_loading import create_test_dataloaders
from networks.model_engine import PPM_clip
from options import BaseOptions
from validation import validate


def set_random_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


def test_epoch(model, dl_test, device, args, time_str):
    sample_dir = "./robustness"
    os.makedirs(sample_dir, exist_ok=True)

    accs, aps, aucs = [], [], []
    data_csv = []

    data_csv.append(["Parameters"])
    for k, v in sorted(vars(args).items()):
        data_csv.append([k, v])
    data_csv.append([])
    data_csv.append(["Dataset", "Accuracy", "AUC", "AP",  "r_acc", "f_acc"])

    with torch.no_grad():
        for test_type, loader in dl_test.items():
            acc, ap, r_acc, f_acc, y_true, y_pred = validate(model, loader, device)
            try:
                auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                auc = 0.0
                logger.warning(
                    f"{test_type}: only one class present in test set, AUC set to 0."
                )

            accs.append(acc)
            aps.append(ap)
            aucs.append(auc)

            data_csv.append(
                [test_type, acc * 100, auc * 100, ap * 100, r_acc * 100, f_acc * 100]
            )
            logger.info(
                f"({test_type:12}) Acc: {acc*100:.2f}; AUC: {auc*100:.2f}; AP: {ap*100:.2f}; "
                f" R_Acc: {r_acc*100:.2f}; F_Acc: {f_acc*100:.2f}"
            )

    mean_acc = np.array(accs).mean() * 100
    mean_auc = np.array(aucs).mean() * 100
    mean_ap = np.array(aps).mean() * 100


    logger.info(
        f"{'Mean':12} Acc: {mean_acc:.2f}; AUC: {mean_auc:.2f}; AP: {mean_ap:.2f}"
    )
    data_csv.append(["MEAN", mean_acc, mean_auc, mean_ap])

    async def write_csv():
        csv_filename = f"{args.dataset}_PPM_CLIP_eval_{time_str}.csv"
        csv_path = os.path.join(sample_dir, csv_filename)
        with open(csv_path, "w", newline="", encoding="utf-8") as file_:
            writer = csv.writer(file_, delimiter=",")
            writer.writerows(data_csv)
        logger.info(f"Results saved to: {csv_path}")

    asyncio.run(write_csv())
    return mean_acc / 100


if __name__ == "__main__":
    args = BaseOptions().parse()
    if not args.gpu or args.gpu == "-1":
        device = torch.device("cpu")
    else:
        gpu_id = args.gpu.split(",")[0]
        device = torch.device(f"cuda:{gpu_id}")

    set_random_seed()
    init_logger()
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    dl_test = create_test_dataloaders(args)

    model = PPM_clip(args).to(device)
    default_model_path = "checkpoints/genimage_PPM_CLIP_1021_0142.pt"
    model_path = args.ckpt_path if args.ckpt_path != ".pt" else default_model_path
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    model.eval()

    with torch.no_grad():
        test_epoch(model, dl_test, device, args, time_str)
