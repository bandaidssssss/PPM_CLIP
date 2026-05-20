import io
import os

import torch
import torchvision
from loguru import logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = {
    "genimage": {
        "train_root": os.path.join("../../datasets/Genimages_SD-V1.4", "train"),
        "val_root": os.path.join("../../datasets/Genimages_SD-V1.4", "val"),
        "test_root": "../../datasets/GenImage",
        "vals": [
            "Midjourney",
            "stable_diffusion_v_1_4",
            "stable_diffusion_v_1_5",
            "ADM",
            "glide",
            "wukong",
            "VQDM",
            "BigGAN",
        ],
    },
    "ojha": {
        "train_root": os.path.join("../../datasets/ForenSynths_4classtrain_val_test", "train"),
        "val_root": os.path.join("../../datasets/ForenSynths_4classtrain_val_test", "val"),
        "test_root": "../../AIGC-Detector/datasets/UniversalFakeDetect_test",
        "vals": [
            "dalle",
            "glide_100_10",
            "glide_100_27",
            "glide_50_27",
            "guided",
            "ldm_100",
            "ldm_200",
            "ldm_200_cfg",
        ],
    },
}

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


class ForenSynths(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["0_real", "1_fake"]
        self.data = []

        from concurrent.futures import ThreadPoolExecutor

        def process_path(root, filename):
            file_path = os.path.join(root, filename)
            if "0_real" in file_path:
                return (file_path, 0)
            if "1_fake" in file_path:
                return (file_path, 1)
            return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for root, _, files in os.walk(self.root_dir):
                for filename in files:
                    futures.append(executor.submit(process_path, root, filename))

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
            return torch.zeros(3, 224, 224), label


def judge_img(img, load_size):
    img_width, img_height = img.size
    if img_width < load_size or img_height < load_size:
        img = torchvision.transforms.Resize(
            (load_size, load_size), interpolation=InterpolationMode.BILINEAR
        )(img)
    return img


def jpeg_compression(img, quality=50):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def train_augment(normalize, load_size):
    transform_list = [
        transforms.Lambda(lambda img: judge_img(img, load_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[normalize], std=STD[normalize]),
    ]
    return transforms.Compose(transform_list)


def test_augment(normalize, load_size):
    transform_list = [
        transforms.Lambda(lambda img: judge_img(img, load_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[normalize], std=STD[normalize]),
    ]
    return transforms.Compose(transform_list)


def create_dataloaders(args):
    dataset_cfg = DATASETS[args.dataset]
    train_root = dataset_cfg["train_root"]
    val_root = dataset_cfg["val_root"]
    test_root = dataset_cfg["test_root"]
    vals = dataset_cfg["vals"]

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    dl_train = torch.utils.data.DataLoader(
        ForenSynths(train_root, train_augment(args.normalize, args.loadSize)),
        shuffle=True,
        **loader_kwargs,
    )
    dl_val = torch.utils.data.DataLoader(
        ForenSynths(val_root, train_augment(args.normalize, args.loadSize)),
        shuffle=False,
        **loader_kwargs,
    )
    dl_test = {}
    for val in vals:
        test_dir = os.path.join(test_root, val)
        dl_test[val] = torch.utils.data.DataLoader(
            ForenSynths(test_dir, test_augment(args.normalize, args.loadSize)),
            shuffle=False,
            **loader_kwargs,
        )
    return dl_train, dl_val, dl_test


def create_test_dataloaders(args):
    dataset_cfg = DATASETS[args.dataset]
    test_root = dataset_cfg["test_root"]
    vals = dataset_cfg["vals"]

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    dl_test = {}
    for val in vals:
        test_dir = os.path.join(test_root, val)
        dl_test[val] = torch.utils.data.DataLoader(
            ForenSynths(test_dir, test_augment(args.normalize, args.loadSize)),
            shuffle=False,
            **loader_kwargs,
        )
    return dl_test
