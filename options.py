import argparse
import os

from loguru import logger


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):
        parser.add_argument("--num_workers", type=int, default=5, help="method name")
        parser.add_argument("--normalize", type=str, default="clip")
        parser.add_argument("--loadSize", type=int, default=256)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--ckpt_path", type=str, default=".pt")
        parser.add_argument("--dataset", type=str, default="genimage", help="try")
        parser.add_argument(
            "--gpu",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )

        parser.add_argument("--backbone", type=str, default="ViT-L/14", help="lora backbone")

        # clip_lora
        parser.add_argument("--lora_encoder", type=str, default="vision", help="lora encoder")
        parser.add_argument("--lora_position", type=str, default="half-up", help="lora position")
        parser.add_argument("--lora_alpha", type=float, default=0.5, help="lora alpha")
        parser.add_argument("--lora_r", type=int, default=4, help="lora r")
        parser.add_argument("--lora_params", type=str, default="qkv", help="lora params")
        parser.add_argument("--lora_dropout_rate", type=float, default=0.0, help="lora dropout rate")

        # text prompt params
        parser.add_argument("--prompt_share_len", type=int, default=3, help="prompt share len")
        parser.add_argument("--prompt_private_len", type=int, default=7, help="prompt private len")
        parser.add_argument("--prompt_class_len", type=int, default=10, help="prompt class len")
        parser.add_argument("--prompt_num", type=int, default=2, help="prompt repository number")
        parser.add_argument("--num_flows", type=int, default=10, help="num flows in prompt flow module")
        parser.add_argument("--sample_num", type=int, default=10, help="sample num in test")

        parser.add_argument("--batch_size", type=int, default=48)
        parser.add_argument("--optim_type", type=str, default="Adam")
        parser.add_argument("--lr", type=float, default=0.0001)

        parser.add_argument("--num_select_rate", type=float, default=0.5, help="num select rate for DCT_patches")
        parser.add_argument("--lambda_rec", type=float, default=0.5, help="lambda for distillation loss")
        parser.add_argument("--lambda_kl", type=float, default=0.001, help="lambda for distillation loss")
        parser.add_argument("--lambda_contra", type=float, default=0.5, help="lambda for distillation loss")
        parser.add_argument("--lambda_ort", type=float, default=1.0, help="lambda for distillation loss")
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            self.parser = self.initialize(self.parser)
        if self.parser is None:
            raise ValueError("Parser not initialized")
        opt, _ = self.parser.parse_known_args()
        return opt

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        logger.info(message)

    def parse(self, print_options=True):
        os.makedirs("./results", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        opt = self.gather_options()
        self.opt = opt
        if print_options:
            self.print_options(opt)
        return self.opt
