import torch
import os

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version (linked with PyTorch): {torch.version.cuda}")

# 这条是最关键的，它会告诉你 import 的 torch 来自哪个文件路径
torch_path = os.path.dirname(torch.__file__)
print(f"PyTorch Path: {torch_path}")