
import torch
import torch.nn as nn
import numpy as np


def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class DCTPatches(nn.Module):

    def __init__(self, window_size=14, stride=14, grade_N=6,num_select_rate=0.5):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.grade_N = grade_N

        # 定义DCT矩阵及其转置
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        # 定义Unfold层
        self.unfold = nn.Unfold(kernel_size=window_size, stride=stride)

        # 初始化grade_filters
        self.grade_filters = nn.ModuleList([
            Filter(window_size, int(window_size * 2. / grade_N * i), int(window_size * 2. / grade_N * (i+1)), norm=True)
            for i in range(grade_N)
        ])

        self.num_select_rate=num_select_rate

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        num_select=(H//self.window_size) * (W//self.window_size)*self.num_select_rate
        num_select=int(num_select)
        # 分割图像为patches [B, C*K*K, L]
        patches = self.unfold(x)
        L = patches.shape[-1]

        K = self.window_size
        # 转换为 [B*L, C, K, K]
        patches_reshaped = patches.permute(0, 2, 1).reshape(B * L, C, K, K)

        # DCT变换
        dct_patches = torch.matmul(self._DCT_patch, torch.matmul(patches_reshaped, self._DCT_patch_T))

        # 计算每个patch的得分
        grade = torch.zeros(B * L, device=device)
        w, k_weight = 1, 2
        for i in range(self.grade_N):
            _x = torch.abs(dct_patches)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[i](_x)
            _x = torch.sum(_x, dim=(1, 2, 3))  # 求和得到每个patch的得分
            grade += w * _x
            w *= k_weight

        # 转换为批次得分 [B, L]
        grade = grade.view(B, L)

        # 选择每个样本的top-n索引
        _, topk_indices = torch.topk(grade, k=num_select, dim=1)  # [B, n]
        all_indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        mask.scatter_(1, topk_indices, True)
        remaining_indices = all_indices[~mask].view(B, L - num_select)
        # _, bottomk_indices = torch.topk(grade, k=num_select, dim=1, largest=False) 

        return topk_indices, remaining_indices
    


class DCTPatches_random(nn.Module):
    def __init__(self, window_size=14, stride=14, grade_N=6,num_select_rate=0.5):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.grade_N = grade_N

        # 定义DCT矩阵及其转置
        # self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        # self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        # 定义Unfold层
        self.unfold = nn.Unfold(kernel_size=window_size, stride=stride)

        # 初始化grade_filters
        # self.grade_filters = nn.ModuleList([
        #     Filter(window_size, int(window_size * 2. / grade_N * i), int(window_size * 2. / grade_N * (i+1)), norm=True)
        #     for i in range(grade_N)
        # ])

        self.num_select_rate=num_select_rate

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # 计算需要选择的patch数量
        num_select = int((H // self.window_size) * (W // self.window_size) * self.num_select_rate)
        
        # 分割图像为patches [B, C*K*K, L]
        patches = self.unfold(x)
        L = patches.shape[-1]  # 总patch数量

        # ===== 随机选择部分 =====
        # 生成随机索引矩阵 [B, L]
        rand_vals = torch.rand(B, L, device=device)
        # 随机选择topk索引 [B, num_select]
        _, topk_indices = torch.topk(rand_vals, k=num_select, dim=1)
        # 生成所有索引的完整集合
        all_indices = torch.arange(L, device=device).expand(B, -1)
        # 创建掩码并获取剩余索引
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        mask.scatter_(1, topk_indices, True)
        remaining_indices = all_indices[~mask].view(B, L - num_select)
        # ======================

        return topk_indices, remaining_indices