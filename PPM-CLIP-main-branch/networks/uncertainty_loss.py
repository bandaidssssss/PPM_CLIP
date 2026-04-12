import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    """
    不确定性加权损失模块 (Uncertainty Weighting)
    
    该模块用于在多任务学习中自动平衡多个损失函数。
    它为每个任务学习一个不确定性参数 `log(σ²)`，并根据此参数
    动态地调整每个任务在总损失中的权重。

    Attributes:
        task_names (list of str): 用于标识每个任务的名称列表，方便调试和打印。
        log_sigma_sq (nn.Parameter): 可学习的参数，尺寸为(num_tasks,)，
                                     代表每个任务的不确定性的对数。
    """
    def __init__(self, task_names):
        """
        初始化不确定性加权损失模块。

        Args:
            task_names (list of str): 任务的名称列表。模块将为列表中的每个
                                      任务创建一个不确定性参数。
                                      例如: ['classification', 'reconstruction', 'kl', 'contrastive']
        """
        super().__init__()
        self.num_tasks = len(task_names)
        self.task_names = task_names
        
        # 关键: 为每个任务创建一个可学习的参数 log(σ²)。
        # 初始化为0，意味着初始时每个任务的不确定性 σ² 都为 exp(0) = 1，
        # 这提供了一个中性的起点。
        self.log_sigma_sq = nn.Parameter(torch.zeros(self.num_tasks, requires_grad=True))

    def forward(self, losses):
        """
        计算加权后的总损失。

        Args:
            losses (dict): 一个字典，键是任务名称，值是该任务的原始损失张量。
                           键必须与初始化时传入的 `task_names` 对应。
                           例如: {'classification': tensor(1.2), 'reconstruction': tensor(5.6), ...}

        Returns:
            torch.Tensor: 加权后的总损失，是一个标量张量，可以直接用于 .backward()。
        """
        if not isinstance(losses, dict) or sorted(losses.keys()) != sorted(self.task_names):
            raise ValueError(f"输入的损失字典键必须与初始化的任务名称完全匹配: {self.task_names}")

        total_loss = 0
        self.log_sigma_sq.data.clamp_(-10, 10)
        for i, name in enumerate(self.task_names):
            # 从学习的参数中恢复不确定性 σ²
            # precision = 1 / σ² = exp(-log_sigma_sq)
            precision = torch.exp(-self.log_sigma_sq[i])
            
            # 计算加权损失
            # loss_term = (1 / σ²) * raw_loss = precision * raw_loss
            # regularization_term = log(σ) = 0.5 * log(σ²) = 0.5 * log_sigma_sq
            # 总损失 = precision * raw_loss + 0.5 * log_sigma_sq
            weighted_loss = 0.5*precision * losses[name] + 0.5 * self.log_sigma_sq[i]
            
            total_loss += weighted_loss
            
        return total_loss

    def get_weights(self):
        """
        返回当前学到的每个任务的权重，方便监控。
        权重定义为 1 / σ²。
        """
        with torch.no_grad():
            weights = torch.exp(-self.log_sigma_sq)
        return {name: w.item() for name, w in zip(self.task_names, weights)}