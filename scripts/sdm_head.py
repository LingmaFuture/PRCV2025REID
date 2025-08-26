# sdm_head.py
"""
SDM (Similarity Distribution Matching) 头模块
包含可学习的温度参数，确保数值稳定性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDMHead(nn.Module):
    """
    SDM头模块，包含可学习的温度参数
    
    特点：
    1. 可学习的logit_scale参数，等价于学习1/tau
    2. 自动L2归一化，确保余弦相似度计算
    3. clamp保证数值稳定性
    """
    
    def __init__(self, init_tau=0.12):
        """
        Args:
            init_tau (float): 初始温度参数，建议0.12起步
        """
        super().__init__()
        # logit_scale = log(1/tau)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))
        
        # 将 1/tau 限制在 [5, 20] -> tau ∈ [0.05, 0.2]
        self.min_logit_scale = math.log(5.0)   # tau = 0.2
        self.max_logit_scale = math.log(20.0)  # tau = 0.05
        
    def forward(self, q, g):
        """
        计算相似度矩阵
        
        Args:
            q (torch.Tensor): 查询特征 [N, D]
            g (torch.Tensor): 图库特征 [M, D]
            
        Returns:
            torch.Tensor: 相似度矩阵 [N, M]
        """
        # clamp 保证数值稳定（非原地操作）
        logit_scale = self.logit_scale.clamp(self.min_logit_scale, self.max_logit_scale)
        scale = logit_scale.exp()  # = 1/tau
        
        # L2归一化
        q = F.normalize(q, dim=1, eps=1e-6)
        g = F.normalize(g, dim=1, eps=1e-6)
        
        # 计算相似度矩阵
        S = (q @ g.t()) * scale
        
        return S
    
    def get_temperature(self):
        """获取当前温度参数"""
        logit_scale = self.logit_scale.clamp(self.min_logit_scale, self.max_logit_scale)
        return 1.0 / logit_scale.exp().item()
    
    def get_scale(self):
        """获取当前缩放因子"""
        logit_scale = self.logit_scale.clamp(self.min_logit_scale, self.max_logit_scale)
        return logit_scale.exp().item()


class SDMLossWithLearnableTemp(nn.Module):
    """
    带可学习温度的SDM损失函数
    """
    
    def __init__(self, init_tau=0.12, eps=1e-6):
        super().__init__()
        self.sdm_head = SDMHead(init_tau=init_tau)
        self.eps = eps
        
    def forward(self, qry_features, gal_features, labels, return_details=False):
        """
        计算SDM损失
        
        Args:
            qry_features (torch.Tensor): 查询特征 [N, D]
            gal_features (torch.Tensor): 图库特征 [M, D]
            labels (torch.Tensor): 身份标签 [N]
            return_details (bool): 是否返回详细信息
            
        Returns:
            torch.Tensor: SDM损失值
            dict (optional): 详细信息
        """
        N, D = qry_features.shape
        M, _ = gal_features.shape
        
        # 使用SDM头计算相似度矩阵
        S = self.sdm_head(qry_features, gal_features)  # [N, M]
        
        # 构造同身份指示矩阵
        labels_qry = labels.view(-1, 1)  # [N, 1]
        labels_gal = labels.view(1, -1)  # [1, M]
        y = (labels_qry == labels_gal).float()  # [N, M]
        
        # 行方向 (i2t) - 查询到图库
        logP_i2t = F.log_softmax(S, dim=1)  # [N, M]
        P_i2t = logP_i2t.exp()
        
        # 计算每行的正样本数量
        q_i2t_sum = y.sum(dim=1, keepdim=True)  # [N, 1]
        valid_i = (q_i2t_sum.squeeze(-1) > 0)  # [N]
        
        # 构造标签分布
        q_i2t = torch.zeros_like(y, dtype=S.dtype)
        q_i2t[valid_i] = y[valid_i] / q_i2t_sum[valid_i]
        
        # 计算KL损失
        logQ_i2t = torch.log(q_i2t.clamp_min(self.eps))
        KL_i2t = (P_i2t * (logP_i2t - logQ_i2t)).sum(dim=1)
        L_i2t = KL_i2t[valid_i].mean() if valid_i.any() else S.new_tensor(0.)
        
        # 列方向 (t2i) - 图库到查询
        logP_t2i = F.log_softmax(S.t(), dim=1)  # [M, N]
        P_t2i = logP_t2i.exp()
        
        y_t = y.t()  # [M, N]
        q_t2i_sum = y_t.sum(dim=1, keepdim=True)  # [M, 1]
        valid_j = (q_t2i_sum.squeeze(-1) > 0)  # [M]
        
        q_t2i = torch.zeros_like(y_t, dtype=S.dtype)
        q_t2i[valid_j] = y_t[valid_j] / q_t2i_sum[valid_j]
        
        logQ_t2i = torch.log(q_t2i.clamp_min(self.eps))
        KL_t2i = (P_t2i * (logP_t2i - logQ_t2i)).sum(dim=1)
        L_t2i = KL_t2i[valid_j].mean() if valid_j.any() else S.new_tensor(0.)
        
        # 总损失
        total_loss = L_i2t + L_t2i
        
        if return_details:
            details = {
                'loss_i2t': L_i2t.item(),
                'loss_t2i': L_t2i.item(),
                'total_loss': total_loss.item(),
                'temperature': self.sdm_head.get_temperature(),
                'scale': self.sdm_head.get_scale(),
                'valid_queries': valid_i.sum().item(),
                'valid_gallery': valid_j.sum().item(),
                'similarity_mean': S.mean().item(),
                'similarity_std': S.std().item(),
                'p_pos_mean': P_i2t[y.bool()].mean().item() if y.bool().any() else 0.0
            }
            return total_loss, details
        
        return total_loss
