# sdm_loss.py
"""
SDM (Similarity Distribution Matching) 损失函数实现
基于 PRCV2025 全模态行人重识别任务的稳定实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class SDMLoss(nn.Module):
    """
    SDM (Similarity Distribution Matching) 损失函数
    
    计算查询特征与图库特征之间的分布匹配损失
    支持多模态查询特征与RGB图库特征的匹配
    """
    
    def __init__(self, temperature=0.1, eps=1e-6):
        """
        Args:
            temperature (float): 温度参数，控制softmax的陡峭程度
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, qry_features, gal_features, labels, return_details=False):
        """
        计算SDM损失
        
        Args:
            qry_features (torch.Tensor): 查询特征 [N, D] - 任意模态或多模态融合
            gal_features (torch.Tensor): 图库特征 [M, D] - RGB特征
            labels (torch.Tensor): 身份标签 [N] - 查询样本的身份ID
            return_details (bool): 是否返回详细信息
            
        Returns:
            torch.Tensor: SDM损失值
            dict (optional): 详细信息
        """
        N, D = qry_features.shape
        M, _ = gal_features.shape
        
        # 1) L2归一化（避免全零向量除零）
        qry_norm = F.normalize(qry_features, dim=1, eps=self.eps)
        gal_norm = F.normalize(gal_features, dim=1, eps=self.eps)
        
        # 2) 计算相似度矩阵
        S = qry_norm @ gal_norm.t()  # [N, M]
        
        # 3) 构造同身份指示矩阵
        # 将labels扩展到[N, M]矩阵，同身份为1，不同身份为0
        labels_qry = labels.view(-1, 1)  # [N, 1]
        labels_gal = labels.view(1, -1)  # [1, M] - 这里需要根据实际情况调整
        
        # 注意：这里需要根据实际的图库标签来构造y矩阵
        # 暂时使用查询标签，实际使用时需要传入图库标签
        y = (labels_qry == labels_gal).float()  # [N, M]
        
        # 4) 行方向 (i2t) - 查询到图库
        logP_i2t = F.log_softmax(S / self.temperature, dim=1)  # [N, M]
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
        
        # 5) 列方向 (t2i) - 图库到查询
        logP_t2i = F.log_softmax(S.t() / self.temperature, dim=1)  # [M, N]
        P_t2i = logP_t2i.exp()
        
        y_t = y.t()  # [M, N]
        q_t2i_sum = y_t.sum(dim=1, keepdim=True)  # [M, 1]
        valid_j = (q_t2i_sum.squeeze(-1) > 0)  # [M]
        
        q_t2i = torch.zeros_like(y_t, dtype=S.dtype)
        q_t2i[valid_j] = y_t[valid_j] / q_t2i_sum[valid_j]
        
        logQ_t2i = torch.log(q_t2i.clamp_min(self.eps))
        KL_t2i = (P_t2i * (logP_t2i - logQ_t2i)).sum(dim=1)
        L_t2i = KL_t2i[valid_j].mean() if valid_j.any() else S.new_tensor(0.)
        
        # 6) 总损失
        total_loss = L_i2t + L_t2i
        
        if return_details:
            details = {
                'loss_i2t': L_i2t.item(),
                'loss_t2i': L_t2i.item(),
                'total_loss': total_loss.item(),
                'temperature': self.temperature,
                'valid_queries': valid_i.sum().item(),
                'valid_gallery': valid_j.sum().item(),
                'similarity_mean': S.mean().item(),
                'similarity_std': S.std().item(),
                'p_pos_mean': P_i2t[y.bool()].mean().item() if y.bool().any() else 0.0
            }
            return total_loss, details
        
        return total_loss


def sdm_loss_stable(qry, gal, y, tau=0.1, eps=1e-6):
    """
    稳定的SDM损失函数实现（函数版本）
    
    Args:
        qry (torch.Tensor): [N, D] 查询特征
        gal (torch.Tensor): [M, D] 图库特征  
        y (torch.Tensor): [N, M] 同身份指示矩阵
        tau (float): 温度参数
        eps (float): 数值稳定性参数
        
    Returns:
        torch.Tensor: SDM损失值
    """
    # 1) L2归一化
    qry = F.normalize(qry, dim=1, eps=eps)
    gal = F.normalize(gal, dim=1, eps=eps)
    
    # 2) 相似度矩阵
    S = qry @ gal.t()  # [N, M]
    
    # 3) 行方向 (i2t)
    logP_i2t = F.log_softmax(S / tau, dim=1)  # [N, M]
    P_i2t = logP_i2t.exp()
    
    q_i2t_sum = y.sum(dim=1, keepdim=True)
    valid_i = (q_i2t_sum.squeeze(-1) > 0)
    q_i2t = torch.zeros_like(y, dtype=S.dtype)
    q_i2t[valid_i] = y[valid_i] / q_i2t_sum[valid_i]
    
    logQ_i2t = torch.log(q_i2t.clamp_min(eps))
    KL_i2t = (P_i2t * (logP_i2t - logQ_i2t)).sum(dim=1)
    L_i2t = KL_i2t[valid_i].mean() if valid_i.any() else S.new_tensor(0.)
    
    # 4) 列方向 (t2i)
    logP_t2i = F.log_softmax(S.t() / tau, dim=1)  # [M, N]
    P_t2i = logP_t2i.exp()
    
    y_t = y.t()
    q_t2i_sum = y_t.sum(dim=1, keepdim=True)
    valid_j = (q_t2i_sum.squeeze(-1) > 0)
    q_t2i = torch.zeros_like(y_t, dtype=S.dtype)
    q_t2i[valid_j] = y_t[valid_j] / q_t2i_sum[valid_j]
    
    logQ_t2i = torch.log(q_t2i.clamp_min(eps))
    KL_t2i = (P_t2i * (logP_t2i - logQ_t2i)).sum(dim=1)
    L_t2i = KL_t2i[valid_j].mean() if valid_j.any() else S.new_tensor(0.)
    
    return L_i2t + L_t2i


# 测试函数
def test_sdm_loss():
    """测试SDM损失函数"""
    print("测试SDM损失函数...")
    
    # 模拟数据
    batch_size = 16
    feature_dim = 768
    num_classes = 8
    
    # 生成特征
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    
    # 生成标签（确保有正样本）
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 构造同身份指示矩阵
    labels_qry = labels.view(-1, 1)
    labels_gal = labels.view(1, -1)
    y = (labels_qry == labels_gal).float()
    
    # 测试SDM损失
    sdm_module = SDMLoss(temperature=0.1)
    loss, details = sdm_module(qry_features, gal_features, labels, return_details=True)
    
    print(f"SDM损失: {loss.item():.4f}")
    print(f"详细信息: {details}")
    
    # 测试函数版本
    loss_func = sdm_loss_stable(qry_features, gal_features, y, tau=0.1)
    print(f"函数版本SDM损失: {loss_func.item():.4f}")
    
    return loss.item() > 0  # 确保损失为正


if __name__ == "__main__":
    test_sdm_loss()
