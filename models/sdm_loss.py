# sdm_loss.py
"""
SDM (Similarity Distribution Matching) 损失函数实现
基于 PRCV2025 全模态行人重识别任务的稳定实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def sdm_loss_stable(qry, gal, y, tau=0.2, eps=1e-6):
    """
    稳健的SDM损失函数实现 - 修复KL散度计算错误
    
    Args:
        qry (torch.Tensor): [N, D] 查询特征
        gal (torch.Tensor): [M, D] 图库特征  
        y (torch.Tensor): [N, M] 同身份指示矩阵
        tau (float): 温度参数（提高默认值到0.2以增强数值稳定性）
        eps (float): 数值稳定性参数
        
    Returns:
        torch.Tensor: SDM损失值
    """
    # 1) 修复的自适应温度：在归一化前统计范数
    qry_norms = qry.norm(dim=1).mean().detach()
    gal_norms = gal.norm(dim=1).mean().detach()
    scale = 0.5 * (qry_norms + gal_norms)
    adaptive_tau = float(torch.clamp(tau * (scale / 8.0), 0.15, 0.5))
    
    # 2) L2归一化
    qry = F.normalize(qry, dim=1, eps=eps)
    gal = F.normalize(gal, dim=1, eps=eps)
    
    def _one_side_ce(S: torch.Tensor, y: torch.Tensor):
        """使用交叉熵形式，杜绝负值"""
        # y: [N, M] ∈ {0,1}
        row_pos = y.sum(dim=1, keepdim=True)         # [N,1]
        valid = (row_pos.squeeze(-1) > 0)            # 只保留至少有一个正样本的行
        if not valid.any():
            return S.new_tensor(0.)

        # 目标分布 q：正样本上均匀分布，负样本严格为 0；不做 eps 抬升
        q = torch.zeros_like(S)
        q[valid] = y[valid] / row_pos[valid]

        # 预测分布 p
        log_p = F.log_softmax(S, dim=1)

        # 交叉熵 H(q,p) = -∑ q * log p   （保证非负）
        ce = -(q * log_p).sum(dim=1)                 # [N]
        return ce[valid].mean()
    
    # 3) 在fp32下计算，避免数值问题
    with torch.cuda.amp.autocast(enabled=False):
        S = (qry.float() @ gal.float().t()) / adaptive_tau  # [N,M]
        L_q2g = _one_side_ce(S, y)          # KL(q||p) 的等价优化目标
        L_g2q = _one_side_ce(S.t(), y.t())  # 对称项
        symmetric = 0.5 * (L_q2g + L_g2q)

    # 4) 交叉熵天然非负，无需 clamp
    result = symmetric
    
    # 5) 异常监控（应该不再需要）
    if torch.isnan(result):
        print(f"⚠️ SDM出现NaN: tau={adaptive_tau:.3f}, scale={scale:.2f}")
        return torch.tensor(0.0, device=qry.device, dtype=qry.dtype)
    
    return result


class SDMLoss(nn.Module):
    """
    SDM (Similarity Distribution Matching) 损失函数
    
    计算查询特征与图库特征之间的分布匹配损失
    支持多模态查询特征与RGB图库特征的匹配
    """
    
    def __init__(self, temperature=0.2, eps=1e-6):
        """
        Args:
            temperature (float): 温度参数，控制softmax的陡峭程度
            eps (float): 数值稳定性参数
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, qry_features, gal_features, qry_labels, gal_labels=None, return_details=False):
        """
        计算SDM损失 - 修复版本，支持两套标签
        
        Args:
            qry_features (torch.Tensor): 查询特征 [N, D] - 任意模态或多模态融合
            gal_features (torch.Tensor): 图库特征 [M, D] - RGB特征
            qry_labels (torch.Tensor): 查询样本的身份ID [N]
            gal_labels (torch.Tensor, optional): 图库样本的身份ID [M]，None时使用qry_labels
            return_details (bool): 是否返回详细信息
            
        Returns:
            torch.Tensor: SDM损失值
            dict (optional): 详细信息
        """        
        if gal_labels is None:
            gal_labels = qry_labels  # 仅在完全对齐的 toy case 使用
        
        # 构造同身份指示矩阵
        y = (qry_labels.view(-1,1) == gal_labels.view(1,-1)).float()  # [N,M]

        # 直接把"未归一化的原始特征"交给 sdm_loss_stable，内部会做 L2 normalize
        total_loss = sdm_loss_stable(qry_features, gal_features, y,
                                     tau=self.temperature, eps=self.eps)

        if return_details:
            with torch.no_grad():
                qn = F.normalize(qry_features, dim=1, eps=self.eps)
                gn = F.normalize(gal_features, dim=1, eps=self.eps)
                S = qn @ gn.t()
            details = {
                'total_loss': float(total_loss),
                'temperature': self.temperature,
                'similarity_mean': float(S.mean()),
                'similarity_std': float(S.std()),
                'valid_queries': int((y.sum(dim=1)>0).sum()),
                'valid_gallery': int((y.sum(dim=0)>0).sum()),
            }
            return total_loss, details
            
        return total_loss


def _quick_check():
    """快速自测函数"""
    torch.manual_seed(0)
    N, M, D = 16, 48, 768
    qry = torch.randn(N, D)
    gal = torch.randn(M, D)
    # 构造部分重叠标签
    qry_labels = torch.randint(0, 10, (N,))
    gal_labels = torch.randint(0, 10, (M,))

    y = (qry_labels.view(-1,1) == gal_labels.view(1,-1)).float()
    loss = sdm_loss_stable(qry, gal, y, tau=0.2)
    assert torch.isfinite(loss), "loss 出现 NaN/Inf"
    assert float(loss) >= 0.0, f"loss 为负: {float(loss)}"
    print("✅ SDM quick check OK:", float(loss))
    
    # 测试SDMLoss模块
    sdm_module = SDMLoss(temperature=0.2)
    module_loss, details = sdm_module(qry, gal, qry_labels, gal_labels, return_details=True)
    assert torch.isfinite(module_loss), "module loss 出现 NaN/Inf"
    assert float(module_loss) >= 0.0, f"module loss 为负: {float(module_loss)}"
    print("✅ SDMLoss module check OK:", float(module_loss))
    print("详细信息:", details)


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
    sdm_module = SDMLoss(temperature=0.2)
    loss, details = sdm_module(qry_features, gal_features, labels, labels, return_details=True)
    
    print(f"SDM损失: {loss.item():.4f}")
    print(f"详细信息: {details}")
    
    # 测试函数版本
    loss_func = sdm_loss_stable(qry_features, gal_features, y, tau=0.1)
    print(f"函数版本SDM损失: {loss_func.item():.4f}")
    
    return loss.item() > 0  # 确保损失为正


if __name__ == "__main__":
    _quick_check()
    test_sdm_loss()
