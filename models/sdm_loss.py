# sdm_loss.py
"""
SDM (Similarity Distribution Matching) 损失函数实现
基于 PRCV2025 全模态行人重识别任务的稳定实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


def sdm_loss_stable(qry, gal, y, tau=0.2, eps=1e-8):
    """
    稳健的SDM损失函数实现 - 按照优化清单修复数值稳定性
    
    Args:
        qry (torch.Tensor): [N, D] 查询特征  
        gal (torch.Tensor): [M, D] 图库特征
        y (torch.Tensor): [N, M] 同身份指示矩阵
        tau (float): 温度参数（推荐0.2避免数值溢出）
        eps (float): 数值稳定性参数
        
    Returns:
        torch.Tensor: SDM损失值（天然非负）
    """
    # 1) 强化数值稳定性：固定温度在安全范围，避免自适应引入的不稳定
    effective_tau = max(0.15, min(0.5, tau))  # 限制在[0.15, 0.5]安全区间
    
    # 2) L2归一化
    qry = F.normalize(qry, dim=1, eps=eps)
    gal = F.normalize(gal, dim=1, eps=eps)
    
    def _one_side_ce(S: torch.Tensor, y: torch.Tensor):
        """使用交叉熵形式，数值稳定版本，强化有效行检查"""
        # y: [N, M] ∈ {0,1}
        row_pos = y.sum(dim=1)  # [N] 每行正样本数
        valid = (row_pos > 0)   # 至少有一个正样本的行
        
        if not valid.any():
            # 没有任何有效行，直接返回0
            return torch.tensor(0.0, device=S.device, dtype=S.dtype)

        # 只处理有效行，提升数值稳定性
        S_valid = S[valid].clamp(-20.0, 20.0)  # [valid_N, M] 裁剪防溢出
        y_valid = y[valid]  # [valid_N, M]
        
        # 构造目标分布q：在正样本上均匀分布
        pos = (y_valid > 0).float()  # [valid_N, M]
        pos_sum = pos.sum(dim=1, keepdim=True).clamp_min(1.0)  # [valid_N, 1] 防零除
        q = pos / pos_sum  # [valid_N, M] 归一化到概率分布
        
        # 预测分布p的log值
        log_p = F.log_softmax(S_valid, dim=1)  # [valid_N, M]
        
        # 交叉熵：H(q,p) = -∑_j q_j * log p_j
        ce_per_row = -(q * log_p).sum(dim=1)  # [valid_N] 每行的交叉熵
        
        # 数值检查
        if not torch.isfinite(ce_per_row).all():
            nan_cnt = (~torch.isfinite(ce_per_row)).sum()
            print(f"⚠️ CE计算出现{nan_cnt}个NaN: S范围[{S_valid.min():.2f}, {S_valid.max():.2f}]")
            # 移除NaN值后计算均值
            ce_finite = ce_per_row[torch.isfinite(ce_per_row)]
            if ce_finite.numel() > 0:
                return ce_finite.mean()
            else:
                return torch.tensor(0.0, device=S.device, dtype=S.dtype)
        
        return ce_per_row.mean()
    
    # 3) 在fp32下计算，避免数值问题
    with torch.amp.autocast('cuda', enabled=False):
        # ✅ 特征和相似度检查
        qry_f, gal_f = qry.float(), gal.float()
        
        # 检查特征是否有异常
        for name, feat in [("qry", qry_f), ("gal", gal_f)]:
            if not torch.isfinite(feat).all():
                print(f"⚠️ {name}特征包含NaN/Inf")
                return torch.tensor(0.0, device=qry.device, dtype=qry.dtype)
            feat_max = feat.abs().max().item()
            if feat_max > 100.0:  # L2归一化后特征不应该过大
                print(f"⚠️ {name}特征幅值异常大: {feat_max:.3f}")
                
        S = qry_f @ gal_f.t() / effective_tau  # [N,M]
        
        # 检查相似度矩阵数值稳定性
        if not torch.isfinite(S).all():
            print(f"⚠️ 相似度矩阵包含NaN/Inf, tau={effective_tau:.3f}")
            return torch.tensor(0.0, device=qry.device, dtype=qry.dtype)
        
        # 相似度裁剪防止极值
        S = torch.clamp(S, min=-20.0, max=20.0)  # 强制裁剪到安全范围
            
        sim_min, sim_max = S.min().item(), S.max().item()
        if abs(sim_min) > 15 or abs(sim_max) > 15:
            print(f"⚠️ 相似度范围较大: [{sim_min:.2f}, {sim_max:.2f}], tau={effective_tau:.3f}")
            
        # ✅ 检查正样本计数
        pos_cnt = y.sum(dim=1)
        zero_pos_rows = (pos_cnt == 0).nonzero(as_tuple=True)[0]
        if len(zero_pos_rows) > 0:
            print(f"⚠️ 发现{len(zero_pos_rows)}行无正样本: {zero_pos_rows[:5].tolist()}")
        
        # 计算对称SDM损失，添加调试信息
        L_q2g = _one_side_ce(S, y)          # 查询->图库
        L_g2q = _one_side_ce(S.t(), y.t())  # 图库->查询（对称项）
        symmetric = 0.5 * (L_q2g + L_g2q)
        
        # 调试监控：检查有效行数和正样本分布
        pos_cnt_per_row = y.sum(dim=1)  # 每行的正样本数
        valid_rows = (pos_cnt_per_row > 0)
        
        # 每50次打印一次调试信息（减少日志噪音）
        import time
        if hasattr(sdm_loss_stable, '_last_debug_time'):
            current_time = time.time()
            if current_time - sdm_loss_stable._last_debug_time > 5.0:  # 每5秒打印一次
                print(f"SDM调试: valid_rows={int(valid_rows.sum())}/{len(valid_rows)}, "
                      f"pos_min={int(pos_cnt_per_row[valid_rows].min()) if valid_rows.any() else 0}, "
                      f"L_q2g={float(L_q2g):.3f}, L_g2q={float(L_g2q):.3f}")
                sdm_loss_stable._last_debug_time = current_time
        else:
            sdm_loss_stable._last_debug_time = time.time()

    # 4) 交叉熵天然非负，无需 clamp
    result = symmetric
    
    # 5) 最终数值检查与保护
    if torch.isnan(result) or torch.isinf(result) or result < 0:
        print(f"⚠️ SDM损失异常: {result:.6f}, tau={effective_tau:.3f}")
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
