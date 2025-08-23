#!/usr/bin/env python3
"""
测试修复后的SDM损失实现
验证KL散度计算是否正确，损失是否为正值
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.sdm_loss import sdm_loss_stable, SDMLoss

def test_sdm_fixes():
    """测试SDM修复效果"""
    print("[TEST] 测试修复后的SDM损失函数...")
    
    # 设置随机种子确保可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 模拟数据
    batch_size = 16
    feature_dim = 768
    num_classes = 8
    
    # 生成特征（模拟不同范数的情况）
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    
    # 测试不同特征范数情况
    print("\n[NORM] 测试不同特征范数情况:")
    for scale in [0.5, 1.0, 5.0, 10.0, 20.0]:
        qry_scaled = qry_features * scale
        gal_scaled = gal_features * scale
        
        # 生成标签（确保有正样本）
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # 构造同身份指示矩阵
        labels_qry = labels.view(-1, 1)
        labels_gal = labels.view(1, -1)
        y = (labels_qry == labels_gal).float()
        
        # 测试修复后的SDM损失
        try:
            loss = sdm_loss_stable(qry_scaled, gal_scaled, y, tau=0.2)
            feat_norm = qry_scaled.norm(dim=1).mean().item()
            
            print(f"  范数 {feat_norm:6.2f}: SDM损失 = {loss.item():8.4f} ({'✅ 非负' if loss.item() >= 0 else '❌ 负值'})")
            
            # 检查是否有NaN
            if torch.isnan(loss):
                print(f"    ❌ 检测到NaN!")
            
        except Exception as e:
            print(f"  范数 {scale:6.2f}: ❌ 错误: {e}")
    
    print("\n🧩 测试SDMLoss模块:")
    sdm_module = SDMLoss(temperature=0.2)
    
    # 测试正常情况
    labels = torch.randint(0, num_classes, (batch_size,))
    try:
        loss, details = sdm_module(qry_features, gal_features, labels, return_details=True)
        print(f"  SDM模块损失: {loss.item():.4f}")
        print(f"  详细信息: {details}")
        print(f"  ✅ SDM模块测试通过" if loss.item() >= 0 else f"  ❌ SDM模块返回负值: {loss.item()}")
    except Exception as e:
        print(f"  ❌ SDM模块错误: {e}")
    
    print("\n🔍 数值稳定性测试:")
    # 测试极端情况
    test_cases = [
        ("零特征", torch.zeros(batch_size, feature_dim)),
        ("极小特征", torch.randn(batch_size, feature_dim) * 1e-8),
        ("极大特征", torch.randn(batch_size, feature_dim) * 1e3),
        ("单位特征", F.normalize(torch.randn(batch_size, feature_dim), dim=1)),
    ]
    
    for name, features in test_cases:
        labels = torch.randint(0, num_classes, (batch_size,))
        try:
            loss, _ = sdm_module(features, features, labels, return_details=True)
            status = "✅ 通过" if torch.isfinite(loss) and loss.item() >= 0 else f"❌ 异常: {loss.item()}"
            print(f"  {name:8s}: {status}")
        except Exception as e:
            print(f"  {name:8s}: ❌ 错误: {str(e)[:50]}...")
    
    print("\n🎯 温度自适应测试:")
    # 测试不同温度参数
    for tau in [0.1, 0.15, 0.2, 0.25, 0.3]:
        labels = torch.randint(0, num_classes, (batch_size,))
        labels_qry = labels.view(-1, 1)
        labels_gal = labels.view(1, -1)
        y = (labels_qry == labels_gal).float()
        
        loss = sdm_loss_stable(qry_features, gal_features, y, tau=tau)
        print(f"  τ={tau:.2f}: SDM损失 = {loss.item():6.4f}")
    
    print("\n✅ SDM修复测试完成!")
    return True

if __name__ == "__main__":
    test_sdm_fixes()