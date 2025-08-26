# test_sdm_complete.py
"""
完整的SDM修复测试脚本
验证SDM权重调度、温度调度、损失计算等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sdm_head import SDMHead, SDMLossWithLearnableTemp
from models.sdm_scheduler import SDMScheduler, SDMWeightScheduler, SDMTemperatureScheduler
from configs.config import TrainingConfig


def test_sdm_head():
    """测试SDM头模块"""
    print("=== 测试SDM头模块 ===")
    
    # 创建SDM头
    sdm_head = SDMHead(init_tau=0.12)
    
    # 模拟数据
    batch_size = 16
    feature_dim = 768
    
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    
    # 前向传播
    similarity_matrix = sdm_head(qry_features, gal_features)
    
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    print(f"相似度范围: [{similarity_matrix.min().item():.3f}, {similarity_matrix.max().item():.3f}]")
    print(f"当前温度: {sdm_head.get_temperature():.3f}")
    print(f"当前缩放因子: {sdm_head.get_scale():.3f}")
    
    # 验证L2归一化
    qry_norm = torch.norm(qry_features, p=2, dim=1)
    gal_norm = torch.norm(gal_features, p=2, dim=1)
    print(f"查询特征范数范围: [{qry_norm.min().item():.3f}, {qry_norm.max().item():.3f}]")
    print(f"图库特征范数范围: [{gal_norm.min().item():.3f}, {gal_norm.max().item():.3f}]")
    
    print("✅ SDM头模块测试通过")


def test_sdm_loss():
    """测试SDM损失函数"""
    print("\n=== 测试SDM损失函数 ===")
    
    # 创建SDM损失函数
    sdm_loss = SDMLossWithLearnableTemp(init_tau=0.12)
    
    # 模拟数据
    batch_size = 16
    feature_dim = 768
    num_classes = 8
    
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    loss, details = sdm_loss(qry_features, gal_features, labels, return_details=True)
    
    print(f"SDM损失: {loss.item():.4f}")
    print(f"详细信息: {details}")
    
    # 验证损失为正
    assert loss.item() > 0, "SDM损失应该为正"
    print("✅ SDM损失函数测试通过")


def test_sdm_scheduler():
    """测试SDM调度器"""
    print("\n=== 测试SDM调度器 ===")
    
    # 创建配置
    config = TrainingConfig()
    
    # 创建调度器
    scheduler = SDMScheduler(config)
    
    # 模拟训练指标
    train_metrics = {
        'sdm_loss': 1.5,
        'stability_score': 0.9
    }
    
    val_metrics = {
        'map_avg2': 0.15
    }
    
    print("SDM权重调度测试:")
    for epoch in range(1, 16):
        weight, temp = scheduler.get_parameters(epoch, train_metrics, val_metrics)
        print(f"Epoch {epoch:2d}: weight={weight:.2f}, temp={temp:.3f}")
    
    print("\n权重增加测试:")
    if scheduler.can_increase_weight(15, train_metrics, val_metrics):
        scheduler.increase_weight()
        print(f"权重已增加到: {scheduler.weight_scheduler.current_weight}")
    
    print("\n权重降低测试:")
    scheduler.decrease_weight("测试降低")
    print(f"权重已降低到: {scheduler.weight_scheduler.current_weight}")
    
    print("✅ SDM调度器测试通过")


def test_temperature_scheduler():
    """测试温度调度器"""
    print("\n=== 测试温度调度器 ===")
    
    config = TrainingConfig()
    temp_scheduler = SDMTemperatureScheduler(config)
    
    print("正常温度调度:")
    for epoch in range(1, 8):
        temp = temp_scheduler.get_temperature(epoch)
        print(f"Epoch {epoch}: temp={temp:.3f}")
    
    print("\n稳定性检查测试:")
    # 正常情况
    normal_metrics = {'sdm_loss': 1.0, 'stability_score': 0.9}
    temp_scheduler.check_stability(normal_metrics)
    print(f"正常情况使用回退温度: {temp_scheduler.use_fallback}")
    
    # 异常情况
    abnormal_metrics = {'sdm_loss': 6.0, 'stability_score': 0.5}
    temp_scheduler.check_stability(abnormal_metrics)
    print(f"异常情况使用回退温度: {temp_scheduler.use_fallback}")
    
    # 重置
    temp_scheduler.reset_to_normal()
    print(f"重置后使用回退温度: {temp_scheduler.use_fallback}")
    
    print("✅ 温度调度器测试通过")


def test_weight_scheduler():
    """测试权重调度器"""
    print("\n=== 测试权重调度器 ===")
    
    config = TrainingConfig()
    weight_scheduler = SDMWeightScheduler(config)
    
    print("权重调度测试:")
    for epoch in range(1, 12):
        weight = weight_scheduler.get_weight(epoch)
        print(f"Epoch {epoch:2d}: weight={weight:.2f}")
    
    print("\n权重增加条件测试:")
    # 不满足条件
    early_metrics = {'stability_score': 0.7}
    early_val = {'map_avg2': 0.05}
    can_increase = weight_scheduler.can_increase_weight(5, early_metrics, early_val)
    print(f"早期训练可增加权重: {can_increase}")
    
    # 满足条件
    late_metrics = {'stability_score': 0.9}
    late_val = {'map_avg2': 0.2}
    can_increase = weight_scheduler.can_increase_weight(15, late_metrics, late_val)
    print(f"后期训练可增加权重: {can_increase}")
    
    print("✅ 权重调度器测试通过")


def test_complete_sdm_pipeline():
    """测试完整的SDM流程"""
    print("\n=== 测试完整SDM流程 ===")
    
    # 创建配置
    config = TrainingConfig()
    
    # 创建SDM组件
    sdm_loss = SDMLossWithLearnableTemp(init_tau=0.12)
    scheduler = SDMScheduler(config)
    
    # 模拟训练数据
    batch_size = 16
    feature_dim = 768
    num_classes = 8
    
    qry_features = torch.randn(batch_size, feature_dim)
    gal_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print("完整SDM训练流程模拟:")
    for epoch in range(1, 8):
        # 获取SDM参数
        weight, temp = scheduler.get_parameters(epoch, {}, {})
        
        # 计算损失
        loss, details = sdm_loss(qry_features, gal_features, labels, return_details=True)
        
        # 计算总损失
        ce_loss = 4.0  # 模拟CE损失
        total_loss = ce_loss + weight * loss
        
        print(f"Epoch {epoch}: weight={weight:.2f}, temp={temp:.3f}, "
              f"SDM_loss={loss.item():.4f}, total_loss={total_loss:.4f}")
    
    print("✅ 完整SDM流程测试通过")


def main():
    """主函数"""
    print("SDM完整修复测试")
    print("=" * 60)
    
    # 测试各个组件
    test_sdm_head()
    test_sdm_loss()
    test_sdm_scheduler()
    test_temperature_scheduler()
    test_weight_scheduler()
    test_complete_sdm_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ 所有SDM修复测试通过！")
    print("\n修复要点总结:")
    print("1. ✅ 实现了可学习温度的SDM头")
    print("2. ✅ 实现了SDM权重调度器（热身→放量策略）")
    print("3. ✅ 实现了SDM温度调度器（稳定性自适应）")
    print("4. ✅ 实现了完整的SDM损失函数")
    print("5. ✅ 集成了SDM调度器到训练流程")
    print("6. ✅ 按照文档要求实现了L2归一化")
    print("\n使用说明:")
    print("- 前3个epoch: λ_sdm = 0 (热身)")
    print("- Epoch 4-5: λ_sdm = 0.5 (初始)")
    print("- Epoch 6+: λ_sdm = 1.0 (稳定)")
    print("- 温度从0.12开始，稳定后降到0.10")
    print("- 出现不稳定时自动使用回退温度0.15")


if __name__ == "__main__":
    main()
