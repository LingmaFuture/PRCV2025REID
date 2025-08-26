# debug_ce_convergence.py
"""
调试CE收敛异常的脚本
分析学习率、特征范数、损失权重等因素对CE收敛的影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim import AdamW
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_learning_rate_schedule():
    """分析学习率调度器的行为"""
    print("=== 学习率调度器分析 ===")
    
    # 模拟配置
    base_lr = 5e-4
    text_lr = base_lr * 0.1
    warmup_epochs = 15
    num_epochs = 150
    
    # 创建虚拟参数组
    param_groups = [
        {'params': [torch.randn(10, 10)], 'lr': text_lr},  # 文本编码器
        {'params': [torch.randn(10, 10)], 'lr': base_lr}   # 其他模块
    ]
    
    optimizer = AdamW(param_groups)
    
    # 创建调度器
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # 记录学习率变化
    lr_history = {'text': [], 'other': []}
    
    for epoch in range(1, num_epochs + 1):
        lr_history['text'].append(optimizer.param_groups[0]['lr'])
        lr_history['other'].append(optimizer.param_groups[1]['lr'])
        
        if epoch < warmup_epochs:
            scheduler.step()
    
    # 分析前20个epoch的学习率
    print(f"前20个epoch的学习率变化:")
    for epoch in range(1, 21):
        print(f"Epoch {epoch:2d}: Text LR = {lr_history['text'][epoch-1]:.2e}, "
              f"Other LR = {lr_history['other'][epoch-1]:.2e}")
    
    # 检查第10个epoch附近的学习率
    print(f"\n第10个epoch附近的学习率:")
    for epoch in range(8, 13):
        print(f"Epoch {epoch:2d}: Text LR = {lr_history['text'][epoch-1]:.2e}, "
              f"Other LR = {lr_history['other'][epoch-1]:.2e}")
    
    return lr_history

def analyze_feature_norms():
    """分析特征范数对CE损失的影响"""
    print("\n=== 特征范数分析 ===")
    
    # 模拟不同范数的特征
    batch_size = 32
    num_classes = 400
    feature_dim = 768
    
    # 测试不同范数的特征
    norms_to_test = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    
    for norm in norms_to_test:
        # 生成指定范数的特征
        features = torch.randn(batch_size, feature_dim)
        features = F.normalize(features, p=2, dim=1) * norm
        
        # 生成logits
        classifier = nn.Linear(feature_dim, num_classes)
        logits = classifier(features)
        
        # 生成随机标签
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # 计算CE损失
        ce_loss = F.cross_entropy(logits, labels)
        
        print(f"特征范数: {norm:5.1f}, CE损失: {ce_loss.item():.4f}, "
              f"Logits范数: {torch.norm(logits, p=2, dim=1).mean().item():.2f}")

def analyze_contrastive_loss_impact():
    """分析对比损失对总损失的影响"""
    print("\n=== 对比损失影响分析 ===")
    
    # 模拟不同的CE损失和对比损失组合
    ce_losses = [5.0, 5.5, 6.0, 4.5, 4.0]
    contrastive_losses = [0.5, 1.0, 1.5, 2.0, 2.5]
    contrastive_weights = [0.05, 0.1, 0.15, 0.2]
    
    print("不同对比损失权重下的总损失:")
    for weight in contrastive_weights:
        print(f"\n对比损失权重: {weight}")
        for ce, cont in zip(ce_losses, contrastive_losses):
            total_loss = ce + weight * cont
            print(f"  CE: {ce:.1f}, Cont: {cont:.1f}, Total: {total_loss:.3f}")

def create_monitoring_script():
    """创建训练监控脚本"""
    print("\n=== 训练监控脚本 ===")
    
    monitoring_code = '''
# 在train_epoch函数中添加以下监控代码

def monitor_training_stability(model, batch, labels, outputs, loss_dict, epoch, batch_idx):
    """监控训练稳定性"""
    # 1. 监控特征范数
    if isinstance(outputs, dict) and 'reid_features_raw' in outputs:
        feat_norms = torch.norm(outputs['reid_features_raw'], p=2, dim=1)
        avg_norm = feat_norms.mean().item()
        max_norm = feat_norms.max().item()
        
        if avg_norm > 20.0 or max_norm > 50.0:
            logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 特征范数异常 - 平均: {avg_norm:.2f}, 最大: {max_norm:.2f}")
    
    # 2. 监控logits范数
    if isinstance(outputs, dict) and 'logits' in outputs:
        logit_norms = torch.norm(outputs['logits'], p=2, dim=1)
        avg_logit_norm = logit_norms.mean().item()
        
        if avg_logit_norm > 100.0:
            logging.warning(f"Epoch {epoch}, Batch {batch_idx}: Logits范数异常 - 平均: {avg_logit_norm:.2f}")
    
    # 3. 监控损失值
    ce_loss = loss_dict.get('ce_loss', torch.tensor(0.0))
    cont_loss = loss_dict.get('contrastive_loss', torch.tensor(0.0))
    
    if ce_loss.item() > 10.0:
        logging.warning(f"Epoch {epoch}, Batch {batch_idx}: CE损失异常 - {ce_loss.item():.4f}")
    
    if cont_loss.item() > 5.0:
        logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 对比损失异常 - {cont_loss.item():.4f}")
    
    # 4. 监控梯度范数
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > 10.0:
        logging.warning(f"Epoch {epoch}, Batch {batch_idx}: 梯度范数异常 - {total_norm:.4f}")

# 在训练循环中调用
monitor_training_stability(model, batch, labels, outputs, loss_dict, epoch, batch_idx)
'''
    
    print(monitoring_code)

def suggest_fixes():
    """建议修复方案"""
    print("\n=== 修复建议 ===")
    
    fixes = [
        "1. 降低学习率: 将base_lr从5e-4降低到3e-4或2e-4",
        "2. 缩短warmup: 将warmup_epochs从15降低到5-10",
        "3. 特征归一化: 在特征投影后添加L2归一化",
        "4. 梯度裁剪: 降低梯度裁剪阈值从1.0到0.5",
        "5. 对比损失权重: 暂时降低contrastive_weight到0.05",
        "6. 添加监控: 在训练循环中添加特征范数和损失值监控",
        "7. 学习率调度: 考虑使用更温和的调度器如StepLR"
    ]
    
    for fix in fixes:
        print(fix)

def main():
    """主函数"""
    print("CE收敛异常分析工具")
    print("=" * 50)
    
    # 分析学习率调度器
    lr_history = analyze_learning_rate_schedule()
    
    # 分析特征范数影响
    analyze_feature_norms()
    
    # 分析对比损失影响
    analyze_contrastive_loss_impact()
    
    # 创建监控脚本
    create_monitoring_script()
    
    # 建议修复方案
    suggest_fixes()
    
    print("\n" + "=" * 50)
    print("分析完成！建议根据上述建议调整训练配置。")

if __name__ == "__main__":
    main()
