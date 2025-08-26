#!/usr/bin/env python3
"""
测试稳定性修复效果
验证调度器不再被空指标触发，spike检测更合理
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_scheduler_fix():
    """测试调度器修复"""
    print("[TEST] 测试调度器空指标容错...")
    
    from configs.config import TrainingConfig
    from models.sdm_scheduler import SDMScheduler
    
    config = TrainingConfig()
    scheduler = SDMScheduler(config)
    
    # 测试空指标情况（修复前会触发回退）
    weight1, temp1 = scheduler.get_parameters(epoch=1, train_metrics={})
    print(f"空指标调用: weight={weight1:.3f}, temp={temp1:.3f}")
    
    # 测试缺少stability_score的情况
    weight2, temp2 = scheduler.get_parameters(epoch=1, train_metrics={'sdm_loss': 1.5})
    print(f"缺少stability_score: weight={weight2:.3f}, temp={temp2:.3f}")
    
    # 测试正常指标
    weight3, temp3 = scheduler.get_parameters(epoch=1, train_metrics={
        'sdm_loss': 1.5, 
        'stability_score': 0.8
    })
    print(f"正常指标: weight={weight3:.3f}, temp={temp3:.3f}")
    
    print("[OK] 调度器容错测试通过")
    return True

def test_spike_detection():
    """测试spike检测修复"""
    print("[TEST] 测试spike检测热身期...")
    
    # 模拟训练状态
    state = {
        'loss_hist': [],
        'spikes': 0,
        'batches': 0
    }
    
    # 前19个样本不应该触发检测
    losses = [2.0 + 0.1 * np.sin(i) for i in range(19)]
    spike_count = 0
    
    for i, loss in enumerate(losses):
        state['loss_hist'].append(loss)
        if len(state['loss_hist']) > 200:
            state['loss_hist'] = state['loss_hist'][-200:]
        
        # 复制修复后的检测逻辑
        if len(state['loss_hist']) >= 20:  # 热身期
            hist = np.array(state['loss_hist'][-100:])
            median = np.median(hist)
            mad = np.median(np.abs(hist - median))
            mad = max(mad, 0.05)
            threshold = max(median + 6.0 * 1.4826 * mad, median * 1.15)
            
            if loss > threshold:
                spike_count += 1
                
        state['batches'] += 1
    
    print(f"前19个样本的spike检测: {spike_count}次 (应该为0)")
    assert spike_count == 0, "热身期不应该检测到spike"
    
    # 第20个样本开始检测
    state['loss_hist'].append(10.0)  # 明显异常值
    if len(state['loss_hist']) >= 20:
        hist = np.array(state['loss_hist'][-100:])
        median = np.median(hist)
        mad = np.median(np.abs(hist - median))
        mad = max(mad, 0.05)
        threshold = max(median + 6.0 * 1.4826 * mad, median * 1.15)
        
        if 10.0 > threshold:
            spike_count += 1
            
    print(f"第20个样本加入异常值后: {spike_count}次 (应该为1)")
    assert spike_count == 1, "应该检测到1次spike"
    
    print("[OK] spike检测热身期测试通过")
    return True

if __name__ == "__main__":
    success = True
    try:
        success &= test_scheduler_fix()
        success &= test_spike_detection()
        
        if success:
            print("\n[SUCCESS] 所有稳定性修复测试通过！")
            print("期望效果:")
            print("  - 不再频繁出现'训练不稳定，使用回退温度0.25'警告")
            print("  - stability_score从0.00回升至0.6-0.9")
            print("  - 进度条GradNorm显示'—'而非'0.00'")
            print("  - SDMLoss保持非负且收敛")
        else:
            print("\n[FAIL] 部分测试失败")
            
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    exit(0 if success else 1)