#!/usr/bin/env python3
# quickfix_test.py
# 快速测试修复效果的脚本

import torch
import logging
import numpy as np
from train import train_multimodal_reid

def test_quickfixes():
    """测试应用修复后的训练状态"""
    
    print("🔧 测试快速修复效果...")
    print("=" * 50)
    
    try:
        # 设置更详细的日志级别
        logging.basicConfig(level=logging.INFO, force=True)
        
        # 运行1个epoch的训练进行验证
        print("开始测试训练（只运行1个epoch）...")
        
        # 你可能需要临时修改config中的num_epochs为1来快速测试
        # 或者在这里中断训练循环
        
        train_multimodal_reid()
        
    except KeyboardInterrupt:
        print("\n测试中断 - 这是正常的")
        analyze_logs()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return True

def analyze_logs():
    """分析训练日志以验证修复效果"""
    
    print("\n📊 修复效果分析:")
    print("=" * 30)
    
    # 这里可以分析最新的log文件
    try:
        import os
        log_file = "./logs/training.log"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 检查最新的几行日志
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
            # 检查修复指标
            ce_values = []
            pairable_ratios = []
            bn_norms = []
            
            for line in recent_lines:
                # CE损失趋势
                if 'CE=' in line:
                    try:
                        ce_part = line.split('CE=')[1].split(',')[0].split()[0]
                        ce_val = float(ce_part)
                        ce_values.append(ce_val)
                    except:
                        pass
                
                # 可配对率
                if '可配对率:' in line:
                    try:
                        ratio_part = line.split('可配对率: ')[1].split('%')[0]
                        ratio = float(ratio_part.replace('%', ''))
                        pairable_ratios.append(ratio)
                    except:
                        pass
                
                # BN特征范数
                if 'Feat(BN)' in line:
                    try:
                        bn_part = line.split('Feat(BN)=')[1].split(',')[0]
                        bn_norm = float(bn_part)
                        bn_norms.append(bn_norm)
                    except:
                        pass
            
            # 分析结果
            print(f"✅ 修复状态检查:")
            
            if ce_values:
                latest_ce = ce_values[-1]
                print(f"  CE损失: {latest_ce:.3f} (目标: <5.5表示开始学习)")
                if latest_ce < 5.5:
                    print("    ✅ CE损失正在下降，分类头已开始学习!")
                elif latest_ce > 5.9:
                    print("    ❌ CE损失仍然卡在随机水平，需要进一步诊断")
            
            if pairable_ratios:
                latest_ratio = pairable_ratios[-1]
                print(f"  可配对率: {latest_ratio:.1f}% (目标: >80%)")
                if latest_ratio > 80:
                    print("    ✅ 采样器修复成功，SDM正样本充足!")
                else:
                    print("    ❌ 可配对率仍然过低，需要检查数据集")
            
            if bn_norms:
                avg_bn_norm = np.mean(bn_norms[-10:])  # 最近10个值的平均
                print(f"  BN特征范数: {avg_bn_norm:.2f}")
                if avg_bn_norm < 2.0:
                    print("    ✅ 特征范数正常，可能使用了L2归一化")
                elif 2.0 <= avg_bn_norm <= 15.0:
                    print("    ✅ 特征范数在合理范围内")
                else:
                    print("    ⚠️ 特征范数偏大，正则化可能需要调整")
            
        else:
            print("❌ 未找到训练日志文件")
            
    except Exception as e:
        print(f"❌ 日志分析失败: {e}")

if __name__ == "__main__":
    print("🚀 开始快速修复验证...")
    print("提示: 按Ctrl+C在看到几个epoch的输出后停止测试")
    print()
    
    test_quickfixes()