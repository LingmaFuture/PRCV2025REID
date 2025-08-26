# test_conservative_scheduler.py
"""
测试保守学习率调度器配置
验证不同调度器类型的学习率变化曲线
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import LinearLR, SequentialLR

def create_dummy_model():
    """创建一个简单的模型用于测试"""
    return torch.nn.Linear(10, 1)

def test_scheduler_curves():
    """测试不同调度器的学习率变化曲线"""
    
    # 基础参数
    base_lr = 5e-4
    num_epochs = 100
    warmup_epochs = 15
    conservative_factor = 0.7
    
    # 创建测试模型和优化器
    model = create_dummy_model()
    
    # 测试不同调度器
    schedulers_config = {
        'cosine_conservative': {
            'type': 'cosine',
            'warmup': True,
            'eta_min': base_lr * 0.01
        },
        'step_conservative': {
            'type': 'step',
            'step_size': int(50 * conservative_factor),
            'gamma': 0.3 + 0.4 * conservative_factor
        },
        'multistep_conservative': {
            'type': 'multistep',
            'milestones': [int(60 * conservative_factor), int(80 * conservative_factor), int(95 * conservative_factor)],
            'gamma': 0.2 + 0.5 * conservative_factor
        },
        'plateau_conservative': {
            'type': 'plateau',
            'factor': 0.5,
            'patience': 8
        }
    }
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, config) in enumerate(schedulers_config.items(), 1):
        plt.subplot(2, 2, i)
        
        # 重新创建优化器
        optimizer = AdamW(model.parameters(), lr=base_lr)
        
        # 创建调度器
        if config['type'] == 'cosine':
            if config.get('warmup', False):
                warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
                cosine_scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=num_epochs - warmup_epochs, 
                    eta_min=config['eta_min']
                )
                scheduler = SequentialLR(
                    optimizer, 
                    [warmup_scheduler, cosine_scheduler], 
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=config['eta_min'])
                
        elif config['type'] == 'step':
            scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
            
        elif config['type'] == 'multistep':
            scheduler = MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
            
        elif config['type'] == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                factor=config['factor'], 
                patience=config['patience'],
                min_lr=base_lr * 0.001
            )
        
        # 记录学习率变化
        learning_rates = []
        epochs = list(range(1, num_epochs + 1))
        
        for epoch in epochs:
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 步进调度器
            if config['type'] == 'plateau':
                # 模拟mAP变化（先上升后平稳）
                if epoch < 30:
                    fake_map = 0.1 + epoch * 0.01
                elif epoch < 60:
                    fake_map = 0.4 + (epoch - 30) * 0.005
                else:
                    fake_map = 0.55 + np.random.normal(0, 0.01)  # 添加噪声模拟波动
                scheduler.step(fake_map)
            else:
                scheduler.step()
        
        # 绘制曲线
        plt.plot(epochs, learning_rates, linewidth=2, label=f'{name}')
        plt.title(f'{name.upper()} 学习率调度器\n'
                 f'配置: {config}', fontsize=10)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 添加关键信息
        min_lr = min(learning_rates)
        max_lr = max(learning_rates)
        plt.text(0.02, 0.98, f'Max LR: {max_lr:.2e}\nMin LR: {min_lr:.2e}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('tools/conservative_scheduler_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 保守调度器测试完成！")
    print(f"📊 学习率变化曲线已保存到: tools/conservative_scheduler_comparison.png")
    print(f"📋 调度器配置对比:")
    for name, config in schedulers_config.items():
        print(f"  - {name}: {config}")

def validate_scheduler_stability():
    """验证调度器稳定性参数"""
    
    base_lr = 5e-4
    conservative_factor = 0.7
    
    print("\n🔍 保守调度器稳定性参数验证:")
    print(f"基础学习率: {base_lr:.2e}")
    print(f"保守系数: {conservative_factor}")
    
    # Cosine调度器
    eta_min = base_lr * 0.01
    print(f"\n📈 Cosine调度器:")
    print(f"  - 最小学习率: {eta_min:.2e} (基础LR的1%)")
    print(f"  - Warmup启动因子: 0.01 (更温和的启动)")
    
    # Step调度器
    step_size = int(50 * conservative_factor)
    gamma = 0.3 + 0.4 * conservative_factor
    print(f"\n📈 Step调度器:")
    print(f"  - 步长: {step_size} epochs")
    print(f"  - 衰减因子: {gamma:.2f}")
    
    # MultiStep调度器
    milestones = [int(60 * conservative_factor), int(80 * conservative_factor), int(95 * conservative_factor)]
    gamma_multi = 0.2 + 0.5 * conservative_factor
    print(f"\n📈 MultiStep调度器:")
    print(f"  - 里程碑: {milestones}")
    print(f"  - 衰减因子: {gamma_multi:.2f}")
    
    # Plateau调度器
    print(f"\n📈 Plateau调度器:")
    print(f"  - 衰减因子: 0.5")
    print(f"  - 耐心值: 8 epochs")
    print(f"  - 最小学习率: {base_lr * 0.001:.2e}")
    
    print(f"\n✅ 所有参数均已设置为更保守的值，有助于提升训练稳定性！")

if __name__ == "__main__":
    print("🧪 开始测试保守学习率调度器...")
    
    try:
        test_scheduler_curves()
        validate_scheduler_stability()
        
        print(f"\n🎉 保守调度器测试成功完成！")
        print(f"📝 建议使用的调度器类型:")
        print(f"  1. 一般情况: cosine (带warmup)")
        print(f"  2. 训练不稳定: plateau (自适应)")
        print(f"  3. 需要阶段性衰减: multistep")
        
    except ImportError as e:
        print(f"⚠️  缺少依赖库: {e}")
        print(f"💡 请安装: pip install matplotlib")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
