# config.py
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import torchvision.transforms as T

@dataclass
class TrainingConfig:
    """修复后的训练配置"""
    # 数据相关
    data_root: str = "./data/train"
    json_file: str = "./data/train/text_annos.json"
    
    # 数据集划分
    val_ratio: float = 0.2
    seed: int = 42
    
    # 模型相关 - 使用ViT-Base预训练模型
    backbone: str = "vit_base_patch16_224"  # 使用ViT-Base预训练模型
    use_pretrained_vision: bool = True      # 启用ImageNet预训练权重

    # 统一的融合维度（视觉与文本都会被投到这个维度）
    fusion_dim: int = 768

    # 文本模型配置
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text: bool = True  # 保持向后兼容
    text_finetune_strategy: str = "top_layers"  # frozen, top_layers, lora, full
    
    # CLIP联合预训练模型配置（暂时禁用排查nan问题）
    use_clip: bool = False  # 暂时禁用CLIP排查nan问题
    clip_model_name: str = "ViT-B/32"  # 可选: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
    
    image_size: int = 224  # 匹配ViT模型要求的输入尺寸
    feature_dim: int = 2048  # ResNet50特征维度
    hidden_dim: int = 512  
    num_classes: int = 999
    dropout_rate: float = 0.1  # 极小dropout
    
    # 训练相关 - 极保守设置以防止NaN
    batch_size: int = 8   # 进一步降低批次大小
    num_epochs: int = 150
    learning_rate: float = 1e-6  # 极小学习率
    weight_decay: float = 1e-4   # 极小权重衰减
    warmup_epochs: int = 3       # 减少warmup
    scheduler: str = "constant"   # 暂时使用常数学习率
    
    # 分层学习率设置（基于base learning_rate的倍数）- 极保守设置
    backbone_lr_mult: float = 0.01     # 预训练骨干：1%基础学习率
    text_lr_mult: float = 0.01         # 文本编码器：1%基础学习率
    fusion_lr_mult: float = 0.1        # 融合模块：10%基础学习率
    heads_lr_mult: float = 0.5         # 分类/检索头：50%基础学习率
    adapters_lr_mult: float = 0.1      # 模态适配器：10%基础学习率
    
    # 损失权重 - 重新平衡（暂时禁用对比损失防止nan）
    ce_weight: float = 1.0
    contrastive_weight: float = 0.0  # 暂时完全禁用对比损失
    
    # 数据增强 - 暂时禁用所有增强
    random_flip: bool = False
    random_crop: bool = False
    color_jitter: bool = False
    random_erase: float = 0.0  # 禁用随机擦除
    
    # 模态dropout - 暂时禁用以简化
    modality_dropout: float = 0.0  # 禁用dropout
    min_modalities: int = 1  # 允许单模态训练
    
    # 模态感知批次采样配置（暂时禁用排查nan问题）
    use_modality_aware_sampling: bool = False  # 暂时禁用模态感知采样器
    min_modality_combinations: int = 3  # 每个批次最少包含的模态组合数
    
    # 设备和并行
    device: str = "cuda"
    num_workers: int = 0  # 临时设置为0避免多进程序列化问题
    pin_memory: bool = True
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 20
    eval_freq: int = 10  # 降低评估频率，从每10轮改为每20轮（训练时不再使用）
    eval_sample_ratio: float = 0.3  # 采样评估，只用30%数据进行快速mAP估算（训练时不再使用）
    
    # 早停和优化参数  
    patience: int = 15  # 早停耐心，组合分数15轮不改善就停止（防止过拟合）
    
    # 验证和推理相关配置
    inference_batch_size: int = 32
    best_model_path: str = "./checkpoints/best_model.pth"
    
    def __post_init__(self):
        """初始化配置后的处理"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置验证集图像预处理
        self.val_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])