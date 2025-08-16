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
    
    # 模型相关 - 基于预训练ViT微调
    backbone: str = "vit_base_patch16_224"  # 使用ViT-Base预训练模型
    use_pretrained_vision: bool = True      # 启用ImageNet预训练权重

    # 统一的融合维度（视觉与文本都会被投到这个维度）
    fusion_dim: int = 768

    # 文本模型配置
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text: bool = True  # 保持向后兼容
    text_finetune_strategy: str = "top_layers"  # frozen, top_layers, lora, full
    
    # CLIP联合预训练模型配置
    use_clip: bool = True  # 安装CLIP后可设为True以启用联合预训练
    clip_model_name: str = "ViT-B/32"  # 可选: ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101
    
    image_size: int = 224  # 匹配ViT模型要求的输入尺寸
    feature_dim: int = 2048  # ResNet50特征维度
    hidden_dim: int = 512  
    num_classes: int = 999
    dropout_rate: float = 0.5  # 增强dropout
    
    # 训练相关 - 预训练模型微调超参数
    batch_size: int = 16  # 降低批次大小，增加泛化
    num_epochs: int = 150
    learning_rate: float = 2e-5  # 预训练模型需要更小学习率
    weight_decay: float = 5e-3  # 适度正则化
    warmup_epochs: int = 5      # 预训练模型收敛更快，减少warmup
    scheduler: str = "cosine"
    
    # 分层学习率设置（基于base learning_rate的倍数）
    backbone_lr_mult: float = 0.1      # 预训练骨干：10%基础学习率
    text_lr_mult: float = 0.1          # 文本编码器：10%基础学习率
    fusion_lr_mult: float = 1.0        # 融合模块：100%基础学习率
    heads_lr_mult: float = 2.0         # 分类/检索头：200%基础学习率
    adapters_lr_mult: float = 1.5      # 模态适配器：150%基础学习率
    
    # 损失权重 - 重新平衡 
    ce_weight: float = 1.0
    contrastive_weight: float = 0.05  # 预训练模型可以使用小权重对比损失
    
    # 数据增强 - 适度增强（预训练模型更稳定）
    random_flip: bool = True
    random_crop: bool = True  # 重新启用随机裁剪
    color_jitter: bool = True
    random_erase: float = 0.3  # 适度随机擦除
    
    # 模态dropout - 适度泛化
    modality_dropout: float = 0.3  # 适度dropout
    min_modalities: int = 1  # 允许单模态训练
    
    # 设备和并行
    device: str = "cuda"
    num_workers: int = 2
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