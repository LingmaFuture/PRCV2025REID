# configs/config.py
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
    freeze_text: bool = False  # 启用文本编码器微调提升跨模态对齐
    
    image_size: int = 224  # 匹配ViT模型要求的输入尺寸
    feature_dim: int = 2048  # ResNet50特征维度（兼容保留）
    hidden_dim: int = 512  
    # num_classes 将在训练时根据实际训练集ID数量动态设置
    dropout_rate: float = 0.5  # 增强dropout
    
    # 训练相关 - 预训练模型微调超参数 (优化后)
    batch_size: int = 32
    num_epochs: int = 150
    learning_rate: float = 3e-4  # 降低基础学习率以稳定对比学习
    weight_decay: float = 1e-4
    warmup_epochs: int = 15
    scheduler: str = "cosine"
    
    # 调度器与稳定性
    conservative_factor: float = 0.7
    adaptive_gradient_clip: bool = True
    stability_monitoring: bool = True
    
    # 损失权重（进一步降低对比损失权重以提高稳定性）
    ce_weight: float = 1.0
    contrastive_weight: float = 0.02  # 进一步降低以避免后期不稳定
    
    # 特征范数正则化参数（控制融合特征范数，提高训练稳定性）
    feature_target_norm: float = 10.0    # 目标特征范数
    feature_norm_band: float = 4.0       # 容忍带宽
    feature_norm_penalty: float = 1e-3   # 正则化权重（保守设置）
    
    # 数据增强
    random_flip: bool = True
    random_crop: bool = True
    color_jitter: bool = True
    random_erase: float = 0.3
    
    # 模态dropout（降低以提高特征稳定性）
    modality_dropout: float = 0.15
    min_modalities: int = 1
    
    # 设备和并行
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 20
    eval_freq: int = 20  # 与注释一致：每20轮评估一次
    eval_sample_ratio: float = 0.3  # 采样评估比例（mAP快速估算）
    
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
