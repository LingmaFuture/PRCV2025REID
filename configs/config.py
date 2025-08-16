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
    
    # 模型相关 - 简化架构，专注核心功能
    backbone: str = "resnet50"      # 支持 "resnet50" 或 "vit_base_patch16_224"
    use_pretrained_vision: bool = False

    # 统一的融合维度（视觉与文本都会被投到这个维度）
    fusion_dim: int = 768

    # 文本模型配置
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text: bool = True
    
    image_size: int = 224  # 匹配ViT模型要求的输入尺寸
    feature_dim: int = 2048  # ResNet50特征维度
    hidden_dim: int = 512  
    num_classes: int = 999
    dropout_rate: float = 0.5  # 增强dropout
    
    # 训练相关 - 优化超参数
    batch_size: int = 16  # 降低批次大小，增加泛化
    num_epochs: int = 150
    learning_rate: float = 1e-4  # 进一步降低学习率
    weight_decay: float = 1e-2  # 极强正则化
    warmup_epochs: int = 10
    scheduler: str = "cosine"
    
    # 损失权重 - 重新平衡 
    ce_weight: float = 1.0
    contrastive_weight: float = 0.0  # 暂时完全关闭对比损失
    
    # 数据增强 - 增强泛化能力
    random_flip: bool = True
    random_crop: bool = True  # 重新启用随机裁剪
    color_jitter: bool = True
    random_erase: float = 0.5  # 增加随机擦除概率
    
    # 模态dropout - 增强泛化能力
    modality_dropout: float = 0.5  # 大幅增加dropout概率
    min_modalities: int = 1  # 允许单模态训练
    
    # 设备和并行
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 20
    eval_freq: int = 10  # 降低评估频率，从每10轮改为每20轮
    eval_sample_ratio: float = 0.3  # 采样评估，只用30%数据进行快速mAP估算
    
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