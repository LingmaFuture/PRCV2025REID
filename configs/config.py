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
    
    # 模型相关 - 基于CLIP-B/16统一编码器
    clip_model_name: str = "openai/clip-vit-base-patch16"  # CLIP-B/16作为统一编码器
    use_clip_backbone: bool = True                         # 使用CLIP预训练权重
    
    # 统一的融合维度（CLIP text projection_dim=512，vision通过projection头对齐）
    fusion_dim: int = 512
    vision_hidden_dim: int = 768  # CLIP ViT-B/16 hidden dimension
    
    # ===== MER (Modality-Expert Router) 配置 =====
    enable_mer: bool = True         # 启用MER模态路由LoRA
    mer_lora_rank: int = 4          # MER LoRA秩r=4
    mer_lora_alpha: float = 1.0     # MER LoRA缩放因子
    
    # 支持的模态列表（各自独立tokenizer + MER路由）
    modalities: List[str] = field(default_factory=lambda: ['rgb', 'ir', 'cpencil', 'sketch', 'text'])
    
    # 各模态tokenizer配置
    patch_size: int = 16            # patch大小（对齐CLIP-B/16）
    # rgb: 3通道, ir/sketch: 1通道, cpencil: 3通道, text: 使用CLIP tokenizer

    # 文本编码器配置（使用CLIP内置文本编码器）
    freeze_text_backbone: bool = False  # 允许CLIP文本编码器微调
    
    # ViT DropPath 正则化参数
    drop_path: float = 0.15  # DropPath率，抑制过拟合
    
    image_size: int = 224  # 匹配ViT模型要求的输入尺寸
    feature_dim: int = 2048  # ResNet50特征维度（兼容保留）
    hidden_dim: int = 512  
    # num_classes 将在训练时根据实际训练集ID数量动态设置
    dropout_rate: float = 0.5  # 增强dropout
    
    # 训练相关 - CLIP统一编码器微调超参数
    batch_size: int = 32
    num_epochs: int = 150
    
    # 分层学习率设置（MER和tokenizer需要更大学习率）
    base_learning_rate: float = 1e-5     # CLIP backbone基础学习率
    mer_learning_rate: float = 5e-5      # MER LoRA学习率  
    tokenizer_learning_rate: float = 5e-5 # 非共享tokenizer学习率
    fusion_learning_rate: float = 5e-5    # 融合层学习率
    
    weight_decay: float = 1e-4
    warmup_epochs: int = 15
    scheduler: str = "cosine"
    
    # 调度器与稳定性
    conservative_factor: float = 0.7
    adaptive_gradient_clip: bool = True
    stability_monitoring: bool = True
    
    # 损失权重（SDM+ID分类组合损失）
    ce_weight: float = 1.0  # ID分类损失权重α=1.0（论文要求）
    contrastive_weight: float = 0.1  # SDM对齐损失权重
    
    # SDM相关配置
    sdm_semantic_dim: int = 512  # SDM语义分离特征维度
    sdm_num_heads: int = 8       # SDM注意力头数
    sdm_temperature: float = 0.1  # RGB锚定对齐温度参数
    sdm_margin: float = 0.3      # 对齐损失边界参数
    
    # 轻量特征混合器配置
    fusion_num_heads: int = 8     # 融合模块多头注意力头数
    fusion_mlp_ratio: float = 2.0  # 融合模块MLP扩展比例
    fusion_dropout: float = 0.1   # 融合模块dropout率
    
    # 特征范数正则化参数（收紧范数控制，防大范数背答案）
    feature_target_norm: float = 10.0    # 目标特征范数
    feature_norm_band: float = 3.0       # 收紧容忍带宽
    feature_norm_penalty: float = 2e-3   # 提高正则化权重
    
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
    eval_freq: int = 15  # 与注释一致：每20轮评估一次
    eval_sample_ratio: float = 0.3  # 采样评估比例（mAP快速估算）
    
    # 验证和推理相关配置
    inference_batch_size: int = 32
    best_model_path: str = "./checkpoints/best_model.pth"
    
    # 加速优化开关
    use_modal_batching: bool = True  # 启用按模态批量前向传播（优化A）
    
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
