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
    
    # 训练相关 - 16GB显存安全配置（只有micro-batch影响显存）
    batch_size: int = 8   # micro-batch大小：显存安全优先
    gradient_accumulation_steps: int = 4  # 保持等效batch_size=32
    freeze_backbone: bool = True  # 冻结 CLIP 主干，只训练 LoRA 和特定模块
    num_epochs: int = 150
    
    # 分层学习率设置（修复CE收敛问题）
    base_learning_rate: float = 5e-6     # 降低CLIP backbone学习率
    mer_learning_rate: float = 3e-5      # 降低MER LoRA学习率  
    tokenizer_learning_rate: float = 3e-5 # 降低非共享tokenizer学习率
    fusion_learning_rate: float = 3e-5    # 降低融合层学习率
    
    weight_decay: float = 1e-4
    warmup_epochs: int = 5      # 缩短warmup避免学习率过高
    scheduler: str = "cosine"
    
    # 调度器与稳定性
    conservative_factor: float = 0.7
    adaptive_gradient_clip: bool = True
    stability_monitoring: bool = True
    
    # SDM损失权重配置（按文档要求）
    ce_weight: float = 1.0  # ID分类损失权重α=1.0（论文要求）
    
    # SDM权重调度配置
    sdm_weight_warmup_epochs: int = 3   # 前3个epoch热身，λ_sdm = 0
    sdm_weight_initial: float = 0.5     # 热身后的初始权重
    sdm_weight_final: float = 1.0       # 稳定后的目标权重
    sdm_weight_max: float = 1.5         # 最大权重上限
    
    # 当前使用的SDM权重（训练过程中动态调整）
    contrastive_weight: float = 0.0     # 初始为0，按调度器调整
    
    # SDM相关配置（按文档要求）
    sdm_semantic_dim: int = 512  # SDM语义分离特征维度
    sdm_num_heads: int = 8       # SDM注意力头数
    
    # 温度参数配置
    sdm_init_temperature: float = 0.12  # 初始温度，按文档建议
    sdm_final_temperature: float = 0.10 # 稳定后的温度
    sdm_fallback_temperature: float = 0.15  # 出现不稳定时的回退温度
    
    # 可学习温度配置
    sdm_learnable_temp: bool = True     # 使用可学习温度
    sdm_temp_warmup_epochs: int = 3     # 温度调整的epoch数
    
    # 轻量特征混合器配置
    fusion_num_heads: int = 8     # 融合模块多头注意力头数
    fusion_mlp_ratio: float = 2.0  # 融合模块MLP扩展比例
    fusion_dropout: float = 0.1   # 融合模块dropout率
    
    # 特征范数正则化参数（修复CE收敛问题）
    feature_target_norm: float = 5.0     # 进一步降低目标特征范数
    feature_norm_band: float = 1.0       # 收紧容忍带宽
    feature_norm_penalty: float = 5e-3   # 提高正则化权重防止特征范数过大
    
    # 数据增强
    random_flip: bool = True
    random_crop: bool = True
    color_jitter: bool = True
    random_erase: float = 0.3
    
    # 模态dropout（降低以提高特征稳定性）
    modality_dropout: float = 0.15
    min_modalities: int = 1
    
    # 设备和并行 (16GB显存优化)
    device: str = "cuda"
    num_workers: int = 1  # 降低数据加载器工作进程数以节省内存
    pin_memory: bool = False  # 关闭内存锁定以节省内存
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 20
    eval_freq: int = 15  # 与注释一致：每20轮评估一次
    eval_sample_ratio: float = 0.3  # 采样评估比例（mAP快速估算）
    
    # 验证和推理相关配置
    inference_batch_size: int = 8   # 推理批次与训练保持一致
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
