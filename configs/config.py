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
    
    # 训练相关 - guide4.py: 化简流程确保梯度流，临时设置accumulation=1
    batch_size: int = 8  # guide3.md: 降低为P×K=4×2，更容易凑齐正对
    # guide9.md Step 2 & guide16.md: 采样器配置，确保K≥2但允许更宽松约束
    num_ids_per_batch: int = 3      # P = 每个batch中的ID数量，guide16建议从4降到3
    num_instances: int = 2          # K = 每个ID的实例数量，强制≥2
    
    # guide18: 采样器配置（防止再早停）
    allow_id_reuse: bool = True     # 允许同epoch内ID复用，防止采样耗尽
    sampling_fallback: bool = True  # 无法满足约束时是否回退到随机采样
    min_modal_coverage: float = 0.6 # 跨模态覆盖率最低要求
    instances_per_id: int = 2       # K - 每个ID的实例数量，与num_instances保持一致
    gradient_accumulation_steps: int = 1  # guide4.py: 临时化简为1，确保梯度流正常
    freeze_backbone: bool = True  # 冻结 CLIP 主干，只训练 LoRA 和特定模块
    num_epochs: int = 60   # 按清单推荐：总60epoch
    
    # 分层学习率设置（保守配置，优先跑通）
    base_learning_rate: float = 5e-6     # 降低CLIP学习率，更稳定
    mer_learning_rate: float = 2e-5      # 降低MER学习率
    tokenizer_learning_rate: float = 2e-5 # 降低tokenizer学习率  
    fusion_learning_rate: float = 2e-5    # 降低融合层学习率
    
    # guide6.md: 分类头LR降档，防权重爆涨
    head_learning_rate: float = 3e-3     # guide6.md: 从Epoch 2起把head LR调到3e-3
    head_lr_warmup_epochs: int = 2       # guide6.md: 从Epoch 2开始降档
    
    weight_decay: float = 1e-4
    warmup_epochs: int = 5      # 前5个epoch线性warmup
    scheduler: str = "cosine"   # warmup后cosine衰减
    
    # 调度器与稳定性
    conservative_factor: float = 0.7
    adaptive_gradient_clip: bool = True
    stability_monitoring: bool = True
    
    # SDM损失权重配置（按文档要求）
    ce_weight: float = 1.0  # ID分类损失权重α=1.0（论文要求）
    
    # SDM权重调度配置 - guide6.md: 从Epoch 2起启用SDM，起始权重0.1
    sdm_weight_warmup_epochs: int = 1   # guide9.md: 修复边界，从Epoch 2起启用SDM
    sdm_weight_schedule: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])  # guide6.md: 渐进权重
    sdm_weight_initial: float = 0.1     # guide6.md: 从0.1开始
    sdm_weight_final: float = 0.5       # guide6.md: 最终0.5，保守稳定
    sdm_weight_max: float = 0.5         # guide6.md: 最大权重限制在0.5
    
    # 当前使用的SDM权重（训练过程中动态调整）
    contrastive_weight: float = 0.0     # 初始为0，按调度器调整
    
    # SDM相关配置（按文档要求）
    sdm_semantic_dim: int = 512  # SDM语义分离特征维度
    sdm_num_heads: int = 8       # SDM注意力头数
    sdm_temperature: float = 0.2  # SDM损失温度参数（修复后稳定版本）
    
    # 温度参数配置（提高稳定性）- guide3.md推荐0.15-0.2
    sdm_init_temperature: float = 0.18  # guide3.md推荐范围内的初始温度
    sdm_final_temperature: float = 0.16 # guide3.md推荐范围内的稳定温度
    sdm_fallback_temperature: float = 0.20  # guide3.md推荐范围内的回退温度
    
    # 可学习温度配置
    sdm_learnable_temp: bool = True     # 使用可学习温度
    sdm_temp_warmup_epochs: int = 3     # 温度调整的epoch数
    
    # 轻量特征混合器配置
    fusion_num_heads: int = 8     # 融合模块多头注意力头数
    fusion_mlp_ratio: float = 2.0  # 融合模块MLP扩展比例
    fusion_dropout: float = 0.1   # 融合模块dropout率
    
    # 简化损失：移除特征范数正则化，只保留CE+SDM核心损失
    
    # 数据增强
    random_flip: bool = True
    random_crop: bool = True
    color_jitter: bool = True
    random_erase: float = 0.3
    
    # 模态dropout（按优化清单热身期配置）
    modality_dropout: float = 0.15
    modality_dropout_warmup_epochs: int = 3  # 前3个epoch关闭dropout等训练稳定
    min_modalities: int = 1
    
    # guide6.md: 强配对采样器配置
    require_modal_pairs: bool = True    # guide6.md: 立即开启强配对
    modal_pair_retry_limit: int = 3     # guide6.md: 软退路重试次数
    modal_pair_fallback_ratio: float = 0.3  # guide6.md: 软退路比例30%
    
    # guide6.md: 跨批记忆库配置
    sdm_memory_steps: int = 6           # guide6.md: 缓存近N=4~8个step的RGB特征
    sdm_memory_enabled: bool = True     # guide6.md: 启用跨批记忆库
    
    # guide6.md: 健康线监控配置
    pair_coverage_target: float = 0.85  # guide6.md: pair_coverage_mavg目标≥0.85
    pair_coverage_window: int = 100     # guide6.md: 滑窗100 step
    
    # 设备和并行（按guide6.md优化DataLoader）
    device: str = "cuda"
    num_workers: int = 2  # guide6.md: 适中的工作进程数
    pin_memory: bool = True  # 开启内存锁定配合non_blocking加速传输
    persistent_workers: bool = True  # guide6.md: 保持工作进程，避免重复创建
    prefetch_factor: int = 2  # guide6.md: 预取因子，平衡内存和性能
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 20
    eval_freq: int = 15  # 与注释一致：每20轮评估一次
    eval_sample_ratio: float = 0.3  # 采样评估比例（mAP快速估算）
    
    # guide13.md: 仅评测四单模态 + 四模态，跳过双/三模态组合
    eval_include_patterns: List[str] = field(default_factory=lambda: [
        "single/nir", "single/sk", "single/cp", "single/text", "quad/nir+sk+cp+text"
    ])
    
    # guide13.md & guide15.md: 确保只在每个epoch结束评测，不在训练步骤中评测
    eval_every_n_epoch: int = 1
    eval_every_n_steps: int = 0  # 必须为0，禁用步数级评测
    do_eval: bool = True  # 是否进行评测
    eval_after_steps: Optional[int] = None  # 首轮体检步数阈值，设为None禁用
    
    # guide14.md: 评测特征缓存配置
    eval_cache_dir: str = "./.eval_cache"
    eval_cache_tag: str = "val_v1"  # 数据或预处理改了就换这个tag
    
    # 验证和推理相关配置
    inference_batch_size: int = 8   # 推理批次与训练保持一致
    best_model_path: str = "./checkpoints/best_model.pth"
    
    # guide6.md: label smoothing配置
    label_smoothing: float = 0.1    # guide6.md: 建议加label smoothing=0.1，更稳
    
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
