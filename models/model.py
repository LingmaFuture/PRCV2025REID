# models/model.py
"""
重构后的多模态ReID模型：CLIP-B/16统一编码器 + MER模态路由 + SDM语义分离
保持与原架构的兼容性，同时集成新的设计理念
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

# 初始化logger
logger = logging.getLogger(__name__)

from .clip_backbone import CLIPUnifiedEncoder

# 导入原有的SDM和损失函数组件
class SemanticDisentanglementModule(nn.Module):
    """
    语义分离模块：将各模态特征投影到语义空间，便于与vis对齐
    保持原有实现，与新架构兼容
    """
    def __init__(self, input_dim: int = 512, semantic_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.semantic_dim = semantic_dim
        
        # 语义分离的多头注意力
        self.semantic_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 投影到语义空间
        self.semantic_proj = nn.Sequential(
            nn.Linear(input_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim, semantic_dim)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] 输入特征
        Returns:
            [B, semantic_dim] 语义分离后的特征
        """
        # 为注意力机制添加序列维度
        x_seq = x.unsqueeze(1)  # [B, 1, input_dim]
        
        # 自注意力语义分离
        attn_out, _ = self.semantic_attn(x_seq, x_seq, x_seq)  # [B, 1, input_dim]
        attn_out = attn_out.squeeze(1)  # [B, input_dim]
        
        # 残差连接
        x = x + attn_out
        
        # 投影到语义空间
        semantic_features = self.semantic_proj(x)  # [B, semantic_dim]
        
        return semantic_features


class RGBAnchoredAlignmentLoss(nn.Module):
    """vis锚定对齐损失：推动所有查询模态对齐到vis目标表征"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, alpha: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin 
        self.alpha = alpha
        
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算vis锚定对齐损失
        Args:
            modality_features: 各模态原始特征字典
            fused_features: 融合后特征
            labels: ID标签
        Returns:
            对齐损失
        """
        if 'vis' not in modality_features:
            return torch.tensor(0.0, device=fused_features.device, requires_grad=True)
        
        vis_features = modality_features['vis']  # [B, D] 可见光目标特征
        total_loss = torch.tensor(0.0, device=fused_features.device, requires_grad=True)
        num_modalities = 0
        
        # 计算每个非可见光模态与可见光的对齐损失
        for modality, features in modality_features.items():
            if modality == 'vis':
                continue
                
            # 归一化特征以稳定相似度计算
            features_norm = F.normalize(features, p=2, dim=1)
            vis_features_norm = F.normalize(vis_features, p=2, dim=1)
            
            # 对比损失：相同ID拉近，不同ID推远
            sim_matrix = torch.matmul(features_norm, vis_features_norm.T) / self.temperature
            
            # 构建正负样本掩码
            batch_size = features.shape[0]
            labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
            pos_mask = (labels_expand == labels_expand.T).float()
            neg_mask = 1.0 - pos_mask
            
            # 避免对角线上的自相似
            eye_mask = torch.eye(batch_size, device=features.device)
            pos_mask = pos_mask * (1.0 - eye_mask)  # 移除对角线
            
            # 正样本损失（相同ID应该相似）- 使用数值稳定的logsumexp
            if pos_mask.sum() > 0:
                pos_sim = sim_matrix * pos_mask
                # 将非正样本位置设为极小值，避免影响logsumexp
                pos_logits = pos_sim + (1.0 - pos_mask) * (-1e9)
                pos_loss = -torch.logsumexp(pos_logits, dim=1).mean()
            else:
                pos_loss = torch.tensor(0.0, device=features.device)
            
            # 负样本损失（不同ID应该不相似）- 使用数值稳定的logsumexp
            if neg_mask.sum() > 0:
                neg_sim = sim_matrix * neg_mask - self.margin
                # 将非负样本位置设为极小值，避免影响logsumexp
                neg_logits = neg_sim + (1.0 - neg_mask) * (-1e9)
                neg_loss = torch.logsumexp(neg_logits, dim=1).mean()
            else:
                neg_loss = torch.tensor(0.0, device=features.device)
            
            modality_loss = pos_loss + neg_loss
            
            # 检查是否为NaN
            if torch.isnan(modality_loss):
                continue
                
            total_loss = total_loss + modality_loss
            num_modalities += 1
        
        if num_modalities > 0:
            return total_loss / num_modalities
        else:
            return torch.tensor(0.0, device=fused_features.device, requires_grad=True)


class SDMContrastiveLoss(nn.Module):
    """SDM对比损失：结合语义分离和vis锚定对齐"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, alpha: float = 1.0):
        super().__init__()
        self.vis_alignment = RGBAnchoredAlignmentLoss(temperature, margin, alpha)  # 保持类名兼容性
        
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """计算SDM对比损失"""
        alignment_loss = self.vis_alignment(modality_features, fused_features, labels)
        return alignment_loss


class FeatureFusion(nn.Module):
    """轻量特征融合器：多头注意力 + MLP混合器"""
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP混合器
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
    def forward(self, features: List[torch.Tensor], masks: List[torch.Tensor] = None) -> torch.Tensor:
        """
        融合多个模态特征，支持mask
        Args:
            features: 特征列表，每个特征shape为[B, feature_dim]
            masks: 掩码列表，每个掩码shape为[B]，1表示有效，0表示无效
        Returns:
            [B, feature_dim] 融合后特征
        """
        if len(features) == 0:
            raise ValueError("No features to fuse")
        
        if len(features) == 1:
            return features[0]
        
        # 堆叠特征：[B, num_modalities, feature_dim]
        stacked_features = torch.stack(features, dim=1)
        B, M, D = stacked_features.shape
        
        # 处理mask
        key_padding_mask = None
        if masks is not None:
            # 堆叠mask: [B, num_modalities]
            stacked_masks = torch.stack(masks, dim=1)  # [B, M]
            # 创建key_padding_mask用于attention（True表示要忽略的位置）
            key_padding_mask = ~stacked_masks.bool()  # [B, M]
            
            # ❶ 防全遮罩 → 引发MHA NaN
            all_masked = key_padding_mask.all(dim=1)  # [B] 检查哪些样本被全部mask
            if all_masked.any():
                # 至少保留第一个位置不被mask，避免softmax(-inf) → NaN
                key_padding_mask[all_masked, 0] = False
                # 为全遮罩样本提供一个稳定的占位特征（使用第一个模态的均值）
                global_mean = stacked_features[~all_masked].mean(dim=[0, 1], keepdim=True)  # [1, 1, D]
                if global_mean.numel() == 0:  # 如果所有样本都被mask，使用零向量
                    global_mean = torch.zeros(1, 1, D, device=stacked_features.device)
                stacked_features[all_masked, 0] = global_mean.squeeze(0)
        
        # 多头注意力融合（带mask）
        attn_out, _ = self.multihead_attn(
            stacked_features, stacked_features, stacked_features,
            key_padding_mask=key_padding_mask  # 忽略mask=0的位置
        )  # [B, M, D]
        
        # 残差连接 + 层归一化
        attn_out = self.norm1(stacked_features + attn_out)
        
        # MLP混合器 + 残差连接
        mlp_out = self.mlp(attn_out)
        fused_features = self.norm2(attn_out + mlp_out)
        
        # ❷ 保险丝：清理任何残留的NaN/Inf
        fused_features = torch.nan_to_num(fused_features, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 带mask的加权平均池化
        if masks is not None:
            # 将无效位置的特征设为0
            valid_mask = stacked_masks.unsqueeze(-1).float()  # [B, M, 1]
            masked_features = fused_features * valid_mask
            
            # 计算有效模态数量
            valid_counts = stacked_masks.sum(dim=1, keepdim=True).float()  # [B, 1]
            valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
            
            # 加权平均
            final_features = masked_features.sum(dim=1) / valid_counts  # [B, D]
        else:
            # 全局平均池化
            final_features = fused_features.mean(dim=1)  # [B, D]
        
        return final_features


class BNNeck(nn.Module):
    """BatchNorm Neck：在特征和分类器之间的BN层"""
    
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        
        # BN层
        self.bn = nn.BatchNorm1d(in_dim)
        self.bn.bias.requires_grad_(False)  # 禁用bias
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 分类器
        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        
        # 初始化
        nn.init.normal_(self.classifier.weight, std=0.001)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, in_dim] 输入特征
        Returns:
            bn_features: [B, in_dim] BN后特征（用于ReID）
            logits: [B, num_classes] 分类logits
        """
        bn_features = self.bn(features)
        
        # guide3.md: 添加L2归一化控制特征范数到~8-10，避免Feat(BN)=22.5过大
        bn_features_normalized = F.normalize(bn_features, p=2, dim=1) * 8.0  # 目标范数=8
        
        dropped_features = self.dropout(bn_features_normalized)
        logits = self.classifier(dropped_features)
        
        return bn_features_normalized, logits


class CLIPBasedMultiModalReIDModel(nn.Module):
    """
    基于CLIP-B/16的多模态ReID模型
    架构：CLIP统一编码器 + MER模态路由 + SDM语义分离 + 特征融合 + ID分类
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        
        # 训练状态跟踪
        self.current_epoch = 0  # 用于控制modality_dropout的热身期
        
        # guide6.md: 跨批记忆库初始化
        if getattr(config, 'sdm_memory_enabled', True):
            from collections import deque
            self.sdm_memory = deque(maxlen=getattr(config, 'sdm_memory_steps', 6))
        else:
            self.sdm_memory = None
        
        # 模态配置
        self.modalities = getattr(config, 'modalities', ['vis', 'nir', 'sk', 'cp', 'text'])
        self.vision_modalities = [m for m in self.modalities if m != 'text']
        
        # 特征维度配置
        self.fusion_dim = getattr(config, 'fusion_dim', 512)
        self.vision_hidden_dim = getattr(config, 'vision_hidden_dim', 768)
        
        # ===== CLIP统一编码器 + MER路由 =====
        self.clip_encoder = CLIPUnifiedEncoder(
            clip_model_name=getattr(config, 'clip_model_name', 'openai/clip-vit-base-patch16'),
            modalities=self.modalities,
            vision_hidden_dim=self.vision_hidden_dim,
            text_hidden_dim=512,  # CLIP text hidden dim
            fusion_dim=self.fusion_dim,
            lora_rank=getattr(config, 'mer_lora_rank', 4),
            lora_alpha=getattr(config, 'mer_lora_alpha', 1.0),
            drop_path=getattr(config, 'drop_path', 0.0),
            freeze_text_backbone=getattr(config, 'freeze_text_backbone', False)
        )
        
        # ===== SDM语义分离模块 =====
        self.sdm_module = SemanticDisentanglementModule(
            input_dim=self.fusion_dim,
            semantic_dim=getattr(config, "sdm_semantic_dim", 512),
            num_heads=getattr(config, "sdm_num_heads", 8)
        )
        
        # ===== 特征融合器 =====
        self.feature_fusion = FeatureFusion(
            feature_dim=self.fusion_dim,
            num_heads=getattr(config, "fusion_num_heads", 8),
            mlp_ratio=getattr(config, "fusion_mlp_ratio", 2.0),
            dropout=getattr(config, "fusion_dropout", 0.1)
        )
        
        # ===== BN Neck + 分类器 =====
        # num_classes将在训练时动态设置
        self.num_classes = None
        self.bn_neck = None
        
        # ===== 损失函数和SDM参数 =====
        self.sdm_temperature = getattr(config, "sdm_temperature", 0.2)  # 保存温度参数
        self.sdm_contrastive_loss = SDMContrastiveLoss(
            temperature=getattr(config, "sdm_temperature", 0.1),
            margin=getattr(config, "sdm_margin", 0.3),
            alpha=1.0
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # guide6.md: 增加到0.1更稳
        
        # 损失权重
        self.ce_weight = getattr(config, 'ce_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        
        # 简化损失：移除特征范数正则化参数
        
        # ===== 可学习的null token占位符 =====
        # 为每个模态创建可学习的null token，用于缺失模态的占位
        self.null_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, self.fusion_dim) * 0.02)
            for modality in self.modalities
        })
        
        logger.info(f"初始化CLIP+MER多模态ReID模型完成")
        logger.info(f"   - 支持模态: {self.modalities}")
        logger.info(f"   - 融合维度: {self.fusion_dim}")
        logger.info(f"   - MER LoRA rank: {getattr(config, 'mer_lora_rank', 4)}")
    
    def set_num_classes(self, num_classes: int):
        """动态设置类别数并初始化分类器"""
        self.num_classes = num_classes
        self.bn_neck = BNNeck(
            in_dim=self.fusion_dim,
            num_classes=num_classes,
            dropout=getattr(self.config, 'dropout_rate', 0.5)
        ).to(self.device)
        
        logger.info(f"设置分类器：{num_classes} 个ID类别")
    
    def forward(self, 
                images: Optional[Dict[str, torch.Tensor]] = None,
                texts: Optional[List[str]] = None,
                modality_masks: Optional[Dict[str, torch.Tensor]] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        模型前向传播
        修复：使用null_token占位+精确mask，避免零特征污染
        Args:
            images: 图像字典 {modality: [B,C,H,W]}
            texts: 文本列表 [str, ...] (可包含空字符串)
            modality_masks: 模态掩码 {modality: [B]} (1=有效, 0=无效)
            return_features: 是否返回中间特征
        Returns:
            输出字典，包含logits、features等
        """
        # 确定batch size
        batch_size = None
        if images is not None:
            for img_tensor in images.values():
                batch_size = img_tensor.shape[0]
                break
        elif texts is not None:
            batch_size = len(texts)
        
        if batch_size is None:
            raise ValueError("无法确定batch size")
        
        modality_features = {}
        raw_modality_features = {}
        feature_masks = {}  # 记录每个模态的mask
        
        # ===== 视觉模态编码 =====
        if images is not None:
            for modality, img_tensor in images.items():
                if modality in self.vision_modalities:
                    # 获取该模态的mask
                    mask = None
                    if modality_masks is not None:
                        # 需要从原始模态名映射回来
                        original_modality = None
                        # 同名映射：模态名直接对应，无需转换
                        original_modality = modality
                        if original_modality and original_modality in modality_masks:
                            mask = modality_masks[original_modality]  # [B]
                    
                    if mask is not None and mask.sum() > 0:
                        # 有部分有效样本，需要selective编码
                        valid_indices = mask.bool()
                        valid_imgs = img_tensor[valid_indices]  # [valid_count, C, H, W]
                        
                        if valid_imgs.shape[0] > 0:
                            # 编码有效样本
                            valid_features = self.clip_encoder.encode_vision(
                                valid_imgs.to(self.device), modality
                            )  # [valid_count, fusion_dim]
                            
                            # 创建完整的特征tensor，用null_token填充无效位置
                            # 修复AMP下的dtype不匹配：将null_token对齐为valid_features的dtype
                            null_tok = self.null_tokens[modality].to(device=self.device, dtype=valid_features.dtype)
                            full_features = null_tok.expand(batch_size, -1).clone()
                            full_features[valid_indices] = valid_features
                        else:
                            # 全部无效，使用null_token
                            full_features = self.null_tokens[modality].expand(batch_size, -1)
                    else:
                        # 全部无效或没有mask，使用null_token
                        full_features = self.null_tokens[modality].expand(batch_size, -1)
                        mask = torch.zeros(batch_size, device=self.device)
                    
                    raw_modality_features[modality] = full_features
                    feature_masks[modality] = mask if mask is not None else torch.ones(batch_size, device=self.device)
                    
                    # SDM语义分离（仅训练时）
                    if self.training:
                        semantic_features = self.sdm_module(full_features)
                        modality_features[modality] = semantic_features
                    else:
                        modality_features[modality] = full_features
        
        # ===== 文本模态编码 =====
        if texts is not None and len(texts) > 0:
            # 获取文本mask
            text_mask = None
            if modality_masks is not None and 'text' in modality_masks:
                text_mask = modality_masks['text']  # [B]
            
            # 始终编码所有文本（包括空字符串），让CLIP处理
            text_features = self.clip_encoder.encode_text(texts)  # [B, fusion_dim]
            
            # 如果有mask，用null_token替换无效位置
            if text_mask is not None:
                invalid_indices = ~text_mask.bool()
                if invalid_indices.any():
                    # 修复AMP下的dtype不匹配：将null_token对齐为text_features的dtype
                    text_features[invalid_indices] = self.null_tokens['text'].to(device=self.device, dtype=text_features.dtype).expand(invalid_indices.sum(), -1)
            else:
                text_mask = torch.ones(batch_size, device=self.device)
            
            raw_modality_features['text'] = text_features
            feature_masks['text'] = text_mask
            
            # SDM语义分离（仅训练时）
            if self.training:
                semantic_text_features = self.sdm_module(text_features)
                modality_features['text'] = semantic_text_features
            else:
                modality_features['text'] = text_features
        
        # ===== 特征融合 =====
        if len(modality_features) == 0:
            raise ValueError("至少需要提供一种有效模态的输入")
        
        # 模态dropout（训练时随机丢弃部分模态）- 修复版：防止样本被"打空" + 热身期控制
        if self.training:
            modality_dropout = getattr(self.config, 'modality_dropout', 0.0)
            min_modalities = getattr(self.config, 'min_modalities', 1)
            
            # ❺ 前2-3个epoch关闭modality_dropout，等训练稳定
            warmup_epochs = getattr(self.config, 'modality_dropout_warmup_epochs', 3)
            if self.current_epoch <= warmup_epochs:
                modality_dropout = 0.0  # 热身期强制关闭dropout
            
            if modality_dropout > 0 and len(modality_features) > min_modalities:
                # 保存原始状态用于回退
                original_modality_features = modality_features.copy()
                original_feature_masks = feature_masks.copy()
                
                # 永不drop 'vis'，优先保留主模态
                keep_modalities = []
                keep_masks = []
                for mod, feat in modality_features.items():
                    if mod == 'vis' or torch.rand(1).item() > modality_dropout:
                        keep_modalities.append((mod, feat))
                        keep_masks.append((mod, feature_masks[mod]))
                
                if len(keep_modalities) >= min_modalities:
                    # 检查dropout后是否会造成样本级"全遮罩"
                    temp_modality_features = dict(keep_modalities)
                    temp_feature_masks = dict(keep_masks)
                    
                    # 检查每个样本是否至少有一个有效模态
                    stacked_masks = torch.stack([temp_feature_masks[mod] for mod in temp_modality_features], dim=1)
                    per_sample_valid = stacked_masks.any(dim=1)  # [B]
                    
                    if per_sample_valid.all():
                        # 安全：所有样本都有至少一个有效模态
                        modality_features = temp_modality_features
                        feature_masks = temp_feature_masks
                    else:
                        # 危险：有样本会被完全遮罩，取消本次dropout
                        modality_features = original_modality_features
                        feature_masks = original_feature_masks
        
        # 融合多模态特征（带mask）
        feature_list = list(modality_features.values())
        mask_list = [feature_masks[mod] for mod in modality_features.keys()]
        
        if len(feature_list) == 1:
            fused_features = feature_list[0]
        else:
            fused_features = self.feature_fusion(feature_list, mask_list)  # [B, fusion_dim]
        
        # ===== ID分类 =====
        outputs = {
            'features': fused_features,  # 融合后特征（BN前）
            'raw_modality_features': raw_modality_features,
            'modality_features': modality_features
        }
        
        if self.bn_neck is not None:
            bn_features, logits = self.bn_neck(fused_features)
            outputs.update({
                'bn_features': bn_features,  # BN后特征（推荐用于检索）
                'logits': logits
            })
        
        if return_features:
            outputs['intermediate_features'] = {
                'raw_modality': raw_modality_features,
                'semantic_modality': modality_features,
                'fused': fused_features
            }
        
        # 输出feature_masks供损失计算时过滤使用
        outputs['feature_masks'] = feature_masks
        
        # guide6.md: 跨批记忆库更新（在compute_loss中处理，这里只缓存特征）
        if self.sdm_memory is not None and self.training:
            # 缓存当前batch的vis特征（标签在compute_loss中处理）
            vis_features = raw_modality_features.get('vis', None)
            vis_mask = feature_masks.get('vis', None)
            
            if vis_features is not None and vis_mask is not None:
                # 找到有效的vis样本
                vis_valid_idx = (vis_mask > 0).squeeze(-1) if vis_mask.dim() > 1 else (vis_mask > 0)
                
                if vis_valid_idx.sum() > 0:
                    # 缓存有效的vis特征（标签在compute_loss中处理）
                    vis_valid_feat = vis_features[vis_valid_idx].detach()
                    self.sdm_memory.append((vis_valid_feat, vis_valid_idx))
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算总损失：ID分类损失 + SDM对齐损失 + 特征范数正则化
        Args:
            outputs: 模型输出
            labels: ID标签
        Returns:
            损失字典
        """
        # ID分类损失 - 添加有效样本计数保护
        if 'logits' not in outputs:
            raise ValueError("模型输出中缺少logits，请确保已设置num_classes")
        
        logits = outputs['logits']
        feature_masks = outputs.get('feature_masks', {})
        
        # 构建有效样本mask：至少有一个模态有效且标签有效
        if feature_masks:
            # 计算每个样本是否至少有一个模态有效
            valid_modality_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            for mod_name, mask in feature_masks.items():
                if mask is not None:
                    mod_valid = (mask > 0).squeeze(-1) if mask.dim() > 1 else (mask > 0)
                    valid_modality_mask = valid_modality_mask | mod_valid
        else:
            # 没有mask信息，假设全部有效
            valid_modality_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
        
        # 添加标签有效性检查
        valid_label_mask = (labels >= 0) & (labels < logits.shape[1])
        valid_ce_mask = valid_modality_mask & valid_label_mask
        valid_ce_cnt = int(valid_ce_mask.sum())
        
        if valid_ce_cnt > 0:
            ce_loss = self.ce_loss(logits[valid_ce_mask], labels[valid_ce_mask])
        else:
            ce_loss = torch.zeros([], device=logits.device)
            logger.warning(f"CE skipped: valid_ce_cnt=0, mod_valid={int(valid_modality_mask.sum())}, label_valid={int(valid_label_mask.sum())}")
        
        # guide5.md: 在warmup期间完全跳过SDM前向计算，节省时间
        use_sdm = (self.current_epoch >= self.config.sdm_weight_warmup_epochs) and (self.contrastive_weight > 0)
        
        if use_sdm:
            # 真正的非负SDM损失：严格按mask过滤，避免占位符污染对齐
            from .sdm_loss import sdm_loss_stable
            raw_modality_features = outputs.get('raw_modality_features', {})
            feature_masks = outputs.get('feature_masks', {})
            
            # 使用fp32计算SDM损失，避免半精度下的数值不稳定  
            with torch.amp.autocast('cuda', enabled=False):
                vis_features = raw_modality_features.get('vis', None)
                vis_mask = feature_masks.get('vis', None)
            
            if vis_features is None or vis_mask is None:
                # 回退：如果没有vis特征或mask，跳过SDM对齐
                sdm_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
            else:
                # 找到有效的vis样本索引
                vis_valid_idx = (vis_mask > 0).squeeze(-1) if vis_mask.dim() > 1 else (vis_mask > 0)
                
                if vis_valid_idx.sum() == 0:
                    # 没有有效vis样本，跳过对齐
                    sdm_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
                else:
                    # 过滤出有效的vis特征和标签
                    vis_valid_feat = vis_features[vis_valid_idx]
                    vis_valid_labels = labels[vis_valid_idx]
                    
                    # guide6.md: 使用跨批记忆库兜底
                    if self.sdm_memory is not None and len(self.sdm_memory) > 0:
                        # 从记忆库中获取历史vis特征和标签
                        mem_feats = []
                        mem_labels = []
                        for mem_feat, mem_label in self.sdm_memory:
                            mem_feats.append(mem_feat)
                            mem_labels.append(mem_label)
                        
                        if mem_feats:
                            # 合并当前batch和记忆库的vis特征
                            mem_feats_cat = torch.cat(mem_feats, dim=0)
                            mem_labels_cat = torch.cat(mem_labels, dim=0)
                            
                            # 扩展当前batch的vis特征和标签
                            vis_valid_feat = torch.cat([vis_valid_feat, mem_feats_cat], dim=0)
                            vis_valid_labels = torch.cat([vis_valid_labels, mem_labels_cat], dim=0)
                    
                    sdm_losses = []
                    # 对每个非vis模态与有效vis做SDM对齐
                    for mod_name, mod_feat in raw_modality_features.items():
                        if mod_name == 'vis':
                            continue
                            
                        mod_mask = feature_masks.get(mod_name, None)
                        if mod_feat is None or mod_mask is None:
                            continue
                            
                        # 找到该模态的有效样本
                        mod_valid_idx = (mod_mask > 0).squeeze(-1) if mod_mask.dim() > 1 else (mod_mask > 0)
                        
                        if mod_valid_idx.sum() == 0:
                            continue  # 该模态没有有效样本
                            
                        # 过滤出有效的模态特征和标签
                        mod_valid_feat = mod_feat[mod_valid_idx]
                        mod_valid_labels = labels[mod_valid_idx]
                        
                        # 构造有效样本间的同身份指示矩阵
                        y = (mod_valid_labels.view(-1, 1) == vis_valid_labels.view(1, -1)).float()
                        
                        # ❸ 轻兜底：检查该模态与vis是否有正对
                        if y.numel() == 0 or y.sum() == 0:
                            if self.training:
                                # 只在训练时警告，推理时静默跳过
                                logger.debug(f"SDM: {mod_name}↔vis无正对，跳过该模态对齐")
                            continue  # 跳过这个模态，不加入sdm_losses
                        
                        # 模态特征 -> vis特征的SDM对齐（天然非负）
                        L = sdm_loss_stable(mod_valid_feat, vis_valid_feat, y, 
                                          tau=self.sdm_temperature)
                        if torch.isfinite(L):
                            sdm_losses.append(L)
                    
                    # 平均所有有效模态的SDM损失
                    if sdm_losses:
                        sdm_loss = torch.stack(sdm_losses).mean()
                    else:
                        # 稳健化改造：没有任何SDM正对时，将SDM损失置零以继续训练
                        sdm_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
                        # 偶发性告警（避免日志过多）
                        if hasattr(self, '_last_no_pairs_warning'):
                            current_time = __import__('time').time()
                            if current_time - self._last_no_pairs_warning > 10.0:  # 10秒打印一次
                                logger.warning("⚠️ 当前batch无SDM正对，已将SDM损失置零以继续训练")
                                self._last_no_pairs_warning = current_time
                        else:
                            self._last_no_pairs_warning = __import__('time').time()
                            logger.warning("⚠️ 当前batch无SDM正对，已将SDM损失置零以继续训练")
        else:
            # guide5.md: warmup期间完全跳过SDM计算
            sdm_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
        
        # ===== 简化损失：只保留CE + SDM =====
        # 检查核心损失的数值稳定性
        for name, loss_val in {"CE": ce_loss, "SDM": sdm_loss}.items():
            if not torch.isfinite(loss_val):
                print(f"⚠️ {name} 损失异常: {loss_val.item()}")
                # 强制重置为0，防止传播
                if name == "SDM":
                    sdm_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
                elif name == "CE":
                    ce_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
                    
        # 简化的总损失：CE + SDM
        total_loss = self.ce_weight * ce_loss + self.contrastive_weight * sdm_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'sdm_loss': sdm_loss,
            'contrastive_loss': sdm_loss,  # 兼容性
            'ce_valid_cnt': valid_ce_cnt  # 添加有效CE样本计数
        }
    
    def get_learnable_params(self) -> List[Dict[str, Any]]:
        """
        获取分层学习率的参数组
        Returns:
            参数组列表，每组包含params和lr
        """
        config = self.config
        
        # 从CLIP编码器获取参数分组
        clip_param_groups = self.clip_encoder.get_learnable_params()
        
        # 构建优化器参数组
        param_groups = [
            {
                'params': clip_param_groups['backbone'],
                'lr': getattr(config, 'base_learning_rate', 1e-5),
                'name': 'clip_backbone'
            },
            {
                'params': clip_param_groups['mer_loras'],
                'lr': getattr(config, 'mer_learning_rate', 5e-5),
                'name': 'mer_loras'
            },
            {
                'params': clip_param_groups['tokenizers'],
                'lr': getattr(config, 'tokenizer_learning_rate', 5e-5),
                'name': 'tokenizers'
            },
            {
                'params': clip_param_groups['projections'],
                'lr': getattr(config, 'fusion_learning_rate', 5e-5),
                'name': 'projections'
            }
        ]
        
        # 添加其他模块的参数 - guide4.py: 分离分类头使用高学习率
        added_params = set()
        for group in param_groups:
            added_params.update(group['params'])
        
        # guide4.py: 专门为分类头设置高学习率（1e-2）
        classifier_params = []
        other_params = []
        for name, param in self.named_parameters():
            if param not in added_params:
                if name.startswith("bn_neck.classifier"):
                    classifier_params.append(param)
                else:
                    other_params.append(param)
        
        # guide6.md: 分类头学习率降档，防止权重爆涨
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': 3e-3,  # guide6.md: 从1e-2降到3e-3，防止权重从7→35爆涨
                'name': 'classification_head'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': getattr(config, 'fusion_learning_rate', 5e-5),
                'name': 'other_modules'
            })
        
        # 过滤空参数组
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        return param_groups
    
    def set_epoch(self, epoch: int):
        """
        设置当前训练epoch，用于控制modality_dropout热身期
        Args:
            epoch: 当前epoch（从1开始）
        """
        self.current_epoch = epoch


if __name__ == "__main__":
    # 简单测试
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        modalities = ['vis', 'nir', 'sk', 'cp', 'text']
        fusion_dim = 512
        vision_hidden_dim = 768
        clip_model_name = 'openai/clip-vit-base-patch16'
        mer_lora_rank = 4
        mer_lora_alpha = 1.0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = TestConfig()
    model = CLIPBasedMultiModalReIDModel(config)
    model.set_num_classes(100)  # 假设100个ID
    
    # 测试前向传播
    images = {
        'vis': torch.randn(2, 3, 224, 224),
        'nir': torch.randn(2, 1, 224, 224)
    }
    texts = ["A person walking", "红外图像中的行人"]
    
    outputs = model(images=images, texts=texts)
    print(f"输出keys: {outputs.keys()}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # 测试损失计算
    labels = torch.randint(0, 100, (2,))
    loss_dict = model.compute_loss(outputs, labels)
    print(f"损失项: {loss_dict.keys()}")
    
    print("✅ CLIP+MER多模态ReID模型测试通过！")
