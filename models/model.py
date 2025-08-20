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

from .clip_backbone import CLIPUnifiedEncoder
from .patch_embeds import MultiModalPatchEmbeds
from .mer_lora import MERLinear, MERMultiheadAttention, MERMLP


# 导入原有的SDM和损失函数组件
class SemanticDisentanglementModule(nn.Module):
    """
    语义分离模块：将各模态特征投影到语义空间，便于与RGB对齐
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
    """RGB锚定对齐损失：推动所有查询模态对齐到RGB目标表征"""
    
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
        计算RGB锚定对齐损失
        Args:
            modality_features: 各模态原始特征字典
            fused_features: 融合后特征
            labels: ID标签
        Returns:
            对齐损失
        """
        if 'rgb' not in modality_features:
            return torch.tensor(0.0, device=fused_features.device, requires_grad=True)
        
        rgb_features = modality_features['rgb']  # [B, D] RGB目标特征
        total_loss = torch.tensor(0.0, device=fused_features.device, requires_grad=True)
        num_modalities = 0
        
        # 计算每个非RGB模态与RGB的对齐损失
        for modality, features in modality_features.items():
            if modality == 'rgb':
                continue
                
            # 归一化特征以稳定相似度计算
            features_norm = F.normalize(features, p=2, dim=1)
            rgb_features_norm = F.normalize(rgb_features, p=2, dim=1)
            
            # 对比损失：相同ID拉近，不同ID推远
            sim_matrix = torch.matmul(features_norm, rgb_features_norm.T) / self.temperature
            
            # 构建正负样本掩码
            batch_size = features.shape[0]
            labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
            pos_mask = (labels_expand == labels_expand.T).float()
            neg_mask = 1.0 - pos_mask
            
            # 避免对角线上的自相似
            eye_mask = torch.eye(batch_size, device=features.device)
            pos_mask = pos_mask * (1.0 - eye_mask)  # 移除对角线
            
            # 正样本损失（相同ID应该相似）
            if pos_mask.sum() > 0:
                pos_sim = sim_matrix * pos_mask
                pos_exp = torch.exp(pos_sim)
                pos_exp_sum = pos_exp.sum(dim=1)
                pos_loss = -torch.log(pos_exp_sum + 1e-8).mean()
            else:
                pos_loss = torch.tensor(0.0, device=features.device)
            
            # 负样本损失（不同ID应该不相似）
            if neg_mask.sum() > 0:
                neg_sim = sim_matrix * neg_mask - self.margin
                neg_exp = torch.exp(neg_sim)
                neg_exp_sum = neg_exp.sum(dim=1)
                neg_loss = torch.log(neg_exp_sum + 1e-8).mean()
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
    """SDM对比损失：结合语义分离和RGB锚定对齐"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, alpha: float = 1.0):
        super().__init__()
        self.rgb_alignment = RGBAnchoredAlignmentLoss(temperature, margin, alpha)
        
    def forward(self, 
                modality_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """计算SDM对比损失"""
        alignment_loss = self.rgb_alignment(modality_features, fused_features, labels)
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
        dropped_features = self.dropout(bn_features)
        logits = self.classifier(dropped_features)
        
        return bn_features, logits


class CLIPBasedMultiModalReIDModel(nn.Module):
    """
    基于CLIP-B/16的多模态ReID模型
    架构：CLIP统一编码器 + MER模态路由 + SDM语义分离 + 特征融合 + ID分类
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = getattr(config, 'device', 'cuda')
        
        # 模态配置
        self.modalities = getattr(config, 'modalities', ['rgb', 'ir', 'cpencil', 'sketch', 'text'])
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
        
        # ===== 损失函数 =====
        self.sdm_contrastive_loss = SDMContrastiveLoss(
            temperature=getattr(config, "sdm_temperature", 0.1),
            margin=getattr(config, "sdm_margin", 0.3),
            alpha=1.0
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # 损失权重
        self.ce_weight = getattr(config, 'ce_weight', 1.0)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
        
        # 特征范数正则化参数
        self.feature_target_norm = getattr(config, 'feature_target_norm', 10.0)
        self.feature_norm_band = getattr(config, 'feature_norm_band', 3.0)
        self.feature_norm_penalty = getattr(config, 'feature_norm_penalty', 2e-3)
        
        # ===== 可学习的null token占位符 =====
        # 为每个模态创建可学习的null token，用于缺失模态的占位
        self.null_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, self.fusion_dim) * 0.02)
            for modality in self.modalities
        })
        
        logging.info(f"✅ 初始化CLIP+MER多模态ReID模型完成")
        logging.info(f"   - 支持模态: {self.modalities}")
        logging.info(f"   - 融合维度: {self.fusion_dim}")
        logging.info(f"   - MER LoRA rank: {getattr(config, 'mer_lora_rank', 4)}")
    
    def set_num_classes(self, num_classes: int):
        """动态设置类别数并初始化分类器"""
        self.num_classes = num_classes
        self.bn_neck = BNNeck(
            in_dim=self.fusion_dim,
            num_classes=num_classes,
            dropout=getattr(self.config, 'dropout_rate', 0.5)
        ).to(self.device)
        
        logging.info(f"设置分类器：{num_classes} 个ID类别")
    
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
                        for orig, new in [('vis', 'rgb'), ('nir', 'ir'), ('sk', 'sketch'), ('cp', 'cpencil')]:
                            if new == modality:
                                original_modality = orig
                                break
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
        
        # 模态dropout（训练时随机丢弃部分模态）
        if self.training:
            modality_dropout = getattr(self.config, 'modality_dropout', 0.0)
            min_modalities = getattr(self.config, 'min_modalities', 1)
            
            if modality_dropout > 0 and len(modality_features) > min_modalities:
                keep_modalities = []
                keep_masks = []
                for mod, feat in modality_features.items():
                    if torch.rand(1).item() > modality_dropout:
                        keep_modalities.append((mod, feat))
                        keep_masks.append((mod, feature_masks[mod]))
                
                if len(keep_modalities) >= min_modalities:
                    modality_features = dict(keep_modalities)
                    feature_masks = dict(keep_masks)
        
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
        # ID分类损失
        if 'logits' not in outputs:
            raise ValueError("模型输出中缺少logits，请确保已设置num_classes")
        
        ce_loss = self.ce_loss(outputs['logits'], labels)
        
        # SDM对比损失：RGB锚定对齐损失
        raw_modality_features = outputs.get('raw_modality_features', {})
        # 强制使用BN后特征进行对齐损失计算，确保与检索特征完全一致
        if 'bn_features' in outputs:
            alignment_features = outputs['bn_features']  # 与检索特征保持完全一致
        else:
            raise ValueError("模型输出缺少bn_features，对齐损失特征不一致")
        
        sdm_loss = self.sdm_contrastive_loss(
            raw_modality_features,
            alignment_features,
            labels
        )
        
        # 特征范数正则化 - 使用与检索一致的特征
        if 'bn_features' in outputs:
            feats = outputs['bn_features']  # 与检索和对齐损失保持一致
        else:
            feats = outputs['features']  # 回退到融合后特征
        fn = feats.norm(p=2, dim=1)  # [B,] 每个样本的L2范数
        
        target_norm = self.feature_target_norm
        band = self.feature_norm_band
        lam = self.feature_norm_penalty
        
        over = torch.clamp(fn - (target_norm + band), min=0)
        under = torch.clamp((target_norm - band) - fn, min=0)
        feat_penalty = (over**2 + under**2).mean() * lam
        
        # 总损失
        total_loss = (self.ce_weight * ce_loss + 
                     self.contrastive_weight * sdm_loss + 
                     feat_penalty)
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'sdm_loss': sdm_loss,
            'contrastive_loss': sdm_loss,  # 兼容性
            'feat_penalty': feat_penalty,
            'feature_norm': fn.mean().item()
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
        
        # 添加其他模块的参数
        added_params = set()
        for group in param_groups:
            added_params.update(group['params'])
        
        other_params = []
        for name, param in self.named_parameters():
            if param not in added_params:
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': getattr(config, 'fusion_learning_rate', 5e-5),
                'name': 'other_modules'
            })
        
        # 过滤空参数组
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        return param_groups


if __name__ == "__main__":
    # 简单测试
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        modalities = ['rgb', 'ir', 'cpencil', 'sketch', 'text']
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
        'rgb': torch.randn(2, 3, 224, 224),
        'ir': torch.randn(2, 1, 224, 224)
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
