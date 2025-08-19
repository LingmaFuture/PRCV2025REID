# models/clip_backbone.py
"""
CLIP-B/16统一编码器实现
集成MER模态路由系统，支持视觉和文本的统一编码
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer, CLIPConfig
from typing import List, Optional, Dict, Union, Tuple
import copy

from .mer_lora import MERMultiheadAttention, MERMLP, MERLinear
from .patch_embeds import MultiModalPatchEmbeds


class MERTransformerBlock(nn.Module):
    """带MER路由的Transformer块"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        modalities: List[str],
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        dropout: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(embed_dim, eps=1e-5)
        
        # MER多头自注意力
        self.attn = MERMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            modalities=modalities,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=dropout
        )
        
        # MER MLP
        self.mlp = MERMLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            modalities=modalities,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=dropout
        )
        
        # DropPath (随机深度)
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)
    
    def forward(self, x: torch.Tensor, modality: str, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer块前向传播
        Args:
            x: [B, seq_len, embed_dim] 输入特征
            modality: 当前模态
            attn_mask: 注意力掩码
        Returns:
            [B, seq_len, embed_dim] 输出特征
        """
        # 自注意力 + 残差连接
        attn_out = self.attn(
            query=self.ln1(x),
            key=self.ln1(x),
            value=self.ln1(x),
            modality=modality,
            attn_mask=attn_mask
        )
        x = x + self.drop_path(attn_out)
        
        # MLP + 残差连接
        mlp_out = self.mlp(self.ln2(x), modality)
        x = x + self.drop_path(mlp_out)
        
        return x
    
    def load_clip_block_weights(self, clip_block):
        """从CLIP的Transformer块加载预训练权重"""
        # 根据检查结果，CLIP transformer层没有layer_norm1，只有layer_norm2
        # Vision模型的layer norm结构需要重新检查
        
        # 先检查实际存在的属性
        if hasattr(clip_block, 'layer_norm1'):
            self.ln1.load_state_dict(clip_block.layer_norm1.state_dict())
        elif hasattr(clip_block, 'pre_layrnorm'):
            self.ln1.load_state_dict(clip_block.pre_layrnorm.state_dict())
        
        if hasattr(clip_block, 'layer_norm2'):
            self.ln2.load_state_dict(clip_block.layer_norm2.state_dict())
        elif hasattr(clip_block, 'post_layrnorm'):
            self.ln2.load_state_dict(clip_block.post_layrnorm.state_dict())
        
        # 加载注意力权重到MER注意力的共享主干
        if hasattr(clip_block, 'self_attn'):  # Vision/Text transformer
            attn = clip_block.self_attn
            self.attn.load_pretrained_weights(
                q_weight=attn.q_proj.weight,
                k_weight=attn.k_proj.weight,
                v_weight=attn.v_proj.weight,
                out_weight=attn.out_proj.weight,
                q_bias=getattr(attn.q_proj, 'bias', None),
                k_bias=getattr(attn.k_proj, 'bias', None),
                v_bias=getattr(attn.v_proj, 'bias', None),
                out_bias=getattr(attn.out_proj, 'bias', None)
            )
        
        # 加载MLP权重到MER MLP的共享主干
        self.mlp.load_pretrained_weights(
            fc1_weight=clip_block.mlp.fc1.weight,
            fc2_weight=clip_block.mlp.fc2.weight,
            fc1_bias=getattr(clip_block.mlp.fc1, 'bias', None),
            fc2_bias=getattr(clip_block.mlp.fc2, 'bias', None)
        )


class DropPath(nn.Module):
    """DropPath (随机深度) 正则化"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class CLIPUnifiedEncoder(nn.Module):
    """CLIP-B/16统一编码器，支持多模态MER路由"""
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        modalities: List[str] = ['rgb', 'ir', 'cpencil', 'sketch', 'text'],
        vision_hidden_dim: int = 768,
        text_hidden_dim: int = 512,
        fusion_dim: int = 512,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        drop_path: float = 0.0,
        freeze_text_backbone: bool = False
    ):
        super().__init__()
        
        self.modalities = modalities
        self.vision_modalities = [m for m in modalities if m != 'text']
        self.vision_hidden_dim = vision_hidden_dim
        self.text_hidden_dim = text_hidden_dim  
        self.fusion_dim = fusion_dim
        self.freeze_text_backbone = freeze_text_backbone
        
        # 加载CLIP预训练模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # 文本分词缓存机制（加速优化）
        self.text_cache = {}  # 缓存文本分词结果
        
        # 多模态非共享patch embeddings
        self.patch_embeds = MultiModalPatchEmbeds(
            embed_dim=vision_hidden_dim,
            patch_size=16,
            img_size=224
        )
        
        # 位置编码 (从CLIP vision encoder复制)
        self.vision_pos_embed = nn.Parameter(
            self.clip_model.vision_model.embeddings.position_embedding.weight.clone()
        )
        
        # CLS token (从CLIP vision encoder复制，添加序列维度)
        self.cls_token = nn.Parameter(
            self.clip_model.vision_model.embeddings.class_embedding.clone().unsqueeze(0)  # [1, 768]
        )
        
        # 视觉MER Transformer层
        vision_config = self.clip_model.vision_model.config
        self.vision_layers = nn.ModuleList([
            MERTransformerBlock(
                embed_dim=vision_hidden_dim,
                num_heads=vision_config.num_attention_heads,
                mlp_dim=vision_config.intermediate_size,
                modalities=self.vision_modalities,  # 视觉模态不包括text
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                drop_path=drop_path * (i / max(1, vision_config.num_hidden_layers - 1))
            )
            for i in range(vision_config.num_hidden_layers)
        ])
        
        # 视觉最终layer norm
        self.vision_ln_final = nn.LayerNorm(vision_hidden_dim, eps=1e-5)
        
        # 文本编码器 (复用CLIP，可选择冻结)
        if freeze_text_backbone:
            for param in self.clip_model.text_model.parameters():
                param.requires_grad = False
        
        # 投影层：统一到fusion_dim
        self.vision_proj = nn.Linear(vision_hidden_dim, fusion_dim, bias=False)
        self.text_proj = nn.Linear(text_hidden_dim, fusion_dim, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化新添加的权重"""
        # 从CLIP加载patch embeddings权重
        clip_patch_embed = self.clip_model.vision_model.embeddings.patch_embedding
        self.patch_embeds.load_clip_weights(
            clip_patch_embed.weight,
            clip_patch_embed.bias if hasattr(clip_patch_embed, 'bias') else None
        )
        
        # 从CLIP加载vision transformer权重到MER层
        for i, (mer_layer, clip_layer) in enumerate(zip(
            self.vision_layers, 
            self.clip_model.vision_model.encoder.layers
        )):
            mer_layer.load_clip_block_weights(clip_layer)
        
        # 加载最终layer norm (vision model使用post_layernorm)
        self.vision_ln_final.load_state_dict(
            self.clip_model.vision_model.post_layernorm.state_dict()
        )
        
        # 初始化投影层 (使用CLIP的投影权重)
        with torch.no_grad():
            if hasattr(self.clip_model, 'visual_projection'):
                # CLIP的visual_projection: [512, 768], 我们的vision_proj: [512, 768] - 形状匹配
                self.vision_proj.weight.copy_(self.clip_model.visual_projection.weight)
            if hasattr(self.clip_model, 'text_projection'):
                # CLIP的text_projection: [512, 512], 我们的text_proj: [512, 512] - 形状匹配
                self.text_proj.weight.copy_(self.clip_model.text_projection.weight)
    
    def encode_vision(self, images: torch.Tensor, modality: str) -> torch.Tensor:
        """
        视觉编码
        Args:
            images: [B, C, H, W] 图像tensor
            modality: 视觉模态名 'rgb'|'ir'|'cpencil'|'sketch'
        Returns:
            [B, fusion_dim] 视觉特征
        """
        B = images.shape[0]
        
        # Patch embedding (模态路由)
        patch_embeds = self.patch_embeds(images, modality)  # [B, num_patches, vision_hidden_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, vision_hidden_dim]
        x = torch.cat([cls_tokens, patch_embeds], dim=1)  # [B, 1+num_patches, vision_hidden_dim]
        
        # 添加位置编码
        x = x + self.vision_pos_embed.unsqueeze(0)
        
        # MER Transformer层
        for layer in self.vision_layers:
            x = layer(x, modality)
        
        # 最终layer norm + CLS token特征
        x = self.vision_ln_final(x)
        cls_features = x[:, 0]  # [B, vision_hidden_dim]
        
        # 投影到fusion_dim
        vision_features = self.vision_proj(cls_features)  # [B, fusion_dim]
        
        return vision_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        文本编码（带缓存和批量优化）
        Args:
            texts: 文本列表
        Returns:
            [B, fusion_dim] 文本特征
        """
        device = next(self.parameters()).device
        
        # 检查缓存，分离已缓存和未缓存的文本
        cached_features = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.text_cache and self.freeze_text_backbone:
                # 只有在冻结文本编码器时才使用缓存
                cached_features.append(self.text_cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 批量处理未缓存的文本（加速优化B）
        if uncached_texts:
            # 一次性分词所有未缓存的文本
            text_inputs = self.tokenizer(
                uncached_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP标准长度
            )
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}
            
            # CLIP文本编码
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                text_outputs = self.clip_model.text_model(**text_inputs)
                text_embeds = text_outputs.pooler_output  # [B, text_hidden_dim]
                
                # 投影到fusion_dim
                text_features = self.text_proj(text_embeds)  # [B, fusion_dim]
            
            # 如果文本编码器冻结，缓存结果
            if self.freeze_text_backbone:
                for i, text in enumerate(uncached_texts):
                    self.text_cache[text] = text_features[i].detach().cpu()
        
        # 重组特征（保持原始顺序）
        if len(uncached_texts) == len(texts):
            # 全部文本都未缓存
            return text_features
        elif len(uncached_texts) == 0:
            # 全部文本都已缓存
            return torch.stack([feat.to(device) for feat in cached_features])
        else:
            # 混合情况：重组特征
            final_features = torch.zeros(len(texts), self.fusion_dim, device=device, dtype=text_features.dtype)
            
            # 填入缓存的特征
            cached_idx = 0
            for i in range(len(texts)):
                if i not in uncached_indices:
                    final_features[i] = cached_features[cached_idx].to(device)
                    cached_idx += 1
            
            # 填入新计算的特征
            for idx, uncached_idx in enumerate(uncached_indices):
                final_features[uncached_idx] = text_features[idx]
            
            return final_features
    
    def forward(
        self, 
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        modality: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        统一前向传播
        Args:
            images: [B, C, H, W] 图像 (可选)
            texts: 文本列表 (可选)  
            modality: 当前模态 (图像必须提供)
        Returns:
            包含各模态特征的字典
        """
        outputs = {}
        
        if images is not None:
            assert modality is not None and modality in self.vision_modalities
            outputs['vision_features'] = self.encode_vision(images, modality)
            outputs['modality'] = modality
        
        if texts is not None:
            outputs['text_features'] = self.encode_text(texts)
        
        return outputs
    
    def get_learnable_params(self) -> Dict[str, List[nn.Parameter]]:
        """获取分层学习率的参数组"""
        param_groups = {
            'backbone': [],      # CLIP共享主干
            'mer_loras': [],     # MER LoRA参数
            'tokenizers': [],    # 非共享tokenizer
            'projections': []    # 投影层
        }
        
        # CLIP backbone参数 (共享主干)
        for name, param in self.named_parameters():
            if 'clip_model' in name and param.requires_grad:
                param_groups['backbone'].append(param)
            elif 'lora' in name.lower():
                param_groups['mer_loras'].append(param)
            elif 'patch_embeds' in name:
                param_groups['tokenizers'].append(param)
            elif any(proj in name for proj in ['vision_proj', 'text_proj']):
                param_groups['projections'].append(param)
        
        # 其他新增参数 (vision layers, position embeds, etc.)
        added_params = set()
        for group in param_groups.values():
            added_params.update(group)
        
        for name, param in self.named_parameters():
            if param not in added_params:
                param_groups['tokenizers'].append(param)
        
        return param_groups


if __name__ == "__main__":
    # 简单测试
    encoder = CLIPUnifiedEncoder()
    
    # 测试视觉编码
    rgb_images = torch.randn(2, 3, 224, 224)
    ir_images = torch.randn(2, 1, 224, 224)
    
    rgb_features = encoder.encode_vision(rgb_images, 'rgb')
    ir_features = encoder.encode_vision(ir_images, 'ir')
    
    print(f"RGB features shape: {rgb_features.shape}")  # [2, 512]
    print(f"IR features shape: {ir_features.shape}")    # [2, 512]
    
    # 测试文本编码  
    texts = ["A person walking", "红外图像显示人体轮廓"]
    text_features = encoder.encode_text(texts)
    print(f"Text features shape: {text_features.shape}")  # [2, 512]
    
    # 测试参数分组
    param_groups = encoder.get_learnable_params()
    for group_name, params in param_groups.items():
        print(f"{group_name}: {len(params)} parameters")
    
    print("✅ CLIP统一编码器测试通过！")
