# models/model.py — 优化可运行版 + LoRA专家
import math
from typing import Dict, List, Optional, Tuple
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoTokenizer

# -------------------------
# LoRA 低秩适配器模块
# -------------------------
class LoRALayer(nn.Module):
    """
    基础LoRA层：在原有线性层基础上添加低秩矩阵A*B
    原理：W' = W + α * A * B, 其中A: (input_dim, r), B: (r, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA矩阵：A用随机初始化，B用零初始化（确保开始时LoRA贡献为0）
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))
        
        # 缩放因子
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_dim) -> (..., output_dim)
        """
        # LoRA前向传播：x @ A @ B
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
        return lora_out * self.scaling


class ModalityLoRAExperts(nn.Module):
    """
    模态专属LoRA专家系统：为每个模态维护独立的LoRA参数
    支持模态路由，根据输入模态激活对应的LoRA专家
    """
    def __init__(self, input_dim: int, output_dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # 为每个模态创建独立的LoRA专家
        self.modality_experts = nn.ModuleDict({
            'vis': LoRALayer(input_dim, output_dim, rank, alpha),
            'nir': LoRALayer(input_dim, output_dim, rank, alpha),
            'sk': LoRALayer(input_dim, output_dim, rank, alpha),
            'cp': LoRALayer(input_dim, output_dim, rank, alpha),
        })
        
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        根据模态类型选择对应的LoRA专家进行前向传播
        Args:
            x: (..., input_dim) 输入特征
            modality: 模态类型 ('vis', 'nir', 'sk', 'cp')
        Returns:
            lora_output: (..., output_dim) LoRA输出
        """
        if modality in self.modality_experts:
            return self.modality_experts[modality](x)
        else:
            # 如果模态不存在，返回零输出（保持原层行为）
            return torch.zeros(x.shape[:-1] + (self.output_dim,), device=x.device, dtype=x.dtype)


class LoRALinear(nn.Module):
    """
    LoRA增强的线性层：包装原有线性层，添加模态专属的LoRA适配
    输出 = original_linear(x) + lora_experts(x, modality)
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        
        # 冻结原始权重（可选）
        # for param in self.original_linear.parameters():
        #     param.requires_grad = False
            
        # 创建模态LoRA专家
        self.lora_experts = ModalityLoRAExperts(
            original_linear.in_features, 
            original_linear.out_features,
            rank=rank, 
            alpha=alpha
        )
        
    def forward(self, x: torch.Tensor, modality: str = 'vis') -> torch.Tensor:
        """
        Args:
            x: 输入特征
            modality: 模态类型，用于选择LoRA专家
        """
        # 原始线性层输出
        original_out = self.original_linear(x)
        
        # LoRA专家输出
        lora_out = self.lora_experts(x, modality)
        
        # 组合输出：原始 + LoRA适配
        return original_out + lora_out


def inject_lora_to_linear_layers(model: nn.Module, rank: int = 4, alpha: float = 1.0, target_modules: Optional[List[str]] = None):
    """
    递归地将模型中的线性层替换为LoRA增强版本
    Args:
        model: 要修改的模型
        rank: LoRA秩
        alpha: LoRA缩放因子  
        target_modules: 目标模块名称列表，如果为None则替换所有线性层
    """
    # ViT中需要替换的关键线性层
    if target_modules is None:
        target_modules = ['qkv', 'proj', 'fc1', 'fc2']  # ViT中的注意力和FFN层
    
    def _replace_linear(module: nn.Module, name: str):
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear):
                # 检查是否是目标模块
                should_replace = any(target in child_name for target in target_modules)
                if should_replace:
                    # 替换为LoRA增强版本
                    lora_linear = LoRALinear(child_module, rank=rank, alpha=alpha)
                    setattr(module, child_name, lora_linear)
                    print(f"替换线性层: {name}.{child_name} -> LoRALinear(rank={rank})")
            else:
                # 递归处理子模块
                _replace_linear(child_module, f"{name}.{child_name}")
    
    _replace_linear(model, "model")


class LoRAViTForward:
    """
    LoRA增强的ViT前向传播工具类
    支持模态路由的token级别前向传播
    """
    
    @staticmethod
    def forward_with_modality(vit_model, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        带模态信息的ViT前向传播
        Args:
            vit_model: ViT模型（已注入LoRA）
            x: 输入图像张量 (B, C, H, W)
            modality: 模态类型
        Returns:
            tokens: (B, 1+N, C) 包含CLS token的完整token序列
        """
        # Patch embedding
        x = vit_model.patch_embed(x)  # (B, N, C)
        
        # 添加CLS token
        cls_tokens = vit_model.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, C)
        
        # 位置编码
        if hasattr(vit_model, 'pos_embed') and vit_model.pos_embed is not None:
            pos_embed = vit_model.pos_embed[:, :x.size(1), :]
            x = x + pos_embed
        
        x = vit_model.pos_drop(x)
        
        # 通过Transformer blocks，每个block都会使用LoRA
        for block in vit_model.blocks:
            x = LoRAViTForward._forward_block_with_modality(block, x, modality)
        
        # 最终层归一化
        x = vit_model.norm(x)
        
        return x
    
    @staticmethod 
    def _forward_block_with_modality(block, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        带模态信息的Transformer block前向传播
        """
        # 获取drop_path函数（兼容不同timm版本）
        drop_path_fn = None
        if hasattr(block, 'drop_path'):
            drop_path_fn = block.drop_path
        elif hasattr(block, 'drop_path1'):
            drop_path_fn = block.drop_path1
        else:
            # 如果没有drop_path，使用恒等函数
            drop_path_fn = lambda x: x
        
        # Self-attention
        if hasattr(block, 'attn'):
            attn_out = LoRAViTForward._forward_attention_with_modality(block.attn, block.norm1(x), modality)
            x = x + drop_path_fn(attn_out)
        
        # FFN
        if hasattr(block, 'mlp'):
            ffn_out = LoRAViTForward._forward_mlp_with_modality(block.mlp, block.norm2(x), modality)  
            x = x + drop_path_fn(ffn_out)
        
        return x
    
    @staticmethod
    def _forward_attention_with_modality(attn, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        带模态信息的注意力层前向传播
        """
        B, N, C = x.shape
        
        # QKV投影（如果已经是LoRALinear，会自动使用模态信息）
        if isinstance(attn.qkv, LoRALinear):
            qkv = attn.qkv(x, modality)
        else:
            qkv = attn.qkv(x)
        
        qkv = qkv.reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力计算
        attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = attn.attn_drop(attn_weights)
        
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        if isinstance(attn.proj, LoRALinear):
            x = attn.proj(x, modality)
        else:
            x = attn.proj(x)
        
        x = attn.proj_drop(x)
        return x
    
    @staticmethod
    def _forward_mlp_with_modality(mlp, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        带模态信息的MLP层前向传播
        """
        # 获取dropout函数（兼容不同timm版本）
        def get_dropout_fn():
            if hasattr(mlp, 'drop'):
                return mlp.drop
            elif hasattr(mlp, 'dropout'):
                return mlp.dropout
            elif hasattr(mlp, 'drop1'):
                return mlp.drop1
            else:
                # 如果没有dropout，使用恒等函数
                return lambda x: x
        
        dropout_fn = get_dropout_fn()
        
        # FC1
        if isinstance(mlp.fc1, LoRALinear):
            x = mlp.fc1(x, modality)
        else:
            x = mlp.fc1(x)
        
        x = mlp.act(x)
        x = dropout_fn(x)
        
        # FC2  
        if isinstance(mlp.fc2, LoRALinear):
            x = mlp.fc2(x, modality)
        else:
            x = mlp.fc2(x)
        
        # 第二个dropout（有些版本有drop2）
        if hasattr(mlp, 'drop2'):
            x = mlp.drop2(x)
        else:
            x = dropout_fn(x)
            
        return x


# -------------------------
# 基础组件
# -------------------------
class LayerScale(nn.Module):
    """LayerScale for better training stability"""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x

class VectorModalityAdapter(nn.Module):
    """
    向量级 FiLM 适配器：输入 (B, C)，输出 (B, C)
    学到逐通道的 scale/bias：y = x * (1 + gamma) + beta
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim * 2)  # -> [gamma, beta]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gb = self.net(x)                 # (B, 2C)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return x * (1.0 + gamma) + beta

class ModalityAdapter(nn.Module):
    """
    模态适配器（FiLM式）：不改变序列长度，避免 ViT pos_embed 尺寸问题。
    给定 token 序列，基于 patch 的全局统计，学习到逐通道的 scale/bias。
    """
    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim * 2),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, 1+N, C) ; 只基于 patch token 做统计
        patch = tokens[:, 1:, :] if tokens.size(1) > 1 else tokens  # 防卫
        pooled = patch.mean(dim=1)  # (B, C)
        gamma_beta = self.mlp(self.ln(pooled))  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # (B, C), (B, C)
        return tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

class CrossModalTransformerBlock(nn.Module):
    """跨模态 Transformer 块（Post-LN 风格，小数据集更优）"""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=attn_drop)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.cross_scale = LayerScale(embed_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=attn_drop)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_scale = LayerScale(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(proj_drop),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn_scale = LayerScale(embed_dim)

    def forward(self, query, key_value, key_padding_mask: Optional[torch.Tensor] = None):
        # Cross-Attn: Attention → Add → LayerNorm
        cross_out, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        x = self.cross_norm(query + self.cross_scale(cross_out))

        # Self-Attn: Attention → Add → LayerNorm  
        self_out, _ = self.self_attn(x, x, x)
        x = self.self_norm(x + self.self_scale(self_out))

        # FFN: MLP → Add → LayerNorm
        x = self.ffn_norm(x + self.ffn_scale(self.mlp(x)))
        return x

class LightweightFeatureMixer(nn.Module):
    """
    轻量特征混合器：多头自注意力 + 1层Transformer block + MLP平均池化
    更高效的多模态融合方案，替代复杂的层次化融合
    """
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 模态嵌入：为每个模态学习可区分的嵌入向量
        self.modality_embeddings = nn.ParameterDict({
            'vis': nn.Parameter(torch.randn(1, embed_dim) * 0.02),
            'nir': nn.Parameter(torch.randn(1, embed_dim) * 0.02), 
            'sk': nn.Parameter(torch.randn(1, embed_dim) * 0.02),
            'cp': nn.Parameter(torch.randn(1, embed_dim) * 0.02),
            'text': nn.Parameter(torch.randn(1, embed_dim) * 0.02),
        })
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # 多头自注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 1层Transformer Block
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-LN for stability
        )
        
        # MLP投影器用于最终融合
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, modality_features: Dict[str, torch.Tensor], 
                modality_mask: Optional[Dict[str, bool]] = None):
        """
        轻量特征混合器前向传播
        
        Args:
            modality_features: 模态特征字典 {modality: (B, C)}
            modality_mask: 模态掩码字典 {modality: bool}，可选
            
        Returns:
            fused_features: (B, embed_dim) 融合后的特征
        """
        if not modality_features:
            # 如果没有输入特征，返回零向量
            return torch.zeros(1, self.embed_dim, device=next(iter(self.parameters())).device)
            
        # 获取批次信息
        ref_feature = next(iter(modality_features.values()))
        batch_size = ref_feature.size(0)
        device = ref_feature.device
        
        # 构建模态序列：每个可用模态贡献一个token
        modality_tokens = []
        valid_modalities = []
        
        for modality, features in modality_features.items():
            # 检查模态是否可用
            if modality_mask is not None and not modality_mask.get(modality, True):
                continue
                
            if modality in self.modality_embeddings:
                # 特征 + 模态嵌入
                modal_embed = self.modality_embeddings[modality].expand(batch_size, -1)  # (B, C)
                enhanced_features = features + modal_embed  # (B, C)
                
                # 添加到序列
                modality_tokens.append(enhanced_features.unsqueeze(1))  # (B, 1, C)
                valid_modalities.append(modality)
        
        if not modality_tokens:
            # 没有有效模态，返回零向量
            return torch.zeros(batch_size, self.embed_dim, device=device)
        
        # 拼接所有模态tokens：(B, N_modalities, C)
        modal_sequence = torch.cat(modality_tokens, dim=1)  # (B, M, C)
        
        # 输入层归一化
        modal_sequence = self.input_norm(modal_sequence)
        
        # 多头自注意力：模态间交互
        attn_output, _ = self.multihead_attn(
            modal_sequence, modal_sequence, modal_sequence
        )  # (B, M, C)
        
        # 残差连接
        modal_sequence = modal_sequence + attn_output
        
        # 1层Transformer Block进一步处理
        enhanced_sequence = self.transformer_block(modal_sequence)  # (B, M, C)
        
        # MLP + 平均池化：融合所有模态信息
        # 先通过MLP增强每个模态特征
        enhanced_features = self.fusion_mlp(enhanced_sequence)  # (B, M, C)
        
        # 平均池化获得最终融合特征
        fused_features = enhanced_features.mean(dim=1)  # (B, C)
        
        return fused_features


class HierarchicalMultiModalFusion(nn.Module):
    """层次化多模态融合（晚期+全局token）——每模态输入为一个向量"""
    def __init__(self, embed_dim, num_layers=3, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim

        self.modality_tokens = nn.ParameterDict({
            'vis': nn.Parameter(torch.randn(1, embed_dim)),
            'nir': nn.Parameter(torch.randn(1, embed_dim)),
            'sk': nn.Parameter(torch.randn(1, embed_dim)),
            'cp': nn.Parameter(torch.randn(1, embed_dim)),
            'text': nn.Parameter(torch.randn(1, embed_dim)),
        })

        self.fusion_layers = nn.ModuleList([
            CrossModalTransformerBlock(embed_dim, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.global_token = nn.Parameter(torch.randn(1, embed_dim))
        self.final_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, modality_features: Dict[str, torch.Tensor], modality_mask: Optional[Dict[str, bool]] = None):
        # modality_features[k]: (B, C)
        ref = next(iter(modality_features.values()))
        device = ref.device
        batch = ref.size(0)

        sequences = []
        for m, feat in modality_features.items():
            if modality_mask is not None and not modality_mask.get(m, True):
                continue
            token = self.modality_tokens[m].unsqueeze(0).expand(batch, -1, -1)  # (B, 1, C)
            feat = feat.unsqueeze(1)  # (B, 1, C)
            sequences.append(torch.cat([token, feat], dim=1))  # (B, 2, C)

        if not sequences:
            return torch.zeros(batch, self.embed_dim, device=device)

        all_seq = torch.cat(sequences, dim=1)  # (B, 2*M, C)
        global_q = self.global_token.unsqueeze(0).expand(batch, 1, -1)  # (B, 1, C)

        # 这里不需要 padding mask（全是有效 token）
        for layer in self.fusion_layers:
            global_q = layer(global_q, all_seq, key_padding_mask=None)

        fused = self.final_projection(global_q.squeeze(1))  # (B, C)
        return fused

# -------------------------
# SDM模块和RGB锚定损失
# -------------------------
class SemanticDisentanglementModule(nn.Module):
    """
    语义分离模块：将各模态特征投影到语义空间，便于与RGB对齐
    """
    def __init__(self, input_dim: int, semantic_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.semantic_dim = semantic_dim
        
        # 语义投影器：将特征映射到语义空间
        self.semantic_projector = nn.Sequential(
            nn.Linear(input_dim, semantic_dim * 2),
            nn.BatchNorm1d(semantic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(semantic_dim * 2, semantic_dim),
            nn.BatchNorm1d(semantic_dim),
        )
        
        # 语义注意力机制：增强关键语义特征
        self.semantic_attention = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 语义正则化器：保持语义一致性
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, input_dim) 输入特征
        Returns:
            semantic_features: (B, semantic_dim) 语义特征
        """
        # 语义投影
        semantic = self.semantic_projector(features)  # (B, semantic_dim)
        
        # 添加序列维度进行注意力计算
        semantic_seq = semantic.unsqueeze(1)  # (B, 1, semantic_dim)
        
        # 语义自注意力增强
        enhanced_semantic, _ = self.semantic_attention(
            semantic_seq, semantic_seq, semantic_seq
        )
        enhanced_semantic = enhanced_semantic.squeeze(1)  # (B, semantic_dim)
        
        # 残差连接和层归一化
        semantic_out = self.semantic_norm(semantic + enhanced_semantic)
        
        return semantic_out

class RGBAnchoredAlignmentLoss(nn.Module):
    """
    RGB锚定对齐损失：将所有模态或其组合对齐到RGB
    实现"所有模态或其组合 → 对齐到RGB"的策略
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, alpha: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha  # ID分类损失权重
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """特征归一化，避免梯度不稳定"""
        return F.normalize(x, p=2, dim=-1, eps=1e-8)
    
    def _compute_alignment_loss(self, rgb_features: torch.Tensor, 
                              other_features: torch.Tensor, 
                              labels: torch.Tensor) -> torch.Tensor:
        """
        计算RGB与其他模态的对齐损失
        Args:
            rgb_features: (B, D) RGB特征作为锚点
            other_features: (B, D) 其他模态特征
            labels: (B,) 身份标签
        """
        rgb_norm = self._normalize(rgb_features)
        other_norm = self._normalize(other_features)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(other_norm, rgb_norm.t()) / self.temperature  # (B, B)
        
        # 构建正样本mask（相同身份ID）
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float()  # (B, B)
        
        # 排除自身
        eye_mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        pos_mask = pos_mask * (1 - eye_mask)
        
        # 计算对齐损失：拉近正样本，推远负样本
        exp_sim = torch.exp(sim_matrix)
        
        # 分子：正样本相似度之和
        pos_sim = torch.sum(exp_sim * pos_mask, dim=1)
        
        # 分母：所有样本相似度之和（除自身）
        all_sim = torch.sum(exp_sim * (1 - eye_mask), dim=1)
        
        # 避免数值不稳定
        pos_sim = torch.clamp(pos_sim, min=1e-8)
        all_sim = torch.clamp(all_sim, min=1e-8)
        
        # 计算负对数似然
        loss = -torch.log(pos_sim / all_sim)
        
        # 只计算有正样本的情况
        valid_mask = torch.sum(pos_mask, dim=1) > 0
        if valid_mask.any():
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=rgb_features.device)
            
        return loss
    
    def forward(self, modality_features: Dict[str, torch.Tensor], 
                fused_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算RGB锚定的对齐损失
        Args:
            modality_features: 各模态特征字典
            fused_features: 融合特征
            labels: 身份标签
        """
        total_loss = torch.tensor(0.0, device=labels.device)
        loss_count = 0
        
        if 'vis' not in modality_features:
            # 如果没有RGB特征，返回0损失
            return total_loss
            
        rgb_features = modality_features['vis']
        
        # 1. 各单模态与RGB对齐
        for modality, features in modality_features.items():
            if modality != 'vis' and features.size(0) > 1:  # 跳过RGB本身，确保batch size > 1
                align_loss = self._compute_alignment_loss(rgb_features, features, labels)
                total_loss += align_loss
                loss_count += 1
        
        # 2. 融合特征与RGB对齐（这是核心：多模态组合 → RGB）
        if fused_features.size(0) > 1:
            fused_align_loss = self._compute_alignment_loss(rgb_features, fused_features, labels)
            total_loss += fused_align_loss * 2.0  # 融合特征对齐更重要，权重加倍
            loss_count += 1
        
        # 平均损失
        if loss_count > 0:
            total_loss = total_loss / loss_count
        
        return total_loss

class SDMContrastiveLoss(nn.Module):
    """
    SDM对比损失：结合语义分离和RGB锚定对齐
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, alpha: float = 1.0):
        super().__init__()
        self.rgb_alignment = RGBAnchoredAlignmentLoss(temperature, margin, alpha)
        
    def forward(self, modality_features: Dict[str, torch.Tensor],
                fused_features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算SDM对比损失
        """
        # RGB锚定对齐损失
        alignment_loss = self.rgb_alignment(modality_features, fused_features, labels)
        
        return alignment_loss

# -------------------------
# 模型主体
# -------------------------
class MultiModalReIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 统一的融合维度
        self.fusion_dim = getattr(config, "fusion_dim", 768)

        # ===== 视觉骨干：支持 ViT 或 ResNet =====
        self.backbone_name = getattr(config, "backbone", "resnet50")
        use_pretrained = getattr(config, "use_pretrained_vision", False)

        if self.backbone_name.startswith("vit"):
            # ViT 分支
            self.backbone_type = "vit"
            self.vision_backbone = timm.create_model(
                self.backbone_name,
                pretrained=use_pretrained,
                num_classes=0,
                drop_path_rate=getattr(config, "drop_path", 0.15)  # 添加DropPath抑制过拟合
            )
            self.vision_out_dim = self.vision_backbone.num_features  # 通常 768

            # ===== 注入LoRA到ViT线性层 =====
            self.enable_lora = getattr(config, "enable_lora", True)  # 默认启用LoRA
            if self.enable_lora:
                lora_rank = getattr(config, "lora_rank", 4)
                lora_alpha = getattr(config, "lora_alpha", 1.0)
                
                print(f"注入LoRA到ViT: rank={lora_rank}, alpha={lora_alpha}")
                inject_lora_to_linear_layers(
                    self.vision_backbone, 
                    rank=lora_rank, 
                    alpha=lora_alpha
                )
                print("LoRA注入完成")
                
                # 使用LoRA时不再需要额外的模态适配器
                self.modality_adapters = None
            else:
                # 传统Token级模态适配器（向后兼容）
                self.modality_adapters = nn.ModuleDict({
                    'vis': ModalityAdapter(self.vision_out_dim),
                    'nir': ModalityAdapter(self.vision_out_dim),
                    'sk':  ModalityAdapter(self.vision_out_dim),
                    'cp':  ModalityAdapter(self.vision_out_dim),
                })

        else:
            # ResNet / CNN 分支
            self.backbone_type = "cnn"
            self.vision_backbone = timm.create_model(
                self.backbone_name,
                pretrained=use_pretrained,
                num_classes=0,
                global_pool='avg'
            )
            self.vision_out_dim = self.vision_backbone.num_features  # resnet50 通常 2048

            # 向量级模态适配器
            self.modality_vec_adapters = nn.ModuleDict({
                'vis': VectorModalityAdapter(self.vision_out_dim),
                'nir': VectorModalityAdapter(self.vision_out_dim),
                'sk':  VectorModalityAdapter(self.vision_out_dim),
                'cp':  VectorModalityAdapter(self.vision_out_dim),
            })

        # 统一视觉输出到 fusion_dim
        self.visual_projection = (
            nn.Identity() if self.vision_out_dim == self.fusion_dim
            else nn.Sequential(
                nn.Linear(self.vision_out_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        )

        # ===== 文本编码器 =====
        txt_name = getattr(config, "text_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.text_tokenizer = AutoTokenizer.from_pretrained(txt_name)
        self.text_encoder  = AutoModel.from_pretrained(txt_name)
        self.freeze_text = getattr(config, "freeze_text", True)
        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.text_in_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_in_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ===== SDM语义分离模块 =====
        self.sdm_module = SemanticDisentanglementModule(
            input_dim=self.fusion_dim,
            semantic_dim=getattr(config, "sdm_semantic_dim", 512),
            num_heads=getattr(config, "sdm_num_heads", 8)
        )

        # ===== 轻量特征混合器融合 =====
        # 使用轻量特征混合器替代复杂的层次化融合
        fusion_num_heads = getattr(config, "fusion_num_heads", 8)
        fusion_mlp_ratio = getattr(config, "fusion_mlp_ratio", 2.0) 
        fusion_dropout = getattr(config, "fusion_dropout", 0.1)
        
        self.fusion_module = LightweightFeatureMixer(
            embed_dim=self.fusion_dim,
            num_heads=fusion_num_heads,
            mlp_ratio=fusion_mlp_ratio,
            dropout=fusion_dropout
        )

        self.bnneck = nn.BatchNorm1d(self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, self.config.num_classes, bias=False)

        self.feature_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
        )

        # ===== SDM对比损失（替换原有对比损失）=====
        self.sdm_contrastive_loss = SDMContrastiveLoss(
            temperature=getattr(config, "sdm_temperature", 0.1),
            margin=getattr(config, "sdm_margin", 0.3),
            alpha=1.0  # ID分类损失权重
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # 文本特征缓存（冻结时启用，微调时禁用确保一致性）
        self.text_cache = {}  # {text_str: encoded_feature_cpu}
        self.cache_enabled = self.freeze_text  # 冻结时启用缓存，微调时禁用

    @property
    def device(self):
        return next(self.parameters()).device

    def _vit_tokens(self, x: torch.Tensor, modality: str = 'vis') -> torch.Tensor:
        """返回 ViT 的完整 token 序列（含 CLS），形状 (B, 1+N, C)"""
        assert self.backbone_type == "vit", "Only ViT backbone uses _vit_tokens()"
        
        # 如果启用LoRA，使用模态感知的前向传播
        if self.enable_lora:
            return LoRAViTForward.forward_with_modality(self.vision_backbone, x, modality)
        else:
            # 传统前向传播
            m = self.vision_backbone
            x = m.patch_embed(x)  # (B, N, C)
            cls = m.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)
            x = torch.cat((cls, x), dim=1)  # (B, 1+N, C)

            pos_embed = m.pos_embed[:, :x.size(1), :]
            x = x + pos_embed
            x = m.pos_drop(x)

            for blk in m.blocks:
                x = blk(x)
            x = m.norm(x)
            return x

    def encode_image(self, image: torch.Tensor, modality_type: str) -> torch.Tensor:
        """返回 (B, fusion_dim)"""
        if self.backbone_type == "vit":
            # 使用模态信息进行ViT前向传播
            tokens = self._vit_tokens(image, modality_type)
            
            # 如果启用LoRA，模态专属性已通过LoRA实现，无需额外适配器
            if not self.enable_lora and self.modality_adapters is not None:
                if modality_type in self.modality_adapters:
                    tokens = self.modality_adapters[modality_type](tokens)
            
            # 提取特征：使用patch tokens的均值（排除CLS token）
            feats = tokens[:, 1:, :].mean(dim=1) if tokens.size(1) > 1 else tokens.squeeze(1)
        else:
            # ResNet/CNN分支保持不变
            feats = self.vision_backbone(image)  # (B, vision_out_dim)
            if modality_type in getattr(self, 'modality_vec_adapters', {}):
                feats = self.modality_vec_adapters[modality_type](feats)
        
        feats = self.visual_projection(feats)
        return feats

    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        batch_size = len(text_descriptions)
        if batch_size == 0:
            return torch.zeros(0, self.fusion_dim, device=self.device)

        valid_texts = [t if isinstance(t, str) and len(t) > 0 else "[UNK]" for t in text_descriptions]
        
        if self.cache_enabled:
            cached_features, uncached_texts, uncached_indices = [], [], []
            for i, text in enumerate(valid_texts):
                if text in self.text_cache:
                    cached_features.append(self.text_cache[text])
                else:
                    uncached_texts.append(text); uncached_indices.append(i)
            if len(uncached_texts) == 0:
                return torch.stack(cached_features).to(self.device)

            enc = self.text_tokenizer(uncached_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            ctx = torch.no_grad() if self.freeze_text else contextlib.nullcontext()
            with ctx:
                out = self.text_encoder(**enc, return_dict=True)
                token = out.last_hidden_state  # (B, L, H)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            text_vec = (token * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            text_vec = self.text_projection(text_vec)

            for i, text in enumerate(uncached_texts):
                self.text_cache[text] = text_vec[i].detach().cpu()
            
            result_features = [None] * batch_size
            cached_idx = 0
            for i in range(batch_size):
                if i not in uncached_indices:
                    result_features[i] = cached_features[cached_idx].to(self.device); cached_idx += 1
            for i, orig_idx in enumerate(uncached_indices):
                result_features[orig_idx] = text_vec[i]
            return torch.stack(result_features)
        else:
            enc = self.text_tokenizer(valid_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            ctx = torch.no_grad() if self.freeze_text else contextlib.nullcontext()
            with ctx:
                out = self.text_encoder(**enc, return_dict=True)
                token = out.last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            text_vec = (token * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            text_vec = self.text_projection(text_vec)
            return text_vec

    def forward(self, batch: Dict, return_features: bool = False):
        images: Dict[str, torch.Tensor] = batch['images']
        text_descriptions: List[str] = batch.get('text_description', [])
        raw_mm = batch.get('modality_mask', {})

        B = next(iter(images.values())).size(0)
        modality_features, modality_masks = {}, {}

        def _mask_vec(name):
            v = raw_mm.get(name, None)
            if isinstance(v, torch.Tensor):
                return v.to(self.device).float().view(-1, 1)
            elif isinstance(v, (list, tuple)):
                return torch.tensor(v, device=self.device, dtype=torch.float32).view(-1,1)
            return torch.ones(B, 1, device=self.device)

        # 图像模态（先编码，再通过SDM语义分离）
        raw_modality_features = {}  # 保存原始特征用于损失计算
        for m in ['vis','nir','sk','cp']:
            if m in images:
                feats = self.encode_image(images[m].to(self.device), m)
                mvec  = _mask_vec(m)
                feats = feats * mvec
                raw_modality_features[m] = feats  # 保存原始特征
                
                # 通过SDM进行语义分离（仅在训练时进行，推理时保持原特征）
                if self.training:
                    semantic_feats = self.sdm_module(feats)
                    modality_features[m] = semantic_feats
                else:
                    modality_features[m] = feats
                    
                modality_masks[m] = mvec.squeeze(1)

        # 文本模态
        add_text = any((isinstance(t, str) and t.strip()) for t in text_descriptions)
        if add_text and 'text' in raw_mm:
            tfeat = self.encode_text(text_descriptions)
            tvec  = _mask_vec('text')
            tfeat = tfeat * tvec
            raw_modality_features['text'] = tfeat  # 保存原始特征
            
            # 通过SDM进行语义分离（仅在训练时进行）
            if self.training:
                semantic_tfeat = self.sdm_module(tfeat)
                modality_features['text'] = semantic_tfeat
            else:
                modality_features['text'] = tfeat
                
            modality_masks['text'] = tvec.squeeze(1)

        # 融合
        mask_for_fusion = {m: bool(modality_masks.get(m, torch.zeros(B, device=self.device)).any().item())
                        for m in modality_features.keys()}

        fused_features = self.fusion_module(modality_features, mask_for_fusion)  # [B,C]
        reid = self.feature_projection(fused_features)
        # 防御：极小范数样本加微噪，避免梯度不稳定
        if self.training:
            eps_mask = (fused_features.norm(p=2, dim=1, keepdim=True) < 1e-6)
            if eps_mask.any():
                fused_features = fused_features + eps_mask.float() * torch.randn_like(fused_features) * 1e-5

        reid_norm = F.normalize(reid, p=2, dim=1)

        if return_features:
            return reid_norm

        neck = self.bnneck(fused_features)
        logits = self.classifier(neck)

        return {
            'logits': logits,
            'features': fused_features,
            'reid_features': reid_norm,
            'modality_features': modality_features,  # 语义分离后的特征
            'raw_modality_features': raw_modality_features,  # 原始模态特征用于RGB锚定
            'modality_masks': modality_masks,
            'reid_features_raw': reid,
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch: Dict=None):
        # ID分类损失（权重α=1.0）
        ce_loss = self.ce_loss(outputs['logits'], labels)
        
        # SDM对比损失：RGB锚定对齐损失
        raw_modality_features = outputs.get('raw_modality_features', {})
        fused_features = outputs['features']
        
        sdm_loss = self.sdm_contrastive_loss(
            raw_modality_features,  # 使用原始模态特征进行RGB锚定
            fused_features,
            labels
        )
        
        # 获取SDM损失权重（用原来的contrastive_weight参数）
        sdm_weight = getattr(self.config, 'contrastive_weight', 0.1)

        # --- 保留特征范数正则化 ---
        feats = outputs['features']  # [B, C] 融合后、进BNNeck前的向量
        fn = feats.norm(p=2, dim=1)
        target = getattr(self.config, 'feature_target_norm', 10.0)
        band = getattr(self.config, 'feature_norm_band', 4.0)
        lam = getattr(self.config, 'feature_norm_penalty', 1e-3)

        # 只惩罚远离[target-band, target+band] 的样本
        over = (fn - (target + band)).clamp_min(0)
        under = ((target - band) - fn).clamp_min(0)
        feat_penalty = (over**2 + under**2).mean() * lam

        # 总损失：ID分类损失（α=1.0）+ SDM对齐损失 + 特征范数正则
        total_loss = ce_loss + sdm_weight * sdm_loss + feat_penalty
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'sdm_loss': sdm_loss,  # 改名以区分新损失
            'contrastive_loss': sdm_loss,  # 保持兼容性，用于现有的监控代码
            'feat_penalty': feat_penalty
        }