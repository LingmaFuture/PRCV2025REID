# models/mer_lora.py
"""
MER (Modality-Expert Router) 模态路由LoRA适配器实现
为每个模态提供独立的LoRA分支，在共享主干基础上添加模态特异性
"""
import torch
import torch.nn as nn
import math
from typing import List, Optional


class LoRAAdapter(nn.Module):
    """单个LoRA适配器：实现低秩分解 W + ∆W = W + B*A"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int, 
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA的两个低秩矩阵
        self.lora_A = nn.Linear(in_dim, rank, bias=False)   # 下投影
        self.lora_B = nn.Linear(rank, out_dim, bias=False)  # 上投影
        
        # Dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化：A用随机初始化，B用零初始化（确保初始时∆W=0）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim] 输入特征
        Returns:
            [..., out_dim] LoRA适配输出
        """
        # LoRA前向：x -> A -> dropout -> B -> scale
        lora_out = self.lora_B(self.dropout(self.lora_A(x)))
        return lora_out * self.scaling


class MERLinear(nn.Module):
    """MER路由线性层：共享主干 + 每模态独立LoRA"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        modalities: List[str],
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = modalities
        
        # 共享的主干线性层（承载CLIP预训练权重）
        self.shared_linear = nn.Linear(in_dim, out_dim, bias=bias)
        
        # 每个模态的LoRA适配器
        self.loras = nn.ModuleDict({
            modality: LoRAAdapter(in_dim, out_dim, lora_rank, lora_alpha, lora_dropout)
            for modality in modalities
        })
    
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        MER前向：共享主干输出 + 当前模态的LoRA适配
        Args:
            x: [..., in_dim] 输入特征
            modality: 当前模态名称
        Returns:
            [..., out_dim] MER输出特征
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}. Supported: {self.modalities}")
        
        # 主干输出
        shared_out = self.shared_linear(x)
        
        # 当前模态的LoRA适配
        lora_out = self.loras[modality](x)
        
        # 组合输出：主干 + LoRA
        return shared_out + lora_out
    
    def load_pretrained_weights(self, pretrained_weight: torch.Tensor, pretrained_bias: Optional[torch.Tensor] = None):
        """从预训练权重（如CLIP）加载到共享主干"""
        with torch.no_grad():
            self.shared_linear.weight.copy_(pretrained_weight)
            if pretrained_bias is not None and self.shared_linear.bias is not None:
                self.shared_linear.bias.copy_(pretrained_bias)


class MERMultiheadAttention(nn.Module):
    """MER多头注意力：为Q/K/V投影添加模态路由LoRA"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        modalities: List[str],
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q/K/V投影使用MER路由
        self.q_proj = MERLinear(embed_dim, embed_dim, modalities, lora_rank, lora_alpha, dropout, bias)
        self.k_proj = MERLinear(embed_dim, embed_dim, modalities, lora_rank, lora_alpha, dropout, bias)
        self.v_proj = MERLinear(embed_dim, embed_dim, modalities, lora_rank, lora_alpha, dropout, bias)
        
        # 输出投影也使用MER路由
        self.out_proj = MERLinear(embed_dim, embed_dim, modalities, lora_rank, lora_alpha, dropout, bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        modality: str,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MER多头注意力前向
        Args:
            query, key, value: [B, seq_len, embed_dim] 
            modality: 当前模态
            attn_mask: 注意力掩码
        Returns:
            [B, seq_len, embed_dim] 注意力输出
        """
        B, seq_len, _ = query.shape
        
        # Q/K/V投影（带模态路由）
        Q = self.q_proj(query, modality)    # [B, seq_len, embed_dim]
        K = self.k_proj(key, modality)
        V = self.v_proj(value, modality)
        
        # 重塑为多头形式
        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用PyTorch 2.x的SDPA（推荐方案B：内存高效+数值稳健）
        try:
            import torch.nn.functional as F
            drop_p = self.dropout.p if self.training else 0.0
            
            # 启用FlashAttention等内存优化内核
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                # attn_mask转换为bool类型，确保数值稳定
                bool_mask = None
                if attn_mask is not None:
                    if attn_mask.dtype != torch.bool:
                        bool_mask = attn_mask.to(torch.bool)
                    else:
                        bool_mask = attn_mask
                
                attn_out = F.scaled_dot_product_attention(
                    Q, K, V, 
                    attn_mask=bool_mask,
                    dropout_p=drop_p,
                    is_causal=False
                )
        except (AttributeError, ImportError):
            # 回退到安全的手写实现（方案A：安全softmax）
            # 计算注意力得分
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, seq_len, seq_len]
            
            # 安全的mask处理
            if attn_mask is not None:
                # 1) 确保mask为布尔类型
                if attn_mask.dtype != torch.bool:
                    attn_mask = attn_mask.to(torch.bool)
                
                # 2) 用-inf替代-1e9，dtype安全
                attn_scores = attn_scores.masked_fill(~attn_mask, -float("inf"))
                
                # 3) 检查全遮行，避免softmax(全-inf)产生NaN
                all_masked = torch.isneginf(attn_scores).all(dim=-1, keepdim=True)  # [B, H, L, 1]
                if all_masked.any():
                    # 给全遮行的对角位置设为0（自注意力占位）
                    seq_len = attn_scores.size(-1)
                    eye = torch.eye(seq_len, device=attn_scores.device, dtype=torch.bool)
                    eye = eye.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
                    attn_scores = torch.where(
                        all_masked & eye, 
                        torch.zeros_like(attn_scores), 
                        attn_scores
                    )
            
            # 4) 以float32做softmax，再回到原dtype（数值稳定）
            attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
            attn_weights = self.dropout(attn_weights)
            
            # 注意力输出
            attn_out = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, self.embed_dim)
        
        # 输出投影（带模态路由）
        output = self.out_proj(attn_out, modality)
        
        return output
    
    def load_pretrained_weights(
        self, 
        q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor, out_weight: torch.Tensor,
        q_bias: Optional[torch.Tensor] = None, k_bias: Optional[torch.Tensor] = None, 
        v_bias: Optional[torch.Tensor] = None, out_bias: Optional[torch.Tensor] = None
    ):
        """从预训练注意力权重加载"""
        self.q_proj.load_pretrained_weights(q_weight, q_bias)
        self.k_proj.load_pretrained_weights(k_weight, k_bias)
        self.v_proj.load_pretrained_weights(v_weight, v_bias)
        self.out_proj.load_pretrained_weights(out_weight, out_bias)


class MERMLP(nn.Module):
    """MER MLP：为前馈网络添加模态路由LoRA"""
    
    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        modalities: List[str],
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        
        # 两层MLP，都使用MER路由
        self.fc1 = MERLinear(embed_dim, mlp_dim, modalities, lora_rank, lora_alpha, dropout)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) 
        self.fc2 = MERLinear(mlp_dim, embed_dim, modalities, lora_rank, lora_alpha, dropout)
    
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        MER MLP前向
        Args:
            x: [..., embed_dim] 输入特征
            modality: 当前模态
        Returns:
            [..., embed_dim] MLP输出
        """
        x = self.fc1(x, modality)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x, modality)
        return x
    
    def load_pretrained_weights(
        self,
        fc1_weight: torch.Tensor, fc2_weight: torch.Tensor,
        fc1_bias: Optional[torch.Tensor] = None, fc2_bias: Optional[torch.Tensor] = None
    ):
        """从预训练MLP权重加载"""
        self.fc1.load_pretrained_weights(fc1_weight, fc1_bias)
        self.fc2.load_pretrained_weights(fc2_weight, fc2_bias)

