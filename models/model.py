# model.py — 优化可运行版
import math
from typing import Dict, List, Optional, Tuple
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoTokenizer

# CLIP相关导入
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("警告: CLIP未安装，将使用默认的视觉+文本编码器")


# -------------------------
# CLIP适配器
# -------------------------
class CLIPAdapter(nn.Module):
    """CLIP模型适配器，用于多模态ReID"""
    
    def __init__(self, clip_model_name="ViT-B/32", target_dim=768):
        super().__init__()
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP未安装，请安装: pip install git+https://github.com/openai/CLIP.git")
        
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device="cpu")
        self.clip_dim = self.clip_model.visual.output_dim  # 通常512或768
        
        # 冻结CLIP的大部分参数，只微调适配层
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 视觉和文本的适配层
        self.visual_adapter = nn.Sequential(
            nn.Linear(self.clip_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_adapter = nn.Sequential(
            nn.Linear(self.clip_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def encode_image(self, images):
        """编码图像，输入应该已经预处理过"""
        with torch.no_grad():
            # CLIP需要特定的预处理，但这里假设外部已经处理
            image_features = self.clip_model.encode_image(images)
            image_features = image_features.float()
        
        return self.visual_adapter(image_features)
    
    def encode_text(self, texts):
        """编码文本"""
        # 文本tokenization
        text_tokens = clip.tokenize(texts, truncate=True).to(next(self.parameters()).device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features.float()
        
        return self.text_adapter(text_features)


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
# 损失函数
# -------------------------
class AdvancedContrastiveLoss(nn.Module):
    """
    稳定的 SupCon + 跨模态 InfoNCE，并带 top-k 硬负样本正则
    - features: (B, C)（融合后）
    - labels: (B,)
    - modal_dict: Dict[str, Tensor]，各模态 (B, C)
    """
    def __init__(self, temperature: float = 0.07, margin: float = 0.2, topk: int = 5,
                 w_fused: float = 1.0, w_xmodal: float = 1.0, anchor_modal: str = 'vis'):
        super().__init__()
        self.tau = temperature
        self.margin = margin
        self.topk = topk
        self.w_fused = w_fused
        self.w_xmodal = w_xmodal
        self.anchor_modal = anchor_modal

    @staticmethod
    def _norm(x):
        return F.normalize(x, dim=-1)

    def _supcon(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 去自对角的监督对比（多正样本求和）
        B = feats.size(0)
        feats = self._norm(feats)
        sim = feats @ feats.t() / self.tau  # (B, B)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t())  # (B, B)
        eye = torch.eye(B, device=feats.device, dtype=torch.bool)
        pos_mask = pos_mask & (~eye)

        # 对每一行：-log( sum_pos exp / sum_all_except_self exp )
        logits = sim
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # 数值稳定
        exp_logits = torch.exp(logits)

        # 分母：去掉自身
        denom = exp_logits.masked_fill(eye, 0.0).sum(dim=1, keepdim=True)  # (B,1)

        # 分子：正样本之和
        pos_exp = exp_logits * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)

        valid = pos_mask.any(dim=1)
        if valid.any():
            loss = -torch.log((pos_sum[valid] + 1e-12) / (denom[valid, 0] + 1e-12)).mean()
        else:
            loss = torch.tensor(0.0, device=feats.device)
        return loss

    def _xmodal_nce(self, fa, fb, labels, ma=None, mb=None):
        """
        a->b 的 InfoNCE：仅在两端样本都“存在该模态”的位置上计算
        fa, fb: (B, C) ; ma, mb: (B,) in {0,1}
        """
        if (ma is not None) and (mb is not None):
            valid = (ma > 0.5) & (mb > 0.5)
            if valid.sum().item() < 2:
                return torch.tensor(0.0, device=fa.device)
            fa, fb, labels = fa[valid], fb[valid], labels[valid]

        B = fa.size(0)
        fa = F.normalize(fa, dim=-1)
        fb = F.normalize(fb, dim=-1)
        sim = fa @ fb.t() / self.tau  # (B,B)

        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t())  # (B,B)

        sim = sim - torch.max(sim, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim)
        pos_exp = exp_sim * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)
        denom = exp_sim.sum(dim=1)

        valid_row = pos_mask.any(dim=1)
        if valid_row.any():
            loss = -torch.log((pos_sum[valid_row] + 1e-12) / (denom[valid_row] + 1e-12)).mean()
        else:
            loss = torch.tensor(0.0, device=fa.device)
        return loss


    def _hard_negative_reg(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = self._norm(feats)
        sim = feats @ feats.t()  # (B,B)
        B = sim.size(0)

        eye = torch.eye(B, device=feats.device, dtype=torch.bool)
        pos = labels.view(-1,1).eq(labels.view(1,-1))
        neg = (~pos) & (~eye)

        sim_neg = sim.masked_fill(~neg, -1e4)
        k = min(self.topk, max(1, B - 1))
        hard_vals, _ = sim_neg.topk(k, dim=1)  # (B,k)
        # 想要负样本相似度 < 1 - margin
        target = 1.0 - self.margin
        reg = F.relu(hard_vals - target).mean()
        return reg

    def forward(self, fused, labels, modal_dict=None, modal_masks=None):
        # 主损失：融合特征的监督对比
        main_loss = self._supcon(fused, labels)
        total_loss = self.w_fused * main_loss
        
        # 跨模态损失：约束不同模态间的一致性
        if self.w_xmodal > 0 and modal_dict is not None and len(modal_dict) >= 2:
            xmodal_loss = 0.0
            num_pairs = 0
            
            # 获取可用的模态组合
            available_modalities = list(modal_dict.keys())
            
            # 计算所有模态对之间的跨模态InfoNCE损失
            for i, mod_a in enumerate(available_modalities):
                for j, mod_b in enumerate(available_modalities):
                    if i < j:  # 避免重复计算
                        # 获取模态掩码
                        mask_a = modal_masks.get(mod_a, None) if modal_masks else None
                        mask_b = modal_masks.get(mod_b, None) if modal_masks else None
                        
                        # 计算双向InfoNCE损失
                        loss_ab = self._xmodal_nce(modal_dict[mod_a], modal_dict[mod_b], labels, mask_a, mask_b)
                        loss_ba = self._xmodal_nce(modal_dict[mod_b], modal_dict[mod_a], labels, mask_b, mask_a)
                        
                        xmodal_loss += (loss_ab + loss_ba) / 2.0
                        num_pairs += 1
            
            if num_pairs > 0:
                xmodal_loss = xmodal_loss / num_pairs
                total_loss += self.w_xmodal * xmodal_loss
        
        # 硬负样本正则化（可选）
        if hasattr(self, 'w_hard_neg') and getattr(self, 'w_hard_neg', 0) > 0:
            hard_neg_loss = self._hard_negative_reg(fused, labels)
            total_loss += self.w_hard_neg * hard_neg_loss
        
        return total_loss


# -------------------------
# 模型主体
# -------------------------
class MultiModalReIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 统一的融合维度（后续所有特征都映射到这个维度）
        self.fusion_dim = getattr(config, "fusion_dim", 768)
        
        # 检查是否使用CLIP
        self.use_clip = getattr(config, "use_clip", False) and CLIP_AVAILABLE
        
        if self.use_clip:
            self._init_clip_encoders(config)
        else:
            self._init_standard_encoders(config)
            
        # 共同的组件初始化
        self._init_common_components(config)
            
    def _init_clip_encoders(self, config):
        """初始化CLIP编码器"""
        clip_model_name = getattr(config, "clip_model_name", "ViT-B/32")
        print(f"使用CLIP模型: {clip_model_name}")
        
        self.clip_adapter = CLIPAdapter(clip_model_name, self.fusion_dim)
        
        # CLIP模式下简化架构
        self.vision_backbone = None
        self.text_encoder = None
        self.text_tokenizer = None
        
        # 仍然保留模态适配器用于不同视觉模态的区分
        self.modality_adapters = nn.ModuleDict({
            'vis': VectorModalityAdapter(self.fusion_dim),
            'nir': VectorModalityAdapter(self.fusion_dim),
            'sk':  VectorModalityAdapter(self.fusion_dim),
            'cp':  VectorModalityAdapter(self.fusion_dim),
        })
        
        # 文本特征缓存
        self.text_cache = {}
        
    def _init_standard_encoders(self, config):
        """初始化标准编码器（非CLIP）"""
        
        # ===== 视觉骨干：支持 ViT 或 ResNet =====
        self.backbone_name = getattr(config, "backbone", "resnet50")
        use_pretrained = getattr(config, "use_pretrained_vision", False)

        if self.backbone_name.startswith("vit"):
            # --- ViT 分支 ---
            self.backbone_type = "vit"
            self.vision_backbone = timm.create_model(
                self.backbone_name,  # e.g., 'vit_base_patch16_224'
                pretrained=use_pretrained,
                num_classes=0,
            )
            self.vision_out_dim = self.vision_backbone.num_features  # 通常 768

            # Token 级模态适配器（保留你的实现）
            self.modality_adapters = nn.ModuleDict({
                'vis': ModalityAdapter(self.vision_out_dim),
                'nir': ModalityAdapter(self.vision_out_dim),
                'sk':  ModalityAdapter(self.vision_out_dim),
                'cp':  ModalityAdapter(self.vision_out_dim),
            })

        else:
            # --- ResNet / CNN 分支 ---
            self.backbone_type = "cnn"
            # timm 的 resnet50：num_classes=0 + global_pool='avg' -> 直接输出 (B, C)
            self.vision_backbone = timm.create_model(
                self.backbone_name,  # e.g., 'resnet50'
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

        # ===== 文本编码器（可控微调策略）=====
        txt_name = getattr(config, "text_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.text_tokenizer = AutoTokenizer.from_pretrained(txt_name)
        self.text_encoder  = AutoModel.from_pretrained(txt_name)
        
        # 可控微调策略
        text_finetune_strategy = getattr(config, "text_finetune_strategy", "top_layers")
        
        if text_finetune_strategy == "frozen":
            # 完全冻结
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.freeze_text = True
        elif text_finetune_strategy == "top_layers":
            # 只微调顶层（最后2层）
            self.freeze_text = False
            num_layers = len(self.text_encoder.encoder.layer)
            
            # 冻结embedding和前面的层
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
            
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i < num_layers - 2:  # 冻结前面的层
                    for p in layer.parameters():
                        p.requires_grad = False
                        
            # 保持pooler可训练
            if hasattr(self.text_encoder, 'pooler') and self.text_encoder.pooler is not None:
                for p in self.text_encoder.pooler.parameters():
                    p.requires_grad = True
                    
        elif text_finetune_strategy == "lora":
            # LoRA适配（简化版本）
            # 这里可以扩展为真正的LoRA实现
            self.freeze_text = False
            # 冻结大部分参数，只训练特定的适配层
            for name, p in self.text_encoder.named_parameters():
                if 'attention.self.query' in name or 'attention.self.key' in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            # 全量微调
            self.freeze_text = False
            for p in self.text_encoder.parameters():
                p.requires_grad = True

        # 动态获取文本编码器维度
        self.text_in_dim = self.text_encoder.config.hidden_size  # 自动获取维度
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_in_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _init_common_components(self, config):
        """初始化通用组件"""
        # ===== 融合模块 / BNNeck / 分类头 / 检索头 =====
        self.fusion_module = HierarchicalMultiModalFusion(self.fusion_dim, num_layers=3, num_heads=8)

        self.bnneck = nn.BatchNorm1d(self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, self.config.num_classes, bias=False)

        self.feature_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),  # 增强dropout
        )

        self.contrastive_loss = AdvancedContrastiveLoss(
            temperature=getattr(config, "contrastive_tau", 0.1),  # 提高温度降低锐度
            margin=getattr(config, "contrastive_margin", 0.2),
            topk=getattr(config, "hard_topk", 5),
            w_fused=1.0, w_xmodal=0.5, anchor_modal='vis'  # 启用跨模态损失
        )

        self.ce_loss = nn.CrossEntropyLoss()
        
        # 文本特征缓存
        if not hasattr(self, 'text_cache'):
            self.text_cache = {}  # {text_str: encoded_features}
        self.cache_enabled = True


    # ---- 工具 ----
    @property
    def device(self):
        return next(self.parameters()).device

    def _vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回 ViT 的完整 token 序列（含 CLS），形状 (B, 1+N, C)
        """
        assert self.backbone_type == "vit", "Only ViT backbone uses _vit_tokens()"
        m = self.vision_backbone
        x = m.patch_embed(x)  # (B, N, C)
        cls = m.cls_token.expand(x.size(0), -1, -1)  # (B, 1, C)
        x = torch.cat((cls, x), dim=1)  # (B, 1+N, C)

        # 位置编码可能比序列长（timm 兼容插值），这里对齐前缀长度
        pos_embed = m.pos_embed[:, :x.size(1), :]
        x = x + pos_embed
        x = m.pos_drop(x)

        # Transformer blocks
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        return x  # (B, 1+N, C)

    def encode_image(self, image: torch.Tensor, modality_type: str) -> torch.Tensor:
        """
        返回 (B, fusion_dim)
        支持CLIP和标准编码器两种模式
        """
        if self.use_clip:
            # CLIP模式
            feats = self.clip_adapter.encode_image(image)  # (B, fusion_dim)
            # 应用模态特定的适配
            if modality_type in self.modality_adapters:
                feats = self.modality_adapters[modality_type](feats)
            return feats
        
        else:
            # 标准编码器模式
            if self.backbone_type == "vit":
                tokens = self._vit_tokens(image)  # (B, 1+N, C)
                if modality_type in self.modality_adapters:
                    tokens = self.modality_adapters[modality_type](tokens)

                if tokens.size(1) > 1:
                    patch = tokens[:, 1:, :]
                    feats = patch.mean(dim=1)     # (B, vision_out_dim)
                else:
                    feats = tokens.squeeze(1)

            else:
                # CNN 分支：timm 返回 (B, C) 已经是全局池化
                feats = self.vision_backbone(image)  # (B, vision_out_dim)
                if modality_type in self.modality_vec_adapters:
                    feats = self.modality_vec_adapters[modality_type](feats)

            # 统一到 fusion_dim
            feats = self.visual_projection(feats)    # (B, fusion_dim)
            return feats



    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        batch_size = len(text_descriptions)
        if batch_size == 0:
            return torch.zeros(0, self.fusion_dim, device=self.device)

        valid_texts = [t if isinstance(t, str) and len(t) > 0 else "[UNK]" for t in text_descriptions]
        
        if self.use_clip:
            # CLIP模式：直接使用CLIP编码文本
            return self.clip_adapter.encode_text(valid_texts)
        
        # 标准编码器模式（原有逻辑）
        
        # 尝试从缓存获取特征
        if self.cache_enabled:
            cached_features = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(valid_texts):
                if text in self.text_cache:
                    cached_features.append(self.text_cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # 如果全部命中缓存
            if len(uncached_texts) == 0:
                return torch.stack(cached_features).to(self.device)
            
            # 只对未缓存的文本进行编码
            if len(uncached_texts) > 0:
                enc = self.text_tokenizer(uncached_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                enc = {k: v.to(self.device) for k, v in enc.items()}

                ctx = torch.no_grad() if self.freeze_text else contextlib.nullcontext()
                with ctx:
                    out = self.text_encoder(**enc, return_dict=True)
                    token = out.last_hidden_state  # (B, L, 384)

                mask = enc['attention_mask'].unsqueeze(-1).float()
                text_vec = (token * mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # (B, 384)
                text_vec = self.text_projection(text_vec)  # (B, fusion_dim)
                
                # 缓存新编码的特征
                for i, text in enumerate(uncached_texts):
                    self.text_cache[text] = text_vec[i].detach().cpu()
            
            # 组合缓存和新编码的特征
            result_features = [None] * batch_size
            
            # 填入缓存的特征
            cached_idx = 0
            for i in range(batch_size):
                if i not in uncached_indices:
                    result_features[i] = cached_features[cached_idx].to(self.device)
                    cached_idx += 1
            
            # 填入新编码的特征
            for i, orig_idx in enumerate(uncached_indices):
                result_features[orig_idx] = text_vec[i]
            
            return torch.stack(result_features)
        
        else:
            # 原始编码逻辑（无缓存）
            enc = self.text_tokenizer(valid_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}

            ctx = torch.no_grad() if self.freeze_text else contextlib.nullcontext()
            with ctx:
                out = self.text_encoder(**enc, return_dict=True)
                token = out.last_hidden_state  # (B, L, 384)

            mask = enc['attention_mask'].unsqueeze(-1).float()
            text_vec = (token * mask).sum(1) / mask.sum(1).clamp_min(1e-6)  # (B, 384)
            text_vec = self.text_projection(text_vec)  # (B, fusion_dim)
            return text_vec


    # ---- 前向 ----
    def forward(self, batch: Dict, return_features: bool = False):
        """
        batch 结构约定：
        {
            'images': {
                'vis': Tensor[B,3,224,224],  # 可能子集
                'nir': Tensor[B,3,224,224],
                'sk' : ...
                'cp' : ...
            },
            'text_description': List[str] (len=B),
            'modality_mask': Optional[Dict[str, bool]]
        }
        """
        # model.py (inside MultiModalReIDModel.forward)

        images: Dict[str, torch.Tensor] = batch['images']
        text_descriptions: List[str] = batch.get('text_description', [])
        raw_mm = batch.get('modality_mask', {})

        B = next(iter(images.values())).size(0)
        modality_features, modality_masks = {}, {}

        def _mask_vec(name):
            v = raw_mm.get(name, None)
            if isinstance(v, torch.Tensor):
                return v.to(self.device).float().view(-1, 1)   # [B,1]
            elif isinstance(v, (list, tuple)):
                return torch.tensor(v, device=self.device, dtype=torch.float32).view(-1,1)
            return torch.ones(B, 1, device=self.device)

        # 图像模态 - 只对真正存在的模态进行编码
        for m in ['vis','nir','sk','cp']:
            if m in images:
                mvec = _mask_vec(m)  # [B,1]
                
                # 检查是否有任何样本在此模态上有效
                if mvec.sum().item() > 0:
                    # 只对有效的样本进行编码
                    valid_mask = mvec.squeeze(1).bool()  # [B]
                    
                    if valid_mask.any():
                        # 创建完整的特征张量
                        feats = torch.zeros(B, self.fusion_dim, device=self.device)
                        
                        # 只对有效样本进行编码
                        if valid_mask.all():
                            # 所有样本都有效，直接编码
                            feats = self.encode_image(images[m].to(self.device), m)
                        else:
                            # 部分样本有效，只编码有效部分
                            valid_images = images[m][valid_mask].to(self.device)
                            if len(valid_images) > 0:
                                valid_feats = self.encode_image(valid_images, m)
                                feats[valid_mask] = valid_feats
                        
                        # 应用掩码（确保无效样本特征为零）
                        feats = feats * mvec
                        modality_features[m] = feats
                        modality_masks[m] = mvec.squeeze(1)  # [B]

        # 文本模态 - 只对有效文本进行编码
        if 'text' in raw_mm:
            tvec = _mask_vec('text')  # [B,1]
            
            # 检查是否有有效的文本描述
            valid_text_mask = tvec.squeeze(1).bool()  # [B]
            has_valid_text = any(
                valid_text_mask[i].item() and isinstance(text_descriptions[i], str) and text_descriptions[i].strip()
                for i in range(len(text_descriptions))
            )
            
            if has_valid_text:
                # 只对有效文本进行编码
                if valid_text_mask.all():
                    # 所有样本都有有效文本
                    tfeat = self.encode_text(text_descriptions)
                else:
                    # 部分样本有有效文本
                    tfeat = torch.zeros(B, self.fusion_dim, device=self.device)
                    
                    # 收集有效的文本描述
                    valid_texts = []
                    valid_indices = []
                    for i, (is_valid, text) in enumerate(zip(valid_text_mask, text_descriptions)):
                        if is_valid.item() and isinstance(text, str) and text.strip():
                            valid_texts.append(text)
                            valid_indices.append(i)
                    
                    if valid_texts:
                        valid_text_feats = self.encode_text(valid_texts)
                        for i, valid_idx in enumerate(valid_indices):
                            tfeat[valid_idx] = valid_text_feats[i]
                
                # 应用掩码
                modality_features['text'] = tfeat * tvec
                modality_masks['text'] = tvec.squeeze(1)

        # 仅用于决定是否在本 batch 中引入该模态的“全局”开关（True 表示本 batch 至少有一个样本该模态存在）
        mask_for_fusion = {m: bool(modality_masks.get(m, torch.zeros(B, device=self.device)).any().item())
                        for m in modality_features.keys()}

        fused_features = self.fusion_module(modality_features, mask_for_fusion)  # [B,C]
        reid = self.feature_projection(fused_features)
        reid_norm = F.normalize(reid, p=2, dim=1)

        if return_features:
            return reid_norm

        neck = self.bnneck(fused_features)
        logits = self.classifier(neck)

        return {
            'logits': logits,
            'features': fused_features,
            'reid_features': reid_norm,
            'modality_features': modality_features,
            'modality_masks': modality_masks,    # <<< 新增：带出逐样本掩码
            'reid_features_raw': reid,
        }


    # model.py 中 compute_loss 里把掩码传进去，并返回详细损失信息
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch: Dict=None):
        ce = self.ce_loss(outputs['logits'], labels)
        
        # 计算对比损失（包含跨模态损失）
        contrastive_total = self.contrastive_loss(
            outputs['reid_features'],
            labels,
            outputs['modality_features'],
            modal_masks=outputs.get('modality_masks', None)
        )
        
        # 使用配置中的对比损失权重
        contrastive_weight = getattr(self.config, 'contrastive_weight', 0.1)
        total = ce + contrastive_weight * contrastive_total
        
        return {
            'total_loss': total, 
            'ce_loss': ce, 
            'contrastive_loss': contrastive_total,
            'contrastive_weight': contrastive_weight
        }

