# model.py — 优化可运行版
import math
from typing import Dict, List, Optional, Tuple
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoTokenizer


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
    """跨模态 Transformer 块（Pre-LN 风格，更稳）"""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_norm_q = nn.LayerNorm(embed_dim)
        self.cross_norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=attn_drop)
        self.cross_scale = LayerScale(embed_dim)

        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=attn_drop)
        self.self_scale = LayerScale(embed_dim)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(proj_drop),
        )
        self.ffn_scale = LayerScale(embed_dim)

    def forward(self, query, key_value, key_padding_mask: Optional[torch.Tensor] = None):
        # Cross-Attn
        q = self.cross_norm_q(query)
        kv = self.cross_norm_kv(key_value)
        cross_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)  # (B,1,C)
        x = query + self.cross_scale(cross_out)

        # Self-Attn
        y = self.self_norm(x)
        self_out, _ = self.self_attn(y, y, y)
        x = x + self.self_scale(self_out)

        # FFN
        z = self.ffn_norm(x)
        x = x + self.ffn_scale(self.mlp(z))
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
        # 简化版：暂时只使用主要的监督对比损失
        main_loss = self._supcon(fused, labels)
        
        # 暂时关闭跨模态损失和硬负样本正则，专注于基础特征学习
        # xmodal_loss = 0.0 (已通过w_xmodal=0.0关闭)
        # hard_neg_loss = 0.0 (暂时关闭)
        
        return self.w_fused * main_loss


# -------------------------
# 模型主体
# -------------------------
class MultiModalReIDModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 统一的融合维度（后续所有特征都映射到这个维度）
        self.fusion_dim = getattr(config, "fusion_dim", 768)

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

        # ===== 文本编码器（与配置对齐）=====
        txt_name = getattr(config, "text_model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.text_tokenizer = AutoTokenizer.from_pretrained(txt_name)
        self.text_encoder  = AutoModel.from_pretrained(txt_name)
        self.freeze_text = getattr(config, "freeze_text", True)
        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # 文本 384 -> fusion_dim
        self.text_in_dim = 384
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_in_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

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
            w_fused=1.0, w_xmodal=0.0, anchor_modal='vis'  # 暂时关闭跨模态损失
        )

        self.ce_loss = nn.CrossEntropyLoss()


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
        - ViT：tokens -> (token级)模态适配器 -> 对 patch 做 mean pool -> 视觉投影
        - ResNet：全局池化特征 -> (向量级)模态适配器 -> 视觉投影
        """
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

        # 图像模态
        for m in ['vis','nir','sk','cp']:
            if m in images:
                feats = self.encode_image(images[m].to(self.device), m)      # [B,C]
                mvec  = _mask_vec(m)                                         # [B,1]
                feats = feats * mvec                                          # 逐样本关门
                modality_features[m] = feats
                modality_masks[m]    = mvec.squeeze(1)                        # [B]

        # 文本模态
        add_text = any((isinstance(t, str) and t.strip()) for t in text_descriptions)
        if add_text and 'text' in raw_mm:
            tfeat = self.encode_text(text_descriptions)                       # [B,C]
            tvec  = _mask_vec('text')                                         # [B,1]
            modality_features['text'] = tfeat * tvec
            modality_masks['text']    = tvec.squeeze(1)

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


    # model.py 中 compute_loss 里把掩码传进去
    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch: Dict=None):
        ce = self.ce_loss(outputs['logits'], labels)
        con = self.contrastive_loss(
            outputs['reid_features'],
            labels,
            outputs['modality_features'],
            # 新增：把掩码传给对比损失
            modal_masks=outputs.get('modality_masks', None)
        )
        # 修复：使用配置中的对比损失权重0.1，而不是硬编码的0.5
        contrastive_weight = getattr(self.config, 'contrastive_weight', 0.1)
        total = ce + contrastive_weight * con
        return {'total_loss': total, 'ce_loss': ce, 'contrastive_loss': con}

