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
    稳定的 SupCon +（可选）跨模态 InfoNCE，并带 top-k 硬负样本正则
    当前默认仅启用 SupCon（与训练稳定性优先策略一致）
    """
    def __init__(self, temperature: float = 0.5, margin: float = 0.2, topk: int = 5,
                 w_fused: float = 1.0, w_xmodal: float = 0.0, anchor_modal: str = 'vis'):
        super().__init__()
        self.tau = temperature
        self.margin = margin
        self.topk = topk
        self.w_fused = w_fused
        self.w_xmodal = w_xmodal
        self.anchor_modal = anchor_modal

    #@staticmethod
    # def _norm(x):
    #     return F.normalize(x, dim=-1)
    @staticmethod
    def _norm(x):
        # 显式 eps，避免零向量导致 NaN 梯度
        return F.normalize(x, dim=-1, eps=1e-6)

    def _supcon(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = feats.size(0)
        feats = self._norm(feats)

        # 数值稳定：用 logsumexp 公式，不手写 exp 再求和
        # sim 范围控制在 [-10, 10] 后再除以 tau，避免 tau 很小造成放大
        raw = torch.matmul(feats, feats.t())
        raw = torch.clamp(raw, min=-10.0, max=10.0)
        sim = raw / max(self.tau, 1e-3)  # 防止 tau 过小
        labels = labels.view(-1, 1)
        pos_mask = labels.eq(labels.t())
        eye = torch.eye(B, device=feats.device, dtype=torch.bool)
        pos_mask = pos_mask & (~eye)

        # 以行为单位做 logsumexp
        # logits_ij = sim_ij
        # denominator = logsumexp_j(sim_ij, j!=i)
        # numerator   = logsumexp_j(sim_ij, j in P(i))
        # loss_i = -(numerator - denominator)
        sim = sim.masked_fill(eye, float('-inf'))  # 排除自身
        denom_lse = torch.logsumexp(sim, dim=1)    # (B,)

        # 若某行没有正样本，跳过
        has_pos = pos_mask.any(dim=1)
        if has_pos.any():
            pos_sim = sim.masked_fill(~pos_mask, float('-inf'))
            nume_lse = torch.logsumexp(pos_sim, dim=1)  # (B,)
            loss_vec = -(nume_lse - denom_lse)
            loss = loss_vec[has_pos].mean()
            # 末端再做一次合理截断，防极端 outlier
            loss = torch.clamp(loss, 0.0, 10.0)
        else:
            loss = torch.tensor(0.0, device=feats.device)
        return loss

    def forward(self, fused, labels, modal_dict=None, modal_masks=None):
        main_loss = self._supcon(fused, labels)
        return self.w_fused * main_loss

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
            )
            self.vision_out_dim = self.vision_backbone.num_features  # 通常 768

            # Token 级模态适配器
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

        # ===== 融合/分类/检索 =====
        self.fusion_module = HierarchicalMultiModalFusion(self.fusion_dim, num_layers=3, num_heads=8)

        self.bnneck = nn.BatchNorm1d(self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, self.config.num_classes, bias=False)

        self.feature_projection = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
        )

        self.contrastive_loss = AdvancedContrastiveLoss(
            temperature=getattr(config, "contrastive_tau", 0.1),
            margin=getattr(config, "contrastive_margin", 0.2),
            topk=getattr(config, "hard_topk", 5),
            w_fused=1.0, w_xmodal=0.0, anchor_modal='vis'  # 先只用 SupCon
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # 文本特征缓存
        self.text_cache = {}  # {text_str: encoded_feature_cpu}
        self.cache_enabled = True

    @property
    def device(self):
        return next(self.parameters()).device

    def _vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """返回 ViT 的完整 token 序列（含 CLS），形状 (B, 1+N, C)"""
        assert self.backbone_type == "vit", "Only ViT backbone uses _vit_tokens()"
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
            tokens = self._vit_tokens(image)
            if modality_type in getattr(self, 'modality_adapters', {}):
                tokens = self.modality_adapters[modality_type](tokens)
            feats = tokens[:, 1:, :].mean(dim=1) if tokens.size(1) > 1 else tokens.squeeze(1)
        else:
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

        # 图像模态
        for m in ['vis','nir','sk','cp']:
            if m in images:
                feats = self.encode_image(images[m].to(self.device), m)
                mvec  = _mask_vec(m)
                feats = feats * mvec
                modality_features[m] = feats
                modality_masks[m]    = mvec.squeeze(1)

        # 文本模态
        add_text = any((isinstance(t, str) and t.strip()) for t in text_descriptions)
        if add_text and 'text' in raw_mm:
            tfeat = self.encode_text(text_descriptions)
            tvec  = _mask_vec('text')
            modality_features['text'] = tfeat * tvec
            modality_masks['text']    = tvec.squeeze(1)

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
            'modality_features': modality_features,
            'modality_masks': modality_masks,
            'reid_features_raw': reid,
        }

    def compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, batch: Dict=None):
        ce = self.ce_loss(outputs['logits'], labels)
        con = self.contrastive_loss(
            outputs['reid_features'],
            labels,
            outputs['modality_features'],
            modal_masks=outputs.get('modality_masks', None)
        )
        contrastive_weight = getattr(self.config, 'contrastive_weight', 0.1)

        # --- 新增：特征范数正则（拉回到一个带宽内）---
        feats = outputs['features']                      # [B, C] 这是融合后、进BNNeck前的向量
        fn = feats.norm(p=2, dim=1)
        target = getattr(self.config, 'feature_target_norm', 10.0)   # 可以在 config 里放个默认 10
        band   = getattr(self.config, 'feature_norm_band', 4.0)      # 容忍带宽
        lam    = getattr(self.config, 'feature_norm_penalty', 1e-3)  # 权重很小即可

        # 只惩罚远离[target-band, target+band] 的样本（Huber-ish）
        over = (fn - (target + band)).clamp_min(0)
        under = ((target - band) - fn).clamp_min(0)
        feat_penalty = (over**2 + under**2).mean() * lam
        # -------------------------------------------

        total = ce + contrastive_weight * con + feat_penalty
        return {'total_loss': total, 'ce_loss': ce, 'contrastive_loss': con, 'feat_penalty': feat_penalty}