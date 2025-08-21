## PRCV2025 多模态行人重识别（CLIP + MER + SDM）

一个面向竞赛与实际应用的多模态 ReID 框架：以 CLIP-B/16 作为统一编码器，结合 MER 模态路由 LoRA 与 SDM 语义分离对齐，统一学习 RGB/IR/Sketch/CPencil/Text 等模态的可检索表征；训练侧强调稳定性与一致性，评估侧严格对齐训练表征。

---

## 技术亮点（TL;DR）

- 统一编码器：共享 CLIP-B/16 backbone，继承大规模图文预训练的跨模态能力
- 模态特异适配（MER）：在 Q/K/V 与 MLP 层插入 LoRA，按模态路由，参数高效且易扩展
- 语义分离对齐（SDM）：RGB 锚定的跨模态对齐损失，训练时将各模态拉齐到统一语义空间
- 掩码与占位：精确的模态掩码与可学习 null token，杜绝“零张量污染”融合
- 轻量融合器：多头注意力 + MLP Mixer，原生支持模态缺失的加权融合
- BNNeck 检索一致性：训练用的对齐损失与评估检索统一使用 bn_features
- 稳定训练：对齐损失长周期 warmup、特征范数正则、AMP + 自适应梯度裁剪、TF32 加速
- 赛制对齐评测：画廊固定 vis，查询组合多模态，mAP/CMC 一致评估

---

## 架构总览

数据（多模态） → 非共享 Tokenizer → CLIP 统一编码器（含 MER 路由） → 语义分离（SDM） → 轻量特征融合（掩码友好） → BNNeck → 分类头/检索特征

- 统一编码器：`models/clip_backbone.py`（共享 ViT-B/16 主干，文本使用 CLIP 文本塔）
- 多模态 Tokenizer：`models/patch_embeds.py`（RGB/IR/Sketch/CPencil 独立 PatchEmbed，自动 1↔3 通道适配）
- MER 模态路由：`models/mer_lora.py`（按模态注入 LoRA 到 Q/K/V 与 MLP）
- SDM 模块与损失：`models/model.py`（语义分离 + RGB 锚定对齐）
- 特征融合器：`models/model.py`（Multi-Head Attention + MLP Mixer，带 key_padding_mask）
- BNNeck + 线性分类器：`models/model.py`

### 关键一致性原则

- 训练与评估统一以 BNNeck 后的 `bn_features` 作为检索表征，并在相似度计算前做 L2 归一化；保证与对齐损失的表征完全一致。

---

## 组件细节

### 1) CLIP 统一编码器 + MER 路由（`models/clip_backbone.py`, `models/mer_lora.py`）

- 使用 CLIP ViT-B/16 共享主干，视觉层按层引入 MER LoRA：
  - 注入位置：自注意力的 Q/K/V、输出投影；FFN 的 FC1/FC2
  - 每模态独立 LoRA 分支，`rank=4, alpha=1.0`，共享主干参数保持稳定
- 文本通过 CLIP 文本编码器，是否冻结可配

### 2) 多模态非共享 Tokenizer（`models/patch_embeds.py`）

- RGB/CPencil：3 通道；IR/Sketch：1 通道
- 自动通道适配（如 1→3 使用 1×1 卷积适配）以对齐 CLIP ViT 输入

### 3) 语义分离对齐 SDM（`models/model.py`）

- 多头注意力进行语义分离，线性投影至统一语义维度（默认 512）
- RGB 锚定对齐损失：对每个非 RGB 模态与 RGB 建立对比约束（温度/边距可配）

### 4) 掩码友好特征融合器（`models/model.py`）

- 堆叠多模态特征后以 Multi-Head Attention + MLP Mixer 融合
- 支持 `key_padding_mask`，并在输出侧做带掩码的加权平均，天然适配“缺模态”

### 5) BNNeck + 线性分类器（`models/model.py`）

- BNNeck 输出 `bn_features`（检索向量），Dropout 后接线性分类器
- 训练损失与评估检索均基于 `bn_features`，强化一致性

---

## 训练流程（`train.py`）

### 数据与采样

- 数据根目录与标注：`configs/config.py`
  - `data_root=./data/train`，图像目录：`vis/`, `nir/`, `sk/`, `cp/`
  - `json_file=./data/train/text_annos.json`（字段：`file_path`, `caption`）
- 数据集实现：`datasets/dataset.py`
  - 构建样本清单与模态路径缓存，训练增强/验证变换分离
  - 返回：`images`、`text_description`、`modality_mask`（按样本与模态精确标注可用性）
- 采样器：`BalancedBatchSampler`（P×K）
  - 默认 `batch_size=32, K=4 → P=8`，为对比目标提供稳定的正样本数

### 前向与掩码/占位机制

- 批处理转模型输入：`convert_batch_for_clip_model`
  - 按 `modality_mask` 过滤无效模态；文本无效位置以空字符串占位
- 缺失模态用“可学习 null token”占位，类型/精度与有效特征对齐，避免零张量污染
- 融合器使用 `key_padding_mask` 忽略无效模态，输出侧再做掩码加权平均

### 损失与正则

- 总损失 = `CE（ID 分类）` + `λ · SDM 对齐` + `特征范数正则`
  - SDM 对齐：以 `bn_features` 作为对齐特征
  - 特征范数正则：目标范数≈10，带容忍带宽（防止“靠放大范数取巧”）

### 训练稳定性与调度

- 对齐损失权重长周期 warmup：前 5 个 epoch 关闭，6→25 线性升温，再恒定
- AMP + GradScaler（CUDA 自动开启），开启 TF32；自适应梯度裁剪（分位数阈）
- 学习率策略（可选）：warmup+cosine（默认）/ step / multistep / plateau
- 分层学习率：
  - CLIP backbone：1e-5（低学习率，保留预训练能力）
  - MER/Tokenizer/Projections/其他模块：5e-5（更快适配新模态/任务）

### 日志与监控

- 记录总损失/CE/SDM/特征范数/梯度范数/异常 spike 次数
- 训练前期打印每 batch 的 ID 组成与 K 值，便于检查 P×K 采样是否生效

---

## 评估流程（赛制对齐）

- 画廊（Gallery）：只使用 `vis` 模态
- 查询（Query）：在 `nir/sk/cp/text` 上构造单/双/三/四模态组合（脚本已内置）
- 特征提取：严格使用 `bn_features` 并做 L2 归一化；相似度 = 余弦相似度（内积）
- 指标：`mAP@100`、`CMC@1/5/10`，并输出各单模态 mAP 便于定位“短板模态”

---

## 快速开始（Windows/PowerShell）

### 准备环境

```powershell
conda activate prvc
pip install -r requirements.txt
```

### 数据准备

目录结构示意：

```
data/train/
  ├─ vis/0001/*.jpg  ─┐
  ├─ nir/0001/*.jpg  ─┤ 可选，不同身份以四位目录名区分（0001、0002、…）
  ├─ sk/0001/*.jpg   ─┤
  └─ cp/0001/*.jpg   ─┘
  └─ text_annos.json   # [{"file_path": "vis/0001/xxx.jpg", "caption": "…"}, ...]
```

### 训练

```powershell
python train.py
```

如需调整超参，编辑 `configs/config.py`（如学习率、调度器、对齐损失权重、评估频率等）。

---

## 关键配置（`configs/config.py`，部分）

- 模型/模态：`clip_model_name`, `modalities`, `fusion_dim`, `vision_hidden_dim`
- MER：`enable_mer`, `mer_lora_rank`, `mer_lora_alpha`
- SDM：`sdm_semantic_dim`, `sdm_temperature`, `sdm_margin`, `contrastive_weight`
- 融合器：`fusion_num_heads`, `fusion_mlp_ratio`, `fusion_dropout`
- 正则：`feature_target_norm`, `feature_norm_band`, `feature_norm_penalty`
- 训练：`num_epochs`, `batch_size`, `scheduler`, `warmup_epochs`, `weight_decay`
- 数据增强：`random_erase`, `color_jitter`
- 模态 dropout：`modality_dropout`, `min_modalities`

---

## 目录结构

```
PRCV2025REID/
├─ configs/
│  └─ config.py
├─ datasets/
│  └─ dataset.py
├─ models/
│  ├─ clip_backbone.py
│  ├─ mer_lora.py
│  ├─ patch_embeds.py
│  └─ model.py
├─ tools/
├─ docs/
│  ├─ CLIP_MER_Architecture.md
│  └─ 评估指标说明.md 等
└─ train.py
```

---

## 常见问题（FAQ）

- 评估结果波动大？
  - 检查是否严格使用 `bn_features` 做检索；对齐损失是否已过 warmup；P×K 是否满足每类≥4
- 某模态 mAP 偏低？
  - 查看日志中的单模态 mAP；适当提高 MER/Tokenizer 学习率或延长训练轮次
- 训练不稳定或损失 spike 频繁？
  - 降低学习率、开启/加强自适应梯度裁剪、减小 `modality_dropout`、检查特征范数是否过大

