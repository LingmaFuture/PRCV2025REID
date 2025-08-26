## PRCV2025REID · 多模态人员重识别（CLIP + MER + SDM）

### 项目概览

PRCV2025REID 是一个面向多模态人员重识别（Re-ID）的训练/评测框架，统一支持 5 种模态：`vis / nir / sk / cp / text`。

- 主干以 CLIP 统一编码器为核心（视觉/文本统一空间）
- 引入 MER（Modality-Expert Router，LoRA 形式）进行模态路由/融合
- 采用 SDM（Semantic Disentanglement Module）以 `vis` 为锚做跨模态语义对齐
- 训练严格对齐评测协议：实例为“vis↔text 锚点” +（nir/sk/cp 身份级随机），批内 P×K 强配对


### 核心特性

- 同名模态映射：全项目统一使用 `vis / nir / sk / cp / text`，并兼容旧命名（rgb/ir/sketch/cpencil/txt）。
- 实例构造与评测协议完全对齐：
  - 固定成对：`vis ↔ text`
  - 身份级池：`nir / sk / cp` 从同一身份池随机抽样（`sk/cp` 支持 `front/back/side` 视角均衡）
- P×K 强配对采样器（Strict）：每个批次保证每个 ID 含 `vis + non-vis`，稳定触发跨模态对比。
- 采样能力分析“从估算到精确”：以“多模态样本”为单位统计，精确计算可配对容量与 batch 上限。
- 训练稳定性：混合精度（AMP/bfloat16）、梯度累积、余弦退火+热身、（可选）自适应梯度裁剪。
- 工程修复：
  - 统一并修复 `actual_batch_size = P×K` 逻辑
  - 清理模态别名与冗余逻辑；`PatchEmbeds` 改为 `vis/nir/sk/cp`
  - 采样统计/容量估算逻辑与采样器完全对齐


### 数据与实例（Instance）

- JSON 标注（`data/train/text_annos.json`）仅列出 `vis` 路径与对应 `caption`（文本），即：`vis ↔ text` 严格成对。
- 对于同一 `person_id`：
  - `nir / sk / cp` 与具体某帧 `vis` 并不逐帧对齐，只保证“同身份”。
  - `sk / cp` 进一步按 `front/back/side` 视角组织，支持训练时视角均衡抽样。

一个训练 instance 的构成：

- 固定：锚点 `vis`（来自 JSON 的该条目）+ 对应 `text`
- 随机：从该 ID 的 `nir/sk/cp` 池中各随机抽 1 张（`sk/cp` 先选目标视角、无则回退）


### 训练流程

1) 读取配置（含 P、K、AMP、调度器、模态列表等）

2) 数据集构建（`datasets/dataset.py`）：
   - 从 JSON 的 `vis` 锚点扩展出多模态样本结构：
     - `images['vis']` 只含锚点图；`nir` 为同 pid 下所有红外；`sk/cp` 按视角分组
     - 生成 `modality_mask` 标记可用模态
   - 智能采样（`__getitem__`）：
     - `vis` 用锚点；`nir` 身份级随机；`sk/cp` 视角优先 + 回退；可启用模态 dropout

3) 批采样（`datasets/dataset.py`）：
   - `ModalAwarePKSampler_Strict` / `ModalAwarePKBatchSampler_Strict`
   - 建立 `pid -> {vis, nonvis}` 索引，`strong_ids` 为两侧都存在的身份
   - 每个 batch 选 P 个 ID，每 ID 取 `K//2 vis + K//2 nonvis`（短缺时回退/复用）

4) 前向（`models/model.py`）：
   - CLIP 编码（视觉/文本） → MER（LoRA 路由/融合） → BN-Neck → 分类头
   - 使用 `modality_mask` 仅对有效模态建模/计损

5) 损失：
   - ID 分类（CE）+ SDM 跨模态对齐（以 `vis` 为锚）

6) 训练细节：
   - AMP（`bfloat16` 推荐）、梯度累积、Warmup+Cosine 调度

7) 评测/保存：
   - 对齐 MM-2/3/4 协议生成查询组；导出指标与权重


### 模型架构

- CLIP 统一编码器（视觉/文本共享表征空间）
- 非共享 PatchEmbeds：`vis/nir/sk/cp` 独立 patch 投影（`models/patch_embeds.py`）
- MER（Modality-Expert Router, LoRA）：按模态进行轻量路由与融合
- BN-Neck + 线性分类头（多类 ID 分类）
- SDM：以 `vis` 为锚的跨模态语义对齐损失，提升跨模态检索一致性


### 关键配置

```python
# configs/config.py（节选）
modalities = ['vis', 'nir', 'sk', 'cp', 'text']

# 采样结构（P×K）
num_ids_per_batch = 4   # P：每个 batch 的身份数（≥3 建议）
instances_per_id = 2    # K：每个身份的实例数（≥2，强配对至少需要 2）

# 训练稳定性
gradient_accumulation_steps = 2
amp_dtype = "bfloat16"  # 或 autocast 默认 fp16
```


### 快速开始

```bash
# 1) 激活环境
conda activate prvc

# 2) 训练（默认配置）
python train.py

# 3) 评测（多模态协议）
python tools/eval_mm_protocol.py

# 4) 生成提交
python tools/generate_submission.py
```


### 工程结构

- 训练入口：`train.py`
- 配置：`configs/config.py`
- 数据集/采样器：`datasets/dataset.py`
- 模型：`models/model.py`
- PatchEmbeds：`models/patch_embeds.py`
- 评测工具：`tools/eval_mm_protocol.py`
- 数据划分与验证：`tools/split.py`


### 常见问题（FAQ）

- Q：实例（instance）是什么？
  - A：一次训练输入的“多模态样本组”。固定为 `vis↔text` 锚点，`nir/sk/cp` 从身份池随机抽取 1 张。

- Q：为什么需要 P×K 强配对？
  - A：确保每个 batch 中每个身份同时包含 `vis` 与 `non-vis`，稳定触发跨模态对齐与对比学习。

- Q：`modality_mask` 的作用？
  - A：前向/损失仅对有效模态参与；被 dropout 或缺失的模态不计损，保障鲁棒性。


### 版本修复要点

- 统一模态命名与映射；删除冗余/冲突逻辑
- 修复 `actual_batch_size = P×K` 与相关计算链路
- 采样统计从“模态实例”改为“多模态样本”，容量估算与采样器严格对齐
- `PatchEmbeds`、`Model`、`Config` 全面切换至 `vis/nir/sk/cp/text`


### 致谢

感谢相关论文与开源实现为本项目提供的灵感与基础。若使用本项目，请在论文或报告中注明来源。


