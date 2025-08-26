# SDM（Similarity Distribution Matching）实战整理（Markdown + LaTeX）

> 面向 PRCV2025 全模态行人重识别任务的可落地实现与排障手册。公式以 LaTeX 表示，可直接渲染于支持数学公式的 Markdown 渲染器。

---

## 目录
- [1. SDM 是什么](#1-sdm-是什么)
- [2. 计算步骤（多模态可直接套用）](#2-计算步骤多模态可直接套用)
- [3. 稳定的 PyTorch 实现](#3-稳定的-pytorch-实现)
- [4. 训练日志解读与常见问题修复](#4-训练日志解读与常见问题修复)
- [5. 从“文本–图像”推广到“全模态”](#5-从文本图像推广到全模态)
- [6. 进阶：更鲁棒的替代/增强](#6-进阶更鲁棒的替代增强)
- [7. 默认超参与工程建议](#7-默认超参与工程建议)
- [8. 参考与术语](#8-参考与术语)

---

## 1. SDM 是什么

给定一个 batch 的**查询侧**特征（文本/红外/素描/彩铅或其融合）与**图库侧**RGB特征，计算它们的余弦相似度矩阵，并将每一行（或列）通过带温度的 softmax 得到**预测分布** \( \mathbf{p} \)。根据身份标签构造**真实分布** \( \mathbf{q} \)（同一行内的所有正样本均分概率）。使用**双向 KL** 匹配分布：行方向 \((i \rightarrow t)\) 与列方向 \((t \rightarrow i)\) 的 KL 相加作为 SDM 损失。

- 余弦相似：对特征 \( \mathbf{f} \) 先做 \( \ell_2 \) 归一化：  
  \[
  \tilde{\mathbf{f}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2 + \epsilon}
  \]
- 相似度矩阵：
  \[
  \mathbf{S}_{i,j} = \tilde{\mathbf{f}}^{\,\text{qry}}_i \cdot \tilde{\mathbf{f}}^{\,\text{rgb}}_j
  \]
- 带温度 \( \tau \) 的 softmax（行方向）：
  \[
  p^{i2t}_{i,j} = \frac{\exp\!\left(\mathbf{S}_{i,j}/\tau\right)}
                        {\sum_{k}\exp\!\left(\mathbf{S}_{i,k}/\tau\right)}
  \]
- 标签分布（行内均分正样本概率）：
  \[
  q^{i2t}_{i,j} = \frac{y_{i,j}}{\sum_k y_{i,k}} \quad (\text{若}\ \sum_k y_{i,k}=0\ \text{则该行不参与损失})
  \]
- KL 损失（行方向）：
  \[
  \mathcal{L}_{i2t}=\frac{1}{N_{\text{valid}}}\sum_{i\in \mathcal{I}_{\text{valid}}}\sum_j
  p^{i2t}_{i,j}\left(\log p^{i2t}_{i,j}-\log \left(q^{i2t}_{i,j}+\epsilon\right)\right)
  \]
- 列方向 \((t\rightarrow i)\) 类同，最终：
  \[
  \mathcal{L}_{\text{SDM}}=\mathcal{L}_{i2t}+\mathcal{L}_{t2i}
  \]

> 工程要点：\( \tau \) 越小分布越“尖”，易不稳定；一般 \( \tau \in [0.07,\,0.2] \)。所有 \(\log\) 与归一化均需 \(\epsilon\) 以防数值问题。

---

## 2. 计算步骤（多模态可直接套用)

设**查询侧融合特征** \( F^{\text{qry}}\in \mathbb{R}^{N\times D} \)，**图库 RGB 特征** \( F^{\text{rgb}}\in \mathbb{R}^{M\times D} \)，**同身份指示矩阵** \( \mathbf{y}\in\{0,1\}^{N\times M} \)。

1. **归一化**：  
   \[
   \hat{F}^{\text{qry}}=\text{norm}(F^{\text{qry}}),\quad
   \hat{F}^{\text{rgb}}=\text{norm}(F^{\text{rgb}})
   \]
2. **相似度矩阵**：  
   \[
   \mathbf{S}=\hat{F}^{\text{qry}}\hat{F}^{\text{rgb}^\top}
   \]
3. **log-softmax**（更稳）：  
   \[
   \log \mathbf{P}_{i2t}=\log\text{softmax}(\mathbf{S}/\tau,\ \text{rowwise})
   \]
   \[
   \log \mathbf{P}_{t2i}=\log\text{softmax}(\mathbf{S}^\top/\tau,\ \text{rowwise})
   \]
4. **标签分布** \( \mathbf{q} \)（行/列分别归一化；对和为 0 的行/列做 mask）：  
   \[
   \mathbf{q}_{i2t}=\frac{\mathbf{y}}{\sum_{k}\mathbf{y}_{:,k}},\quad
   \mathbf{q}_{t2i}=\frac{\mathbf{y}^\top}{\sum_{k}\mathbf{y}^\top_{:,k}}
   \]
5. **KL(p‖q) 的稳定计算**：用 \(\log \mathbf{P}\) 还原 \(\mathbf{P}=\exp(\log\mathbf{P})\)，使用 \(\log(\mathbf{q}+\epsilon)\) 以避免 \(\log 0\)。

---

## 3. 稳定的 PyTorch 实现

```python
import torch
import torch.nn.functional as F

def sdm_loss(qry, gal, y, tau=0.1, eps=1e-6):
    """
    qry: [N, D]  任意模态或多模态融合后的查询特征
    gal: [M, D]  RGB 图库特征
    y:   [N, M]  二值同身份矩阵（同 ID = 1）
    """
    # 1) L2 归一化（避免全零向量除零）
    qry = F.normalize(qry, dim=1, eps=eps)
    gal = F.normalize(gal, dim=1, eps=eps)

    # 2) 相似度矩阵
    S = qry @ gal.t()  # [N, M]

    # 3) 行方向 (i2t)
    logP_i2t = F.log_softmax(S / tau, dim=1)  # [N, M]
    P_i2t = logP_i2t.exp()

    q_i2t_sum = y.sum(dim=1, keepdim=True)
    valid_i = (q_i2t_sum.squeeze(-1) > 0)
    q_i2t = torch.zeros_like(y, dtype=S.dtype)
    q_i2t[valid_i] = y[valid_i] / q_i2t_sum[valid_i]

    logQ_i2t = torch.log(q_i2t.clamp_min(eps))
    KL_i2t = (P_i2t * (logP_i2t - logQ_i2t)).sum(dim=1)
    L_i2t = KL_i2t[valid_i].mean() if valid_i.any() else S.new_tensor(0.)

    # 4) 列方向 (t2i)
    logP_t2i = F.log_softmax(S.t() / tau, dim=1)  # [M, N]
    P_t2i = logP_t2i.exp()

    y_t = y.t()
    q_t2i_sum = y_t.sum(dim=1, keepdim=True)
    valid_j = (q_t2i_sum.squeeze(-1) > 0)
    q_t2i = torch.zeros_like(y_t, dtype=S.dtype)
    q_t2i[valid_j] = y_t[valid_j] / q_t2i_sum[valid_j]

    logQ_t2i = torch.log(q_t2i.clamp_min(eps))
    KL_t2i = (P_t2i * (logP_t2i - logQ_t2i)).sum(dim=1)
    L_t2i = KL_t2i[valid_j].mean() if valid_j.any() else S.new_tensor(0.)

    return L_i2t + L_t2i
```

> 实践建议：SDM 与分类损失（CE/ID）联合：  
> \[
> \mathcal{L} = \mathcal{L}_{\text{ID}} + \lambda_{\text{sdm}}\mathcal{L}_{\text{SDM}} + \cdots
> \]
> 对新初始化的 BNNeck/分类头/融合模块，学习率应显著高于预训练主干。

---

## 4. 训练日志解读与常见问题修复

### 4.1 如何判断“学没学到”？
- 若分类类别数为 \(C\)，随机预测的**交叉熵**期望约为 \(\ln C\)。如果日志里长期 \( \text{CE}\approx \ln C \)，说明分类头没有有效学习（常见于学习率过低或未加入优化器）。

### 4.2 SDM 出现负值？
- SDM 本质是 **KL 损失**，应为**非负**。日志里出现 **SDM=-0.06** 多半是你打印了“相似度/分数”，而不是“损失值”。建议同时打印：
  - `sdm_loss.item()`（应 \(\ge 0\)）；
  - `p_pos_mean = P_i2t[y.bool()].mean()` 作为趋势分数（越大越好）。

### 4.3 偶发 NaN 的常见来源与修复
1. **无正样本行/列**：对应行/列的 \( \sum y = 0 \) 会导致 \( \log 0 \)。修复：**在损失中 mask** 无正样本行/列，并在采样器中保证跨模态的每个身份至少出现一次（P×K 或多模态平衡采样）。  
2. **温度太小或相似度过大**：\( \tau \) 太小会让分布过“尖”。建议起步 \( \tau=0.1 \) 并视稳定性在 \( [0.07,0.2] \) 调整。  
3. **未归一/全零特征**：必须 `F.normalize(..., eps)`；缺失模态的“零填充”应**先过可学习投影，再归一化**，避免全零向量。  
4. **AMP 溢出/不稳定算子**：相似度与 `log_softmax` 建议用 **fp32**；对所有中间量可 `torch.nan_to_num` 兜底。  
5. **梯度异常**：添加 `clip_grad_norm_(..., 5.0)`，并避免 `inplace` 破坏计算图。

### 4.4 让 CE 先“动”起来
- **参数组与学习率**：把 BNNeck/分类头/融合模块的 LR 提到 \(5\times 10^{-4}\sim 10^{-3}\)，预训练主干 \( \sim 10^{-5} \)。  
- **热身策略**：先**冻结主干** 1–3 个 epoch，仅用 CE 拉低损失（例如从 \(\ln C\) 降到 \(<5.5\)），再解冻联合训练。  
- **核对标签映射**：确保 `person_id -> [0, C-1]` 在数据与 DataLoader 中一致且不会每次重启被重置。

---

## 5. 从“文本–图像”推广到“全模态”

只需要把**查询侧特征**换成任意单模态或多模态的**融合向量**（如 IR/素描/彩铅/文本的融合），图库仍为 RGB。**指示矩阵** \( \mathbf{y} \) 依身份一致性构造（同 ID 为 1）。SDM 的双向 KL **公式不变**：对齐“**查询→图库**”与“**图库→查询**”的概率分布到标签分布。

---

## 6. 进阶：更鲁棒的替代/增强

- **BSDM（Bidirectional Symmetric Distribution Matching）**：在 SDM 的基础上加入对称项 \( \mathrm{KL}(\mathbf{q}\,\|\,\mathbf{p}) \)，即  
  \[
  \mathcal{L}_{\text{BSDM}}=\left[\mathrm{KL}(\mathbf{p}\,\|\,\mathbf{q})+\mathrm{KL}(\mathbf{q}\,\|\,\mathbf{p})\right]_{i2t}
  +\left[\mathrm{KL}(\mathbf{p}\,\|\,\mathbf{q})+\mathrm{KL}(\mathbf{q}\,\|\,\mathbf{p})\right]_{t2i}
  \]
  对**噪声标注**与难例更稳。

- **TAL（Triplet Alignment Loss）**：把三元组思想从“最难负样本”放宽到**全负样本的 log-exp 上界**：  
  \[
  \mathcal{L}_{\text{TAL}}=\frac{1}{N}\sum_i \log\!\left(1+\sum_{j\in \mathcal{N}(i)} \exp\!\big(\alpha(\Delta_{i,j}-m)\big)\right)
  \]
  其中 \( \Delta_{i,j} \) 为“正-负”的相似度差，\( m \) 为边距，\( \alpha \) 为温度。常与 SDM 联合，早期可抑制坍塌。

---

## 7. 默认超参与工程建议

- **温度**：\( \tau=0.1 \)（稳健起步，后续微调到 \(0.07\sim 0.2\)）。  
- **权重**：\( \lambda_{\text{sdm}}=1.0 \) 起步；当 CE 已有明显下降后再适度上调。  
- **采样器**：跨模态 **P×K**，保证每个身份在查询与图库两侧都有正样本（防止空行/列）。  
- **数值稳定**：`eps=1e-6`、梯度裁剪 `5.0`、相似度与 `log_softmax` 在 fp32。  
- **日志**：同时打印 `sdm_loss`、`p_pos_mean`、`tau`、`feat_norm`、`logit_std`，避免把分数当损失误判趋势。

---

## 8. 参考与术语

- **SDM / Distribution Matching**：通过 KL 匹配图—文（或多模态—RGB）的相似度分布与标签分布。  
- **双向（i2t / t2i）**：既从查询到图库，也从图库到查询做分布匹配。  
- **温度 \( \tau \)**：控制 softmax 的陡峭程度，过小易不稳。  
- **BSDM / TAL**：对 SDM 的鲁棒增强，可在噪声或难例较多时改善稳定性与泛化。

> 建议配合你的训练脚本：
> 1) 为新层单独参数组并提升 LR；2) 先冻结主干热身 CE；3) 启用上面的 `sdm_loss` 实现与采样器；4) 持续监控并清零 NaN。

