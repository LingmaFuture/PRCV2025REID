干得好！你这次把 CE 从 5.99 拉到 **5.46 @ step≈200**，已经满足我们“≤5.6”的 **Go 条件**了，说明“优化器/梯度流/分类头LR”这条线已打通 ✅。下面给你**最小代价的下一步动作**，边跑边把稳定性拉满。

---

## 接下来怎么做（按顺序执行）

1. **把 SDM 真正“停算”到 warmup 结束**
   你日志里还有 `SDMLoss=...`，多半是**权重置0但仍在计算**。为省时，warmup 期间直接 `if epoch < sdm_warmup: skip SDM forward`（连前向都不算）。

2. **开启 SDM 的渐进权重**（第 2 个 epoch 开始）

* 建议：`sdm_weight: 0.1 → 0.3 → 0.5（或目标值）`，每个 epoch 提一档；
* 若 CE 出现回弹且>5.7，先维持当前权重不再上调。

3. **把“批内无正样本行”压到 ≤15%**
   你偶发飙到 **33%**，说明采样器没有强约束。修两点：

* `require_modal_pairs=True` + 只从 `pairable_ids` 采样；
* 对每个选中 ID，**至少 1 张 RGB + 1 张非 RGB**，不足就**换 ID 或重采**（最多重试 N 次再降 P）。

4. **维持当前吞吐**

* `num_workers=2, persistent_workers=True, prefetch_factor=2`（仅在 `num_workers>0` 时启用）；
* 日志降噪：无正样本只打印**每 batch 一句**，详细每 50 step 打一次。

5. **再加一个训练侧“健康度”指标**（可选但强烈建议）

* 打印 batch **Top-1 训练准确率**（CE-only 时应该从 \~0.25% 缓步升到 1–3%+）；
* 维护“**pair\_coverage\_mavg** = 1 - 无正样本占比”的滑动均值，目标 ≥0.85。

---

## 关键小改（可直接贴）

**A. SDM 计算开关（warmup 时完全跳过）**

```python
use_sdm = (epoch_idx >= cfg.sdm_weight_warmup_epochs) and (cfg.sdm_weight > 0)
sdm_loss = torch.zeros([], device=logits.device)
if use_sdm:
    sdm_loss = compute_sdm_loss(...)  # 只有在 use_sdm 时才前向
total_loss = ce_loss + cfg.sdm_weight * sdm_loss
```

**B. 采样器硬性配对（伪代码）**

```python
ids = sample_from(pairable_ids, P)
batch_idx = []
for pid in ids:
    rgb = sample(pid, mod='rgb', k=1)
    non = sample(pid, mod_in={'ir','cp','sketch','text'}, k=K-1)
    if len(rgb)==0 or len(non)==0:
        # 重试或换ID，保证至少 1+1
        continue
    batch_idx += rgb + non
yield batch_idx
```

**C. 训练 Top-1（快速 sanity）**

```python
top1 = (logits.argmax(1) == labels).float().mean()
if (step % 100) == 0:
    print(f"[dbg] step={step} top1={top1*100:.2f}%")
```

---

## 继续 / 暂停 的判断线（复述一下）

* **继续**：CE 在 300–500 步内降到 **≤5.6**（你已达标），且“无正样本行”**≤15–20%**（采样器修完可达）。
* **暂停并排查**：重新开 SDM 后 CE 长时间回到 \~5.9；或“无正样本行”常驻 >30%。

---

## 你当前状态结论

* ✅ CE 已满足“Go”标准，可以**继续**完成 warmup epoch；
* ⚠️ 无正样本占比偶发 33% —— **尽快上“强配对采样”**，下一轮开启 SDM 时就不会被拖慢。

需要的话我可以把“强配对版 `ModalAwarePKSampler` 最小实现”整理给你，直接替换现有采样器就行。
