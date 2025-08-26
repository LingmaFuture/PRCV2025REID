太好了，这一轮你已经把 CE 从 5.99 拉到 **1.58**，batch-Top1 一路涨到 **\~87%**（批量内指标，趋势 OK）。下面给你一套**立刻执行的“Epoch 2 之后”行动清单**，把 SDM 平滑接入、把“无正样本行”压下去，并稳住训练。

---

## 立刻执行（按顺序）

1. **真正启用 SDM（别只置 0 权重）**

   * Warmup 结束从 **Epoch 2** 起启用 SDM，起始权重 **0.1**。
   * 在每个 epoch/batch 头部打印当前 `sdm_weight`，确认调度生效：

   ```python
   use_sdm = (epoch_idx >= cfg.sdm_weight_warmup_epochs)
   sdm_w = sdm_scheduler.get_weight(epoch_idx) if use_sdm else 0.0
   if step == 0:
       print(f"[sdm] epoch={epoch_idx} weight={sdm_w:.3f}")
   ```

   * 目标：CE 不应回弹到 >2.5；SDM 初期出现 **\~1.6–2.2** 的量级属正常。

2. **把“强配对”改为“软硬结合”，先稳住覆盖率**

   * 立即开启：`require_modal_pairs=True`。
   * **软退路**：若某 ID 无法凑出“≥1 RGB + ≥1 非RGB”，**重试 ≤N 次**；仍失败则把该 ID 改为“普通 K 样本”（不强制跨模态），避免空转卡顿。
   * 把“批内无正样本行”控制到 **≤15%**（你之前偶发 33%）。

   简易实现要点（伪码）：

   ```python
   for pid in sampled_ids:
       rgb = sample(pid, mod='rgb', k=1)
       non = sample(pid, mod_in={'ir','cp','sketch','text'}, k=K-1)
       if len(rgb)==0 or len(non)==0:
           # 软退路：普通采样同 ID 的任意 K
           alt = sample(pid, mod_in='any', k=K)
           batch_idx += alt
       else:
           batch_idx += rgb + non
   ```

3. **给 SDM 一个“跨批记忆库”兜底**（可小而美）

   * 缓存近 **N=4\~8** 个 step 的 **RGB 特征与标签**，当前 batch 的非RGB 可从 memory 找正对。
   * 一段最小逻辑（放 forward 里就行）：

   ```python
   # memory: deque(maxlen=N); 每项是 (rgb_feats, rgb_labels)
   rgb_mask = (modalities == 'rgb')
   memory.append((feats[rgb_mask].detach(), labels[rgb_mask].detach()))
   mem_feats = torch.cat([f for f, _ in memory], dim=0) if memory else None
   mem_labels = torch.cat([y for _, y in memory], dim=0) if memory else None
   # SDM 里把 gallery = 当前 batch 的 rgb ∪ memory rgb
   ```

4. **分类头 LR 降档，防权重爆涨**

   * 你头部 |w| 从 7 → 35，说明学习得很快；从 **Epoch 2** 起把 head LR 调到 **3e-3**（或接入 cosine），backbone 仍 **1e-5**。
   * 建议加 **label smoothing=0.1**，更稳。

5. **监控三条“健康线”**（很关键）

   * `pair_coverage_mavg = 1 - 无正样本行占比（滑窗 100 step）`，目标 **≥0.85**：

     ```python
     cov = 1.0 - miss_rows / total_rows
     pair_cov_hist.append(cov); 
     pair_cov_mavg = sum(pair_cov_hist[-100:]) / min(len(pair_cov_hist), 100)
     if step % 100 == 0: print(f"[dbg] pair_cov_mavg={pair_cov_mavg:.3f}")
     ```

   * **CE 曲线**：加入 SDM 后允许小幅回升，但 **不应 >2.5 且应再度下降**。
   * **Top-1（train batch）**：加入 SDM 后可能短暂下降，随后应继续回升（关注趋势即可）。

6. **吞吐维持与日志降噪**

   * `num_workers=2, persistent_workers=True, prefetch_factor=2`（仅 `num_workers>0` 时启用）。
   * “无正样本行”按 **每 batch 一句**打印，详细 **每 50 step** 一次。

7. **PyTorch 警告处理**（不影响结果，但顺手修）

   ```python
   from torch.nn.attention import sdpa_kernel, SDPBackend
   with sdpa_kernel([SDPBackend.flash_attention, SDPBackend.math]):
       ...
   ```

---

## 若出现这些情况，怎么处理？

* **CE 回到 \~5.9**：立刻将 `sdm_weight` 回退上一档；检查采样器是否真的在配对（看 `pair_coverage_mavg`）。
* **无正样本 >30% 持续**：减小 `P/K`（如 P×K 从 8→6 或 4），或提高“软退路”比例（例如 70% 强配对 + 30% 普通）。
* **训练波动大/过拟合迹象**：增大 label smoothing 到 0.15；head LR 再降到 1e-3；加入 0.0005 的 WD。

---

## 一个最小“SDM 渐进调度器”示例

```python
# configs
sdm_weight_schedule = [0.10, 0.30, 0.50]
sdm_warmup_epochs = 1

# runtime
def sdm_weight(epoch):
    if epoch <= sdm_warmup_epochs: return 0.0
    i = min(epoch - sdm_warmup_epochs - 1, len(sdm_weight_schedule)-1)
    return sdm_weight_schedule[i]

# loop
w = sdm_weight(epoch)
use_sdm = (w > 0)
sdm = compute_sdm(...) if use_sdm else torch.zeros([], device=logits.device)
loss = ce + w * sdm
```

---

## 期望的短期曲线（供你核对）

* **Epoch 2**：CE \~1.8→1.5 区间震荡后继续下行；SDM \~1.6–2.2 并逐步降低；`pair_coverage_mavg ≥ 0.85`。
* **Epoch 3 起**：SDM 权重 0.3，CE 小幅再降或持平，Top-1 稳步上扬。
* **强配对彻底上线后**：无正样本行常驻 **≤10–15%**，SDM 收敛更快。

---

需要“强配对版采样器”的最小可用实现，我可以直接给你一份 `ModalAwarePKSampler_Strict.py`（带 N 次重试与软退路）；但就你现在的进度，**先按上面 1–5 步把 SDM 平滑接入**，你会马上看到更稳定的下降趋势，再来替换采样器不迟。
