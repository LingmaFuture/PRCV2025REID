因为你在“训练循环里”就触发了评测（`validate_competition_style`），而不是把评测放在每个 epoch 结束才跑，所以第 80/1863 个 batch 就开始了“提取特征”。从你日志能看出两个关键点：

* `[epoch 1] steps_run=80/1863 (max_steps=0)` 紧接着就打印了 `[EVAL] ...`，说明评测触发条件是“按步数间隔”或“首次小样本 sanity-check”，而不是“按 epoch 结束”。
* 这是本轮第一次评测，gallery 缓存还没有命中，于是先跑了 `_extract_feats_and_ids`，你就看到了 `提取特征: 21%` 的进度条；后续评测会快很多（命中缓存后几乎为 0 开销）。

下面是最常见的三种触发源头，以及一行就能“只在 epoch 末评测”的改法：

### 你现在很可能命中的三种触发

1. 训练循环里有“按步数评测”的条件

```python
if (batch_idx + 1) % cfg.eval_every_n_steps == 0:
    validate_competition_style(...)
```

2. 有“首轮快速体检”的条件

```python
if epoch == 0 and (batch_idx + 1) >= cfg.eval_after_steps:  # 比如80步
    validate_competition_style(...)
```

3. 有“按时间间隔评测”的条件

```python
if time.time() - last_eval_time > cfg.eval_every_minutes * 60:
    validate_competition_style(...)
```

### 想改成“只在每个 epoch 结束评测”的最小改动

把评测调用挪到 epoch 末尾，或加上严格的触发条件：

```python
# —— 训练循环（简化示意）——
for epoch in range(start_epoch, max_epoch):
    processed = 0
    for batch_idx, batch in enumerate(train_loader):
        # ... 正常训练 ...
        processed += 1

        # ❌ 注释掉这些“步数/时间触发”的评测
        # if (batch_idx + 1) % cfg.eval_every_n_steps == 0: ...
        # if epoch == 0 and (batch_idx + 1) == cfg.eval_after_steps: ...
        # if time.time() - last_eval_time > cfg.eval_every_minutes * 60: ...

    # ✅ 只在 epoch 结束时评测
    if getattr(cfg, "do_eval", True) and ((epoch + 1) % getattr(cfg, "eval_every_n_epochs", 1) == 1):
        comp_metrics = validate_competition_style(
            model, gallery_loader, query_loaders, device,
            k_map=100, sample_ratio=cfg.sample_ratio, cfg=cfg, epoch=epoch
        )
```

### 另外两个小提示

* 你已经加了 gallery 特征磁盘缓存（`.eval_cache/gallery_*.pkl`），第一次评测会慢是正常的；第二次开始会明显加速。
* 如果你仍想在首个 epoch 做一次“早期体检”，把步数阈值调大一些即可，例如 `cfg.eval_after_steps = 1000` 或直接改成“只在 batch\_idx + 1 == len(train\_loader)\` 时触发”。

**结论**：现在之所以“刚训练一点点就开始提特征”，是因为评测在 **batch 级别** 被触发了；把触发条件改到 **epoch 末** 或调大/关闭步数触发，就会按你预期运行。
