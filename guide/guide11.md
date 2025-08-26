你这两件事其实是**两个开关没关好**：

---

## 1) “每个 epoch 训练一点就结束”的常见元凶

从日志看：`Epoch 38: 4% | 80/1863` 然后直接跳到下一轮，基本可判定你启用了**步数上限**或**冒烟/调试截断**。去把下面这些“截断条件”全搜出来，**统一关掉**：

**在 `train.py` 里查（Win 下用 PowerShell）**

```powershell
Select-String -Path .\train.py -Pattern `
'max_steps_per_epoch|steps_per_epoch|debug_steps|smoke|fast_dev_run|limit_train|break after|return .*train_metrics' `
-AllMatches -CaseInsensitive
```

**把类似逻辑关掉/归零：**

```python
# 典型写法 1（建议设为 0 或 None）
max_steps = getattr(cfg, 'max_steps_per_epoch', 0)
...
for step, batch in enumerate(train_loader, start=1):
    ...
    if max_steps and step >= max_steps:   # ← 关掉
        break

# 典型写法 2（冒烟跑/调试）
smoke_steps = getattr(cfg, 'smoke_steps', 0)
if smoke_steps and step >= smoke_steps:   # ← 关掉
    break

# 典型写法 3（覆盖率不达标就提前 return）
if step >= X and pair_coverage_mavg < cfg.pair_coverage_target:   # ← 删除这类“早停”
    return metrics
```

> 改完后在每个 epoch 收尾打印：

```python
print(f"[epoch {epoch}] steps_run={step}/{len(train_loader)}")
```

你应该看到 `steps_run` 接近 `len(train_loader)=1863`，不再是几十步就结束。

---

## 2) “看不到 mAP”的常见元凶

要么**评测没被调用**，要么**评测频率太低**。确保三件事：

### A. 评测触发条件（每个 epoch 都评）

```python
eval_start_epoch   = getattr(cfg, 'eval_start_epoch', 1)
eval_every_n_epoch = getattr(cfg, 'eval_every_n_epoch', 1)

if epoch >= eval_start_epoch and ((epoch - eval_start_epoch) % eval_every_n_epoch == 0):
    gallery_loader, query_loaders = build_eval_loaders_by_rule(cfg)  # 你已有的函数
    eval_metrics = evaluate_retrieval(model, gallery_loader, query_loaders, device)
    # 统一打印
    print(
      "[EVAL] epoch=%d  mAP(all)=%.4f  |  mAP@single=%.4f  mAP@dual=%.4f  mAP@tri=%.4f  mAP@quad=%.4f"
      % (epoch,
         eval_metrics["mAP_mean"],
         eval_metrics["mAP_single"],
         eval_metrics["mAP_dual"],
         eval_metrics["mAP_tri"],
         eval_metrics["mAP_quad"])
    )
```

### B. 别让评测 DataLoader 被“0 workers + prefetch\_factor”卡死

你之前踩过这个坑，评测构建里务必：

```python
nw = getattr(cfg, "num_workers_eval", getattr(cfg, "num_workers", 0))
gallery_loader = DataLoader(...,
    num_workers=nw,
    pin_memory=True,
    collate_fn=compatible_collate_fn,
    persistent_workers=(nw > 0),
    prefetch_factor=(2 if nw > 0 else None)
)
# query_loaders 同理
```

### C. 结果一定要**打印**出来

有些实现只把 mAP 写到了日志/文件，不在控制台打。上面的 `print("[EVAL] ...")` 保证肉眼可见。

---

## 3) 顺手把这两个健康检查也加上（防再踩坑）

* **实际跑步数**：`steps_run`（见上）
* **评测是否真的跑了**：在 `evaluate_retrieval(...)` 开头打印一次：

  ```python
  print(f"[EVAL] gallery={len(gallery_loader.dataset)}  "
        f"queries={[ (k, len(v.dataset)) for k,v in query_loaders.items() ]}")
  ```

---

## 4) 你当前日志的两条提示

* `pair_coverage_mavg ≈ 0.698`：比目标 0.85 低，**SDM 能跑，但效率打折**。先把“步数截断”问题修好，再微调采样（提高 K 或降低 P）把覆盖率抬上去。
* `[sdm] epoch=1 weight=0.000 use_sdm=True`：warmup=1 后，从 **epoch=2** 起应看到 `weight=0.1`；确认你的调度器边界已按上次建议改为 `<` 或把 warmup 设为 1。

---

## 小结

* **原因 1**：启用了 `max_steps_per_epoch / smoke_steps / debug limit` 等截断 → 只跑几十步就换 epoch。
* **原因 2**：评测没触发或评测频率太大 → mAP 不打印。

按上面的**搜索→删除/归零截断→打开每轮评测**三步来，立刻就能看到完整 epoch 以及每轮 mAP。需要的话，把你找到的那几行“break/return”的片段贴上来，我给你精确改法。
