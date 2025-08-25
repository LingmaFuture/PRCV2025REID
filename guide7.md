定位很清楚：现在是**调度器接口不匹配**——你的代码在 `train_epoch` 里调用了
`model.sdm_scheduler.get_weight(epoch)`，但当前 `SDMScheduler` 类**没有**这个方法，于是直接 `AttributeError`。

给你两种“最快修复”，任选其一即可（我更推荐方案 A）。

---

## 方案 A（推荐）：给 `SDMScheduler` 补齐 `get_weight`（并做成可调用）

在 `models/sdm_scheduler.py` 里把类改成下面这个最小实现（兼容你的暖启动与列表日程）：

```python
# models/sdm_scheduler.py
from typing import List, Optional

class SDMScheduler:
    def __init__(
        self,
        warmup_epochs: int = 1,
        weight_schedule: Optional[List[float]] = None,  # 例如 [0.1, 0.3, 0.5]
        final_weight: Optional[float] = None,           # 若未给就取 schedule 的最后一个
        mode: str = "stair",                            # 预留: "stair" | "linear"
        start_weight: float = 0.0,
        end_epoch: Optional[int] = None,                # 线性模式用
    ):
        self.warmup_epochs = int(warmup_epochs)
        self.weight_schedule = weight_schedule or []
        self.final_weight = float(final_weight if final_weight is not None
                                  else (self.weight_schedule[-1] if self.weight_schedule else 0.0))
        self.mode = mode
        self.start_weight = float(start_weight)
        self.end_epoch = end_epoch

    def get_weight(self, epoch: int) -> float:
        """返回当前 epoch 的 SDM 权重"""
        if epoch <= self.warmup_epochs:
            return 0.0

        # 有离散 schedule 就按阶梯
        if len(self.weight_schedule) > 0:
            idx = max(0, min(epoch - self.warmup_epochs - 1, len(self.weight_schedule) - 1))
            return float(self.weight_schedule[idx])

        # 没给 schedule：按 final_weight 常量（或线性爬升）
        if self.mode == "linear" and self.end_epoch is not None:
            total = max(1, self.end_epoch - self.warmup_epochs)
            t = max(0, min(epoch - self.warmup_epochs, total))
            alpha = t / total
            return self.start_weight + alpha * (self.final_weight - self.start_weight)

        return self.final_weight

    # 让实例可直接调用：scheduler(epoch) 等价于 get_weight(epoch)
    __call__ = get_weight
```

> 如果你在 `model.__init__` 里实例化它，确保把 `warmup_epochs` 和 `weight_schedule` 从配置传进去。`final_weight` 不传也行，会默认取 `schedule` 最后一个。

`train_epoch` 中原来的调用就不需要改动了：

```python
sdm_w = model.sdm_scheduler.get_weight(epoch)
```

---

## 方案 B（不动调度器）：在 `train_epoch` 里做“多态兜底”

如果你不想动 `SDMScheduler` 文件，只在 `train.py` 里包一层兼容：

```python
def _resolve_sdm_weight(scheduler, epoch):
    # 优先 get_weight
    if hasattr(scheduler, "get_weight"):
        return float(scheduler.get_weight(epoch))
    # 次选可调用
    if callable(scheduler):
        try:
            return float(scheduler(epoch))
        except TypeError:
            pass
    # 兜底取属性
    return float(getattr(scheduler, "weight", 0.0))

# train_epoch 中
sdm_w = _resolve_sdm_weight(model.sdm_scheduler, epoch)
```

这样无论你的调度器实现成 `get_weight`、`__call__` 还是暴露 `weight` 属性，都能跑。

---

## 小提醒（一起确认一下）

* 你的配置里已经写了：

  * `sdm_weight_warmup_epochs = 2`
  * `sdm_weight_schedule = [0.1, 0.3, 0.5]`
    记得把这两项**传给** `SDMScheduler(...)` 的构造函数。

* 开始一个新 epoch 时打印一下，确认数值正确：

  ```python
  if step == 0:
      print(f"[sdm] epoch={epoch} weight={sdm_w:.3f}")
  ```

* 仍然建议保留 CE-only 的第一轮（你已经做了），第二轮起按 0.1→0.3→0.5 平滑上来；若 CE 明显回弹（>2.5 且持续），就把权重回退一档，同时检查“批内配对覆盖率”。

---

## 立即可执行的最小改动清单

* （优先）把上面的 `SDMScheduler` 粘进 `models/sdm_scheduler.py`，保存。
* 重启训练脚本；看到 `[sdm] epoch=2 weight=0.100 use_sdm=True` 就说明调度器已接好。

有别的接口冲突/报错日志，直接贴上来，我按行给你补齐。
