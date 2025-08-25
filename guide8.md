定位：这是个**作用域命名错误**。你在 `train_epoch` 里用到了 `config`（如 `pair_coverage_window = getattr(config, 'pair_coverage_window', 100)`），但函数作用域里并没有 `config` 这个变量；你的工程里统一用的是 `cfg`。

下面给你**最小可行修复**，两行改动+两处替换就能跑。

---

## ✅ 首选修复（推荐）

### 1) 给 `train_epoch` 传入 `cfg`

**调用处（`train_multimodal_reid`）**把这行：

```python
train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip, accum_steps, autocast_dtype)
```

改为：

```python
train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip, accum_steps, autocast_dtype, cfg)
```

### 2) 修改 `train_epoch` 函数签名

把：

```python
def train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip, accum_steps, autocast_dtype):
```

改为：

```python
def train_epoch(model, train_loader, optimizer, device, epoch, scaler, adaptive_clip, accum_steps, autocast_dtype, cfg):
```

### 3) 将函数体内所有 `config` 改为 `cfg`

例如把这两处改掉（保持默认值防止缺配置时报错）：

```python
pair_coverage_window = getattr(cfg, 'pair_coverage_window', 100)
pair_coverage_target = getattr(cfg, 'pair_coverage_target', 0.85)
```

如果有别处也写成了 `config.xxx`，同理替换为 `cfg.xxx`。

> 这一版不改变你的整体结构，也不会引入全局变量副作用，是最稳的修法。

---

## 🩹 备选兜底（不改函数签名）

如果你**不想改函数签名**，可在 `train_epoch` 开头加一段“就地解析 cfg”的兜底：

```python
# at top of train_epoch
_local_cfg = getattr(model, 'config', None)
if _local_cfg is None:
    try:
        _local_cfg = cfg  # 若外层有全局/闭包变量
    except NameError:
        class _D: pass
        _local_cfg = _D()
setattr(_local_cfg, 'pair_coverage_window', getattr(_local_cfg, 'pair_coverage_window', 100))
setattr(_local_cfg, 'pair_coverage_target', getattr(_local_cfg, 'pair_coverage_target', 0.85))

# 后续统一用 _local_cfg
pair_coverage_window = _local_cfg.pair_coverage_window
pair_coverage_target = _local_cfg.pair_coverage_target
```

但这招可读性差、容易埋雷，不如首选方案干净。

---

## 顺手自检（避免同类坑再现）

- 搜索一遍：`grep -R "config\." train.py models/ datasets/`  
  把训练/验证/测试循环里误用的 `config.` 全部替换成 `cfg.` 或 `model.config`（如果你在模型里挂了配置）。

- 监控初始化处：  

  ```python
  pair_cov_hist = deque(maxlen=getattr(cfg, 'pair_coverage_window', 100))
  ```

  确保这里也用 `cfg`，并在 Windows 环境下 `from collections import deque` 已导入。

---

## 预期结果

改完后再跑，`NameError: name 'config' is not defined` 会消失；训练将继续执行到你设置的 warmup 与 SDM 权重调度阶段。如果还有报错，把新日志贴上来，我按行给你补齐。
