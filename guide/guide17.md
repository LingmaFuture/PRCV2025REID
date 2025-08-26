好消息：你的 `debug_modality.py` 已经把核心症状暴露出来了——**每个样本被“推断的模态”恒等于 `{'text'}`**，因此全数据集统计只有 `text: 18420`，`vis+非vis` 组合为 0%。这说明\*\*`infer_modalities_of_sample` 的实现只认文本、不认图像\*\*（忽略了 `images` / `modality_mask`），导致之后的强配对/采样器都以为“没有 RGB↔非RGB 的可配对样本”，训练自然配不起来。

下面给你最小补丁，直接把“模态推断”修正为：优先用 `modality_mask`，其次用 `images`，最后才把 `text_description` 作为可选补充。并做统一命名（vis→rgb, nir→ir, sk→sketch, cp/ccpencil→cp）。

---

## 🔧 修复点 1：重写 `infer_modalities_of_sample`（关键）

**文件**：`datasets/dataset.py`

```python
# 放在文件顶部（或你的常量区）
MOD_CANON = {
    'vis':'rgb', 'rgb':'rgb',
    'nir':'ir',  'ir':'ir',
    'sk':'sketch','sketch':'sketch',
    'cp':'cp','cpencil':'cp',
    'txt':'text','text':'text'
}

def _canon(name: str) -> str:
    return MOD_CANON.get(str(name).lower().strip(), str(name).lower().strip())

@torch.no_grad()
def infer_modalities_of_sample(dataset, idx, *, include_text=True, prefer_mask=True):
    """
    统一、稳健地推断样本可用模态：
    1) 优先 modality_mask（>0.5 视为可用）
    2) 其次 images（非空张量）
    3) 可选 text（text_description 非空）
    返回：{'rgb','ir','sketch','cp', ['text']}
    """
    # 尽量不触发昂贵的图像加载，优先用 dataset 的元信息结构；否则退回 dataset[idx]
    s = dataset.samples[idx] if hasattr(dataset, "samples") else dataset[idx]

    mods = set()

    # 1) modality_mask（推荐）
    if prefer_mask and isinstance(s.get('modality_mask', None), dict):
        for k, v in s['modality_mask'].items():
            try:
                vf = float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
            except Exception:
                vf = 0.0
            if vf > 0.5:
                mods.add(_canon(k))

    # 2) images（作为兜底或补充）
    if ('images' in s) and isinstance(s['images'], dict):
        for k, img in s['images'].items():
            if torch.is_tensor(img) and img.numel() > 0:
                # 粗检非空：避免全零张量
                try:
                    if img.dtype.is_floating_point:
                        non_empty = bool((img.abs() > 1e-6).any())
                    else:
                        non_empty = True
                except Exception:
                    non_empty = True
                if non_empty:
                    mods.add(_canon(k))

    # 3) text（可选）
    if include_text:
        td = s.get('text_description', None)
        has_text = (isinstance(td, str) and len(td.strip()) > 0) or \
                   (hasattr(td, '__len__') and len(td) > 0)
        if has_text:
            mods.add('text')

    return mods
```

> 解释：
>
> * 你当前样本里 `modality_mask` 明明显示了 `vis/nir/sk/cp`（甚至都是 1.0），但“推断的模态”却只有 `text`，说明之前函数**没有把 mask/images 纳入判定**。
> * 上面补丁会先看 `mask`，再看 `images`，最后（可选）把文本加入，且做了**命名规范化**以和训练/采样的 `rgb/ir/sketch/cp/text` 完全对齐。

---

## 🔧 修复点 2：调试脚本也要“看图像模态为主”

把 `debug_modality.py` 里统计部分改为分别统计**仅图像模态**与**包含文本的模态**，你就能明确看到“配对覆盖”的真实面貌。

```python
# 统计（图像为主）
img_mods = infer_modalities_of_sample(full_dataset, i, include_text=False)
all_mods = infer_modalities_of_sample(full_dataset, i, include_text=True)

# vis+非vis（图像角度）：rgb + {ir,sketch,cp} 是否共存
has_rgb = 'rgb' in img_mods
has_nonrgb = any(m in img_mods for m in ['ir','sketch','cp'])
if has_rgb and has_nonrgb:
    vis_nonvis_pairs += 1

# …分别累计 img_mods / all_mods 的直方统计，打印两组结果
```

期望输出会从：

```
模态出现次数:
  text: 18420
有vis+非vis组合的样本数: 0
比例: 0.0%
```

变成类似（示例）：

```
[Image-only] 模态出现次数:
  rgb: 17xxx
  ir:  12xxx
  sketch: 18xxx
  cp: 17xxx
有 rgb+非rgb 组合的样本数: xxxx
比例: 85.3%
[Image+Text] 还会多出 text: 18420
```

---

## 🔧 修复点 3：采样器用“规范名”做配对判定

在 `ModalAwarePKSampler_Strict` 里，构建 `pairable_ids` 时请以**规范名**判定：

```python
# 对每个 pid 汇总模态（来自 dataset 的 mask 或 meta，别逐个加载图片）
mods = infer_modalities_of_sample(dataset, sample_idx, include_text=False)  # 图像模态
# 或者自己按 mask 聚合并 _canon()

has_rgb = 'rgb' in mods
has_nonrgb = any(m in mods for m in ['ir','sketch','cp'])  # 仅图像非RGB；若要把 text 也算非RGB，可另设开关
if has_rgb and has_nonrgb:
    pairable_ids.add(pid)
```

> 这样你的“强配对采样器”就不会再被“恒等于 {'text'}”误导，`pair_coverage_mavg` 也会回到合理区间（预期 ≥0.85）。

---

## ✅ 快速验收清单（5 分钟）

1. 替换 `infer_modalities_of_sample`，保存。
2. 运行 `python debug_modality.py`，确认：

   * “推断的模态”不再恒等于 `{'text'}`；
   * `vis+非vis`（现在是 `rgb+非rgb`）组合比例显著 > 0%。
3. 启动训练前，打印一次采样器的 `pairable_ids` 数量（或比例），应远大于 0。
4. 训练跑前 200 步，观察：

   * `pair_coverage_mavg` 从 \~0.7 提升到 ≥0.85；
   * SDM 不再 0.000，Top-1 不再恒 0%；
   * CE 不会卡在 \~5.99。

---

## 小结

你的“配不齐正样本”的根因其实是**模态推断函数的逻辑错误**——一直把“文本”当唯一模态。修好 `infer_modalities_of_sample` 后，采样器能正确看到 `rgb↔{ir,sketch,cp}` 的可配对结构，**强配对**与**pair-coverage**才会真的发挥作用，训练曲线才正常。



完全可以按数据集的命名来！
你现在看到的 `rgb/ir/sketch/cp/text` 只是我为了“通用可读性”给出的**内部规范名**，方便区分“可见光(RGB)”和“近红外(IR)”。但 ORBench（你这套数据）本来就用的是：

* `vis`（可见光）
* `nir`（近红外）
* `sk`（素描）
* `cp`（彩铅）
* `text`（文本）

如果你更希望**端到端都用数据集原生名**，完全 OK，关键是“一处统一，处处一致”。下面给你最小改动方案。

---

## 推荐做法：统一到数据集原生名（vis/nir/sk/cp/text）

### 1) 定一个“唯一真理表”（utils.modal.py）

```python
# 目标：统一到数据集原生名
CANON_DS = {
    'vis':'vis', 'rgb':'vis',
    'nir':'nir', 'ir':'nir',
    'sk':'sk', 'sketch':'sk',
    'cp':'cp', 'cpencil':'cp', 'ccpencil':'cp',
    'txt':'text', 'text':'text'
}

def canon_mod(name: str) -> str:
    return CANON_DS.get(str(name).lower().strip(), str(name).lower().strip())
```

> 这一步把任何历史/外来命名（如 rgb/ir/sketch/cpencil）**规范化为数据集名**。

### 2) 修改模态推断与采样逻辑，**全部用 vis/nir/sk/cp/text**

* `infer_modalities_of_sample(...)` 返回集合只包含 `{'vis','nir','sk','cp'[,'text']}`。
* 采样器“强配对”判断：

  ```python
  has_vis    = 'vis' in mods
  has_nonvis = any(m in mods for m in ['nir','sk','cp'])
  if has_vis and has_nonvis: pairable_ids.add(pid)
  ```

### 3) collate/batch 中的字段也用原生名

* `batch['modality']` 里每个元素都走 `canon_mod()` 之后再存。
* 你打印/调试时显示的也会是 `vis/nir/sk/cp/text`。

### 4) 评测白名单也用原生名

你说“只评四个单模态 + 四模态”，那就：

```python
eval_include_patterns = [
  "single/nir", "single/sk", "single/cp", "single/text",
  "quad/nir+sk+cp+text"
]
```

> 注意：**gallery 固定是 vis**，所以单模态查询只需要非 vis 的四个；四模态就是 `nir+sk+cp+text`。

### 5) 统一名称规范化函数（可选）

如果你的 `query_loaders` 里有历史名称混用，做一个“规范化”再匹配白名单：

```python
def canon_query_name(name: str) -> str:
    # 例如 "single/sketch_v1" → "single/sk"
    name = name.strip().lower()
    if "/" not in name: return name
    head, tail = name.split("/", 1)
    mods = [canon_mod(m) for m in tail.replace("_v1","").split("+")]
    return f"{head}/" + "+".join(mods)
```

---

## 一键替换清单（把 rgb/ir/sketch/cpencil 换成 vis/nir/sk/cp）

1. `infer_modalities_of_sample` / 任何模态来源统一经过 `canon_mod`。
2. 强配对采样器用 `vis`/`nir`/`sk`/`cp` 判断。
3. `compatible_collate_fn` 注入的 `batch['modality']` 也是 `vis/nir/sk/cp/text`。
4. 评测白名单、日志打印、健康线统计（如 `pair_coverage_mavg`）的模态名，全部用 `vis/nir/sk/cp/text`。
5. 如果你做了**特征缓存**（磁盘/内存），**改名后要清缓存**，避免 “旧命名” 的特征拿来复用。

---

## 为什么我先前用 rgb/ir/sketch？

* 这是很多跨模态 ReID/多模态论文常用的“通用名”，跨数据集时不容易混淆。
* 你这套项目现在**只面向 ORBench**，直接用数据集原生名更顺手，避免“vis↔rgb”心智负担。

---

## 小提示

* 评测里 **gallery 永远是 vis**，不用把 `single/vis` 放入白名单；
* 训练/采样里的“非 vis”就只看 `nir/sk/cp`（`text` 只在跨模态检索时当 query）；
* 名称统一后，你之前的 `debug_modality.py` 统计会更直观：不要再看到“推断模态恒等于 {text}”。

---

需要的话我可以把你现有 `infer_modalities_of_sample` / 采样器 / 评测过滤的具体 patch（基于 vis/nir/sk/cp/text）按文件与行号给到你，直接粘贴替换就行。
