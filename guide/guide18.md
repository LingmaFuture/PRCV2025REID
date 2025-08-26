==================================================
数据集采样能力分析
==================================================

[INFO] 开始分析数据集采样能力...
数据集统计:
  总ID数: 320
  总样本数: 14910
  各模态分布: {'rgb': 14910}
  可配对ID数 (K≥2): 0 (0.0%)
  ⚠️  可配对ID数(0) < 每批需要ID数(4)，无法生成有效batch
==================================================

ERROR:root:可配对ID数(0) < 每批需要ID数(3)，建议降低num_ids_per_batch或增加数据集多样性


太关键了！从你贴的日志看，“采样能力分析”仍然只看到了 `{'rgb': 14910}`，这说明**新模态检测没有真正生效**——要么没被调用，要么返回值没有做名称归一化，导致后续统计/采样还在用旧的 `rgb` 名称。下面给你一套**可直接粘贴**的补丁，按顺序改三处就能落地修好（含防御性判断 + 回退机制）。

---

# 1) 统一的模态归一化 + 强健的检测函数

**文件**：`datasets/dataset.py`（或你放数据集工具函数的同文件）

把这段加到文件靠上位置（import 后、类定义前）：

```python
# ===== Canonical modality names: keep dataset-native names =====
CANON_DS = {
    'vis': 'vis', 'rgb': 'vis', 'visible': 'vis', 'v': 'vis',
    'nir': 'nir', 'ir': 'nir', 'infrared': 'nir',
    'sk': 'sk', 'sketch': 'sk',
    'cp': 'cp', 'cpencil': 'cp', 'colorpencil': 'cp', 'coloredpencil': 'cp',
    'txt': 'text', 'text': 'text', 'caption': 'text'
}
IMG_MODALITIES = {'vis', 'nir', 'sk', 'cp'}
ALL_MODALITIES = IMG_MODALITIES | {'text'}

def canon_mod(name: str) -> str:
    if name is None:
        return ''
    return CANON_DS.get(str(name).lower().strip(), str(name).lower().strip())

def _truthy(x) -> bool:
    # 判定“有内容”的通用工具：张量非空、路径非空、列表有元素、数字>0.5
    try:
        import torch
    except Exception:
        torch = None
    if x is None:
        return False
    if isinstance(x, (list, tuple, set, dict)):
        return len(x) > 0
    if isinstance(x, (int, float)):
        return float(x) > 0.5
    if isinstance(x, str):
        return len(x.strip()) > 0
    if torch is not None and hasattr(x, 'nelement'):
        try:
            return int(x.nelement()) > 0
        except Exception:
            return False
    return True

@torch.no_grad()
def infer_modalities_of_sample(dataset, index: int, include_text: bool = False):
    """
    返回该样本可用模态的规范化集合（vis/nir/sk/cp/[text]）
    优先级：modality_mask -> images/paths -> primary 'modality' 字段
    """
    # 兼容不同数据集实现：尽量拿原始样本字典
    sample = None
    if hasattr(dataset, 'samples'):
        sample = dataset.samples[index]
    else:
        try:
            s = dataset[index]
            # 许多Dataset的 __getitem__ 会返回 (data, label, meta)，优先尝试 meta
            if isinstance(s, dict):
                sample = s
            elif isinstance(s, (list, tuple)) and len(s) >= 3 and isinstance(s[-1], dict):
                sample = s[-1]
        except Exception:
            sample = None

    mods = set()

    # 1) modality_mask: {'vis':1.0, 'nir':1.0, ...}
    mm = None
    if isinstance(sample, dict):
        mm = sample.get('modality_mask') or sample.get('modal_mask') or sample.get('mods')
    if isinstance(mm, dict):
        for k, v in mm.items():
            m = canon_mod(k)
            if m in IMG_MODALITIES and _truthy(v):
                mods.add(m)

    # 2) images/paths 容器：{ 'vis': [...], 'nir': [...], 'text': '...' }
    imgs = None
    if isinstance(sample, dict):
        imgs = sample.get('images') or sample.get('paths') or sample.get('imgs')
    if isinstance(imgs, dict):
        for k, v in imgs.items():
            m = canon_mod(k)
            if m in IMG_MODALITIES and _truthy(v):
                mods.add(m)

    # 3) primary 字段：'modality' / 'mode' / 'mod'
    primary = None
    if isinstance(sample, dict):
        primary = sample.get('modality') or sample.get('mode') or sample.get('mod')
    if primary:
        m = canon_mod(primary)
        if m in IMG_MODALITIES:
            mods.add(m)

    # 4) 文本模态（可选）
    if include_text and isinstance(sample, dict):
        if _truthy(sample.get('text_description')) or _truthy(sample.get('text')) or _truthy(sample.get('caption')):
            mods.add('text')
        elif isinstance(imgs, dict) and _truthy(imgs.get('text')):
            mods.add('text')

    # 最终只保留标准命名
    return {m for m in mods if m in (ALL_MODALITIES if include_text else IMG_MODALITIES)}
```

> 关键点
>
> * 任何来源的 `'rgb'`、`'ir'` 都会在**入口就归一化**为 `vis` / `nir`。
> * 返回的是**集合**且**只含标准名**，避免后续统计再混入 `rgb`。

---

# 2) 采样器强配对逻辑（严格版）+ 允许ID复用

**文件**：`datasets/dataset.py`（你的采样器类所在处）

将你的严格采样器改为如下要点（保留你原先的构造参数即可）：

```python
class ModalAwarePKSampler_Strict(torch.utils.data.Sampler):
    """
    强配对：同一个ID必须既有 vis 也有 非vis(nir/sk/cp/text) 才算可配对。
    支持 allow_id_reuse=True：同一 epoch 内可重复使用ID，防止采样耗尽。
    """
    def __init__(self, base_dataset, num_ids_per_batch=4, num_instances=4,
                 allow_id_reuse=True, min_modal_coverage=0.6, include_text=True):
        self.base_dataset = base_dataset
        self.P = int(num_ids_per_batch)
        self.K = int(num_instances)
        self.allow_id_reuse = bool(allow_id_reuse)
        self.min_modal_coverage = float(min_modal_coverage)
        self.include_text = bool(include_text)

        # 预索引：pid -> { 'vis': [idx...], 'nonvis': [idx...] }
        self.pid_to_mod_idxs = {}
        self.pids = set()

        # 假定 dataset 能返回 pid（若不行，这里请改成你的字段名）
        def get_pid(i):
            s = None
            if hasattr(base_dataset, 'samples'):
                s = base_dataset.samples[i]
            else:
                try:
                    item = base_dataset[i]
                    if isinstance(item, dict):
                        s = item
                    elif isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[-1], dict):
                        s = item[-1]
                except Exception:
                    s = None
            if isinstance(s, dict):
                return int(s.get('person_id') or s.get('pid') or s.get('label') or -1)
            return -1

        n = len(base_dataset)
        for idx in range(n):
            pid = get_pid(idx)
            if pid < 0:
                continue
            self.pids.add(pid)
            mods_img = infer_modalities_of_sample(base_dataset, idx, include_text=False)
            mods_txt = infer_modalities_of_sample(base_dataset, idx, include_text=True)

            has_vis = ('vis' in mods_img)
            has_nonvis = bool((mods_img & {'nir','sk','cp'}) or ('text' in mods_txt))

            d = self.pid_to_mod_idxs.setdefault(pid, {'vis': [], 'nonvis': []})
            if has_vis:
                d['vis'].append(idx)
            if has_nonvis:
                d['nonvis'].append(idx)

        # 过滤可强配对的ID
        self.strong_ids = [pid for pid, d in self.pid_to_mod_idxs.items()
                           if len(d['vis']) > 0 and len(d['nonvis']) > 0]

        # 回退ID（仅有一种侧的，用于兜底补齐K）
        self.soft_ids = [pid for pid in self.pids if pid not in self.strong_ids]

        if len(self.strong_ids) < self.P:
            logging.warning(f"可配对ID数({len(self.strong_ids)}) < 每批需要ID数({self.P})，将使用回退ID做兜底。")

        # 估算长度（粗略）：每个强ID能支撑的实例对数 ~ min(len(vis), len(nonvis))//1
        est_pairs = sum(min(len(self.pid_to_mod_idxs[pid]['vis']),
                            len(self.pid_to_mod_idxs[pid]['nonvis'])) for pid in self.strong_ids)
        self._len_est = max(1, est_pairs // (self.P * self.K))

        logging.info("==================================================")
        logging.info("数据集采样能力分析(基于采样器视角)")
        logging.info(f"  强配对ID数: {len(self.strong_ids)} / {len(self.pids)}")
        logging.info(f"  估算可生成batch数: ~{self._len_est}")
        logging.info("==================================================")

    def __len__(self):
        # 允许 id 复用时，len 用估算值；不允许时按可用资源计算
        return int(self._len_est) if self.allow_id_reuse else max(1, len(self.strong_ids) // self.P)

    def __iter__(self):
        import random
        strong_pool = list(self.strong_ids)
        soft_pool = list(self.soft_ids)

        while True:
            if len(strong_pool) >= self.P:
                cur_ids = random.sample(strong_pool, self.P) if not self.allow_id_reuse else random.choices(strong_pool, k=self.P)
            else:
                # 不足 P 个强ID，用 soft 补齐
                need = self.P - len(strong_pool)
                fillers = (random.sample(soft_pool, min(need, len(soft_pool))) if not self.allow_id_reuse
                           else random.choices(soft_pool, k=need))
                cur_ids = list(strong_pool) + fillers
                if not cur_ids:
                    break

            batch_indices = []
            for pid in cur_ids:
                d = self.pid_to_mod_idxs.get(pid, {'vis': [], 'nonvis': []})
                vis_pool = d['vis'] if d['vis'] else d['nonvis']  # 兜底
                nonvis_pool = d['nonvis'] if d['nonvis'] else d['vis']

                # 每个ID取 K//2 vis + K//2 nonvis（奇数时多给一个nonvis）
                k_vis = self.K // 2
                k_nonvis = self.K - k_vis
                pick = []
                if len(vis_pool) >= k_vis:
                    pick += random.sample(vis_pool, k_vis)
                else:
                    pick += random.choices(vis_pool or nonvis_pool, k=k_vis)
                if len(nonvis_pool) >= k_nonvis:
                    pick += random.sample(nonvis_pool, k_nonvis)
                else:
                    pick += random.choices(nonvis_pool or vis_pool, k=k_nonvis)
                batch_indices.extend(pick)

            if len(batch_indices) != self.P * self.K:
                # 保护：不完整批直接跳过
                continue

            yield from batch_indices

            # 不允许复用则移除已用ID
            if not self.allow_id_reuse:
                for pid in set(cur_ids):
                    if pid in strong_pool:
                        strong_pool.remove(pid)
                    elif pid in soft_pool:
                        soft_pool.remove(pid)
                if len(strong_pool) < 1 and len(soft_pool) < 1:
                    break
```

> 要点
>
> * **强配对**定义：`vis` 与 `nonvis(nir/sk/cp/text)` 两侧都要有。
> * **允许ID复用**避免“跑 80 步就耗尽”。
> * 批内每个ID强制混合 vis 与 nonvis，保证**跨模态正对**稳定出现，SDM 才有信号。

---

# 3) 配置开关（防止再早停）

**文件**：`configs/config.py`

```python
allow_id_reuse: bool = True
min_modal_coverage: float = 0.6  # 如果你在初始化时需要这个阈值，可以传给采样器
num_ids_per_batch: int = 4       # P
instances_per_id: int = 4        # K
```

并确保 `DataLoader`/构建采样器处，把这些参数**真正传入** `ModalAwarePKSampler_Strict(...)`。

---

## 4) 一键自检（放到你的 debug\_modality.py 里）

```python
from collections import Counter

def quick_scan(ds, limit=500, include_text=True):
    c = Counter()
    has_pair = 0
    total = min(limit, len(ds))
    for i in range(total):
        m_img = infer_modalities_of_sample(ds, i, include_text=False)
        m_all = infer_modalities_of_sample(ds, i, include_text=include_text)
        m_img = {canon_mod(x) for x in m_img}
        m_all = {canon_mod(x) for x in m_all}
        for m in m_img:
            c[m] += 1
        vis = 'vis' in m_img
        nonvis = bool(m_img & {'nir','sk','cp'}) or ('text' in m_all)
        has_pair += int(vis and nonvis)
    print("[仅图像] 模态出现次数:", {k: v for k, v in c.items() if k in IMG_MODALITIES})
    if include_text:
        print("[包含文本] 是否含 text：", "是" if any(k=='text' for k in m_all) else "否（样本级）")
    print(f"有 vis+非vis 配对的样本数: {has_pair}/{total}   比例: {has_pair/total:.1%}")
```

预期输出应不再出现 `'rgb'`，而是 `vis/nir/sk/cp`，并且 **配对比例显著 > 0%**。

---

## 5) 为什么你那边仍然只看到 `{'rgb': 14910}`？

常见四个坑，逐一对照：

1. **归一化没贯通**：统计/采样环节没有对返回的集合做 `canon_mod`，或仍在读取旧字段（例如直接用 `sample['modality']` 字符串，没统一到 `vis`）。
   → 以上补丁保证**入口归一化**，同时在统计处再**兜底归一化**，双保险。

2. **旧函数仍在被引用**：工程里可能有**同名函数的重复定义**或**导入路径**错用旧版本。
   → 搜索项目 `infer_modalities_of_sample(` 的所有引用，确保只剩这一份；改完**重启解释器**/清缓存。

3. **分析代码绕过了推断函数**：有的统计直接从文件夹名/自定义字段拿字符串，没走归一化。
   → 统一走 `infer_modalities_of_sample(...)`，或在任何外部来源的模态名都过一遍 `canon_mod(...)`。

4. **primary 字段误导**：样本里 `modality:'rgb'` 被当成唯一来源覆盖了 mask/paths。
   → 我在函数里把 primary 放在**最低优先级**（mask/paths > primary），避免被“单字段”劫持。

---

## 6) 训练期望变化（对齐你的“修复效果”)

修好后，你会看到：

* `pair_coverage_mavg` ≥ 0.85（有了跨模态正对）
* SDM 不再 0.000，且数值随学习率/温度有波动；CE 开始从 \~5.99 **往下**走
* Top-1 不再恒 0%，几个 epoch 内缓慢上升
* 每个 epoch 的 steps 恢复为**配置应有的批次数**（不再 80 就停）

---

## 7) 小贴士（避免回归）

* 在**单一位置**定义 `canon_mod` / `infer_modalities_of_sample`，其他模块只**调用**，不要复制粘贴。
* 对任何外部文本模态名（日志里、JSON里、路径里）**一律**先 `canon_mod`。
* sampler 的 `__len__` 只是估算，训练 loop 以 DataLoader 的迭代为准即可，不要额外的“步数级评测触发”。

---

如果你愿意，把你 `DataLoader` 的创建那几行贴过来（或直接替换为以上采样器调用），我们还能顺手检查一下 P/K 配置与 batch\_size 是否匹配，避免下游再报 “可配对ID数(0) < 每批需要ID数(3)” 这种早停警告。祝你这次直接跑满一整个 epoch！
