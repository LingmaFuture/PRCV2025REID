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
Traceback (most recent call last):
  File "c:\important\source\PRCV2025REID\train.py", line 1850, in <module>
    train_multimodal_reid()
  File "c:\important\source\PRCV2025REID\train.py", line 1497, in train_multimodal_reid
    train_sampler = ModalAwarePKSampler_Strict(
TypeError: ModalAwarePKSampler_Strict.__init__() got an unexpected keyword argument 'batch_size'

好消息：问题已经被你“锁定”得很准——现在有三处在“漏油”：

1. 统计面板完全__绕过了新推断函数__（它还在读旧字段，直接把所有都记成 `rgb`）。
2. 采样器构造时__传了不支持的参数__（`batch_size`），导致直接抛错。
3. `Subset` → `BaseDataset` 的索引映射没处理，导致模态推断永远看不到真实键（`nir/sk/cp`）。

下面给一套一次性可落地的补丁：把它们贴进去，训练就能恢复“有配对、能跑满一个 epoch”。

---

# A. 统一规范化 + 健壮推断（放 `datasets/dataset.py` 顶部）

```python
import torch
from torch.utils.data import Sampler
import logging
from collections import Counter

# ===== 统一模态命名（只用数据集原生名）=====
CANON_DS = {
    'vis':'vis', 'rgb':'vis', 'visible':'vis', 'v':'vis',
    'nir':'nir', 'ir':'nir', 'infrared':'nir',
    'sk':'sk', 'sketch':'sk',
    'cp':'cp', 'cpencil':'cp', 'colorpencil':'cp', 'coloredpencil':'cp',
    'txt':'text', 'text':'text', 'caption':'text',
}
IMG_MODALITIES = {'vis','nir','sk','cp'}
ALL_MODALITIES = IMG_MODALITIES | {'text'}

def canon_mod(name: str) -> str:
    if name is None: return ''
    return CANON_DS.get(str(name).lower().strip(), str(name).lower().strip())

def _truthy(x) -> bool:
    if x is None: return False
    if isinstance(x, (list, tuple, set, dict)): return len(x) > 0
    if isinstance(x, (int, float)): return float(x) > 0.5
    if isinstance(x, str): return len(x.strip()) > 0
    if torch.is_tensor(x):
        try:
            ok = int(x.nelement()) > 0
            if ok and x.dtype.is_floating_point:  # 过滤全零张量
                return bool((x.abs().sum() > 1e-6).item())
            return ok
        except Exception:
            return False
    return True

@torch.no_grad()
def infer_modalities_of_sample(dataset, index: int, include_text: bool = True):
    """
    返回样本可用模态集合（vis/nir/sk/cp/[text]）
    优先级：modality_mask > images/paths > primary('modality'/'mode'/'mod')
    """
    # 兼容 Subset：优先拿原始样本字典
    sample = None
    if hasattr(dataset, 'samples'):
        sample = dataset.samples[index]
    else:
        try:
            s = dataset[index]
            if isinstance(s, dict):
                sample = s
            elif isinstance(s, (list, tuple)) and len(s) >= 3 and isinstance(s[-1], dict):
                sample = s[-1]  # (data, label, meta)
        except Exception:
            sample = None

    mods = set()

    # 1) modality_mask
    if isinstance(sample, dict):
        mm = sample.get('modality_mask') or sample.get('modal_mask') or sample.get('mods')
        if isinstance(mm, dict):
            for k, v in mm.items():
                m = canon_mod(k)
                if m in IMG_MODALITIES and _truthy(v):
                    mods.add(m)

    # 2) images / paths 容器
    if isinstance(sample, dict):
        imgs = sample.get('images') or sample.get('paths') or sample.get('imgs')
        if isinstance(imgs, dict):
            for k, v in imgs.items():
                m = canon_mod(k)
                if m in IMG_MODALITIES and _truthy(v):
                    mods.add(m)

    # 3) primary 字段（兜底、优先级最低）
    if isinstance(sample, dict):
        primary = sample.get('modality') or sample.get('mode') or sample.get('mod')
        if primary:
            m = canon_mod(primary)
            if m in IMG_MODALITIES:
                mods.add(m)

    # 4) 文本（可选）
    if include_text and isinstance(sample, dict):
        if _truthy(sample.get('text_description')) or _truthy(sample.get('text')) or _truthy(sample.get('caption')):
            mods.add('text')
        else:
            imgs = sample.get('images') or sample.get('paths') or sample.get('imgs')
            if isinstance(imgs, dict) and _truthy(imgs.get('text')):
                mods.add('text')

    return {m for m in mods if m in (ALL_MODALITIES if include_text else IMG_MODALITIES)}
```

---

# B. 替换“采样能力分析”（你的统计面板现状等于没走规范化）

把你现在打印 `{'rgb': 14910}` 的那段__全部替换__为下面函数，并在原地调用它：

```python
def analyze_sampling_capability(ds, limit=None):
    # 兼容 Subset
    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base_ds = ds.dataset
        indices = list(ds.indices)
    else:
        base_ds = ds
        indices = list(range(len(ds)))

    if limit:
        indices = indices[:min(limit, len(indices))]

    c = Counter()
    pid_set = set()
    strong_ids = set()

    def get_pid(idx):
        s = None
        if hasattr(base_ds, 'samples'):
            s = base_ds.samples[idx]
        else:
            try:
                item = base_ds[idx]
                if isinstance(item, dict):
                    s = item
                elif isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[-1], dict):
                    s = item[-1]
            except Exception:
                s = None
        if isinstance(s, dict):
            return int(s.get('person_id') or s.get('pid') or s.get('label') or -1)
        return -1

    for orig_idx in indices:
        pid = get_pid(orig_idx)
        if pid >= 0:
            pid_set.add(pid)
        mods_img = infer_modalities_of_sample(base_ds, orig_idx, include_text=False)
        mods_all = infer_modalities_of_sample(base_ds, orig_idx, include_text=True)
        # 统计只计图像模态
        for m in mods_img:
            c[canon_mod(m)] += 1
        has_vis = ('vis' in mods_img)
        has_nonvis = bool(mods_img & {'nir','sk','cp'}) or ('text' in mods_all)
        if pid >= 0 and has_vis and has_nonvis:
            strong_ids.add(pid)

    print("数据集统计:")
    print(f"  总ID数: {len(pid_set)}")
    print(f"  总样本数: {len(indices)}")
    # 只展示 vis/nir/sk/cp
    print("  各模态分布:", {k:v for k,v in c.items() if k in IMG_MODALITIES})
    print(f"  可配对ID数 (K≥2): {len(strong_ids)} ({len(strong_ids)/max(1,len(pid_set)):.1%})")
```

> 这样，无论上游有没有把字符串写成 `rgb`，这个分析都__强制走规范化__，不会再出现 `{'rgb': ...}`。

---

# C. 用“批采样器”，别再把 `batch_size` 传给采样器

你当前的报错：`ModalAwarePKSampler_Strict.__init__() got an unexpected keyword argument 'batch_size'`
根因：PyTorch 的惯例是__自定义分组应通过 `batch_sampler`__，不是把 `batch_size` 塞进普通 `sampler`。

新增一个__批采样器__类（仍放 `datasets/dataset.py`），专门产出一整个 batch 的索引列表：

```python
class ModalAwarePKBatchSampler_Strict(Sampler):
    """
    产出“整批索引列表”的 Batch Sampler：
    - 强配对：同一ID需有 vis 和 nonvis 侧
    - 每个ID在一个batch内取 K//2 vis + K//2 nonvis（奇数时 nonvis 多1）
    - 允许 ID 复用，避免早停
    """
    def __init__(self, dataset, num_ids_per_batch=4, num_instances=4,
                 allow_id_reuse=True, include_text=True, min_modal_coverage=0.6, **_):
        self.P = int(num_ids_per_batch)  # IDs
        self.K = int(num_instances)      # instances per ID
        self.allow_id_reuse = bool(allow_id_reuse)
        self.include_text = bool(include_text)
        self.min_modal_coverage = float(min_modal_coverage)

        # 兼容 Subset
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            self.base_dataset = dataset.dataset
            self.indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))

        # 建立 pid -> {vis:[...], nonvis:[...]}
        self.pid_to_mod_idxs = {}
        self.pids = set()

        def get_pid_by_orig_idx(orig_idx):
            s = None
            if hasattr(self.base_dataset, 'samples'):
                s = self.base_dataset.samples[orig_idx]
            else:
                try:
                    item = self.base_dataset[orig_idx]
                    if isinstance(item, dict):
                        s = item
                    elif isinstance(item, (list, tuple)) and len(item) >= 3 and isinstance(item[-1], dict):
                        s = item[-1]
                except Exception:
                    s = None
            if isinstance(s, dict):
                return int(s.get('person_id') or s.get('pid') or s.get('label') or -1)
            return -1

        for orig_idx in self.indices:
            pid = get_pid_by_orig_idx(orig_idx)
            if pid < 0:
                continue
            self.pids.add(pid)

            mods_img = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=False)
            mods_all = infer_modalities_of_sample(self.base_dataset, orig_idx, include_text=True)

            has_vis = ('vis' in mods_img)
            has_nonvis = bool(mods_img & {'nir','sk','cp'}) or ('text' in mods_all)

            d = self.pid_to_mod_idxs.setdefault(pid, {'vis': [], 'nonvis': []})
            if has_vis: d['vis'].append(orig_idx)
            if has_nonvis: d['nonvis'].append(orig_idx)

        self.strong_ids = [pid for pid, d in self.pid_to_mod_idxs.items()
                           if len(d['vis']) > 0 and len(d['nonvis']) > 0]
        self.soft_ids = [pid for pid in self.pids if pid not in self.strong_ids]

        # 粗略估算可用 batch 数（仅用来 __len__）
        est_pairs = sum(min(len(self.pid_to_mod_idxs[pid]['vis']),
                            len(self.pid_to_mod_idxs[pid]['nonvis'])) for pid in self.strong_ids)
        self._len_est = max(1, est_pairs // max(1, (self.P * self.K)))

        logging.info("==================================================")
        logging.info("数据集采样能力分析(采样器视角)")
        logging.info(f"  强配对ID数: {len(self.strong_ids)} / {len(self.pids)}")
        logging.info(f"  估算可生成batch数: ~{self._len_est}")
        logging.info("==================================================")

    def __len__(self):
        # 允许复用 → 用估算值；不允许复用可按资源上限估计
        return int(self._len_est) if self.allow_id_reuse else max(1, len(self.strong_ids) // self.P)

    def __iter__(self):
        import random
        strong_pool = list(self.strong_ids)
        soft_pool = list(self.soft_ids)

        while True:
            # 选 P 个ID（不足用 soft 或复用补齐）
            if len(strong_pool) >= self.P:
                cur_ids = (random.sample(strong_pool, self.P)
                           if not self.allow_id_reuse else random.choices(strong_pool, k=self.P))
            else:
                need = self.P - len(strong_pool)
                fillers = (random.sample(soft_pool, min(need, len(soft_pool)))
                           if not self.allow_id_reuse else random.choices(soft_pool, k=need))
                cur_ids = list(strong_pool) + fillers
                if not cur_ids:
                    break

            batch_indices = []
            for pid in cur_ids:
                d = self.pid_to_mod_idxs.get(pid, {'vis': [], 'nonvis': []})
                vis_pool = d['vis'] if d['vis'] else d['nonvis']  # 兜底
                nonvis_pool = d['nonvis'] if d['nonvis'] else d['vis']

                k_vis = self.K // 2
                k_nonvis = self.K - k_vis  # 奇数→nonvis 多1
                if len(vis_pool) >= k_vis:
                    batch_indices += random.sample(vis_pool, k_vis)
                else:
                    batch_indices += random.choices(vis_pool or nonvis_pool, k=k_vis)
                if len(nonvis_pool) >= k_nonvis:
                    batch_indices += random.sample(nonvis_pool, k_nonvis)
                else:
                    batch_indices += random.choices(nonvis_pool or vis_pool, k=k_nonvis)

            if len(batch_indices) != self.P * self.K:
                # 防御：不完整 batch 直接跳过
                continue

            yield batch_indices

            if not self.allow_id_reuse:
                for pid in set(cur_ids):
                    if pid in strong_pool:
                        strong_pool.remove(pid)
                    elif pid in soft_pool:
                        soft_pool.remove(pid)
                if len(strong_pool) < 1 and len(soft_pool) < 1:
                    break
```

> 之后在 __train.py__ 用 `batch_sampler=` 挂载它（不要把 `batch_size` 传给采样器了）：

```python
# ====== train.py 关键替换 ======
P = cfg.num_ids_per_batch         # e.g. 4
K = cfg.instances_per_id          # e.g. 4
effective_bs = P * K

train_batch_sampler = ModalAwarePKBatchSampler_Strict(
    train_subset,
    num_ids_per_batch=P,
    num_instances=K,
    allow_id_reuse=cfg.allow_id_reuse,
    include_text=True,
    min_modal_coverage=cfg.min_modal_coverage
)

train_loader = DataLoader(
    train_subset,
    batch_sampler=train_batch_sampler,   # ✅ 用 batch_sampler
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=compatible_collate_fn
)
# 注意：此时不要再传 batch_size / shuffle / sampler
```

---

# D. 你的“vis 每个子ID约10张”提示

这句话非常关键：它说明你__真实数据里__确实有其他模态（nir/sk/cp），只是统计面板把它们都算进了 “rgb/vis”。常见根因有两种：

* __目录解析__把 `nir/sk/cp` 也塞进了 `'images': {'rgb': [...全部图片...]}`。
* __样本 primary 字段__硬写成 `rgb` 覆盖了上层信息（你前一个版本就是被它误导的）。

建议你检查一次「样本构造」的位置（通常在你自定义 `Dataset` 的构造或索引阶段），确保映射正确：

```python
# 伪代码：按人ID收集各模态路径
MOD_DIR2MOD = {'vis':'vis', 'nir':'nir', 'sk':'sk', 'sketch':'sk', 'cp':'cp'}
images = { 'vis':[], 'nir':[], 'sk':[], 'cp':[] }
for mod_dir, canon in MOD_DIR2MOD.items():
    # e.g. root/<pid>/<mod_dir>/*.jpg
    paths = list_all_images(pid_dir/mod_dir)
    if paths:
        images[canon].extend(paths)

sample = {
    'person_id': pid,
    'images': {k: v for k,v in images.items() if len(v)>0},
    # 禁止把所有都写到 'rgb' 或 'vis' 里
    'modality_mask': {k: float(len(v)>0) for k,v in images.items()},
    # primary 可不写；若写，务必不要覆盖上面两项的真实分布
}
```

---

# E. 快速自检（跑 200 个样本就能看出修复成效）

```python
# 放在 debug_modality.py 或 train.py 开头调用一次
def quick_scan(ds, n=200):
    if hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
        base, idxs = ds.dataset, ds.indices[:min(n, len(ds.indices))]
    else:
        base, idxs = ds, list(range(min(n, len(ds))))
    c = Counter()
    pair = 0
    pids = set()
    for orig_idx in idxs:
        m_img = infer_modalities_of_sample(base, orig_idx, include_text=False)
        m_all = infer_modalities_of_sample(base, orig_idx, include_text=True)
        for m in m_img:
            c[canon_mod(m)] += 1
        vis = 'vis' in m_img
        nonvis = bool(m_img & {'nir','sk','cp'}) or ('text' in m_all)
        pair += int(vis and nonvis)
        # pid 统计可选
    print("[仅图像] 模态计数:", {k:v for k,v in c.items() if k in IMG_MODALITIES})
    print(f"vis+非vis 配对样本: {pair}/{len(idxs)}  比例: {pair/len(idxs):.1%}")

# 调用
# quick_scan(train_subset, n=500)
# analyze_sampling_capability(train_subset, limit=2000)
```

预期：你会看到 __不再有 `rgb`__，而是 `{'vis': xxx, 'nir': xxx, 'sk': xxx, 'cp': xxx}`，且配对比例明显 > 0%。

---

## 常见“回归点”再提醒

* 搜索全工程，统一改用规范名：
  `grep -R "rgb\>|ir\>|sketch\>|cpencil" -n`
  所有落点要么删除，要么用 `canon_mod()` 归一化。
* 所有统计/日志/可视化都__调用推断函数__，不要直接读 `sample['modality']`。
* `DataLoader` 用 `batch_sampler=` 绑定你上面的 __ModalAwarePKBatchSampler\_Strict__；
  不要再把 `batch_size` 传进采样器（那会再次 TypeError）。

---

## 期望训练变化

* 采样面板不再是 `{'rgb': 14910}`；
* 强配对 ID 数从 0 → 大多数（>70% 很常见）；
* 一个 epoch 的 step 数恢复到预期（不再 80 步早停）；
* `pair_coverage_mavg ≥ 0.85`，`SDM` 开始有值，`CE` 从 \~5.99 下降，Top-1 由 0% 起步上升。

把以上四段补丁按顺序贴好再跑一次；如果还有“只见 vis 没见 nir/sk/cp”，就去 Dataset 构造那一段核对目录→模态键的映射（D 段的伪代码），基本就能一次到位。
