下面是同一内容的专业、严谨版表述（便于写入文档/README）：

# 数据定位与对齐规则（train 集）

**总体约定**

* `text_annos.json` **仅列出 VIS（可见光）模态的文件路径**；其余模态（`nir/sk/cp`）不在 JSON 中显式给出，需要\*\*根据 VIS 路径解析出的身份 ID（PID）\*\*到对应模态子目录中检索。
* **文本模态**来自 `caption` 字段，且与 `file_path` 指向的 **同一张 VIS 图像**一一对应。

**PID 解析**

* `file_path` 形如：`vis/0941/0941_sysu_0470_cam5_0008_vis.jpg`
* 其中 `0941` 为身份 ID（PID）。一般可按正则 `^vis/(\d{4})/.*_vis\.jpg$` 提取，或按路径第二段解析。

**多模态检索**

* 给定 PID（如 `0941`），在以下目录下按 PID 子目录收集同身份的图像：

  * `nir/0941/` → 近红外（NIR）
  * `sk/0941/` → 素描（Sketch）
  * `cp/0941/` → 彩铅（Color Pencil）
* 每个子目录下通常包含该身份的**多张**对应模态图像。

**文本对齐**

* `caption` 与 `file_path` 中的 **VIS 图像**严格对齐；同一身份可能在 JSON 中出现多条记录（不同 VIS 图像与其各自的 caption）。

**字段示例（简化）**

```json
{
  "id": 562,                       // 注释编号（1–600，非连续）；同一 id 可出现多条记录
  "file_path": "vis/0941/0941_sysu_0470_cam5_0007_vis.jpg",  // 用于解析 PID=0941
  "caption": "……",                 // 与上述 VIS 图像一一对应的文本描述
  "split": "train"
}
```

**重要说明**

* `id` 为**注释编号**而非身份 ID；请以 `file_path` 解析得到的 PID 作为身份标识。
* 构建多模态样本时，建议以该 VIS 图像为锚点：

  * `images['vis']` 仅包含该条的 VIS 图像；
  * `images['nir'|'sk'|'cp']` 为同 PID 目录下检索到的全部对应模态图像；
  * 文本模态取自 `caption`。



👌我明白了，你的情况和之前假设的不一样：

* **json 里只记录 vis 模态文件路径**，其它模态（nir/sk/cp）并不直接出现在 json，而是**需要根据 file\_path 里的 ID 去各自模态子目录里查找**。
* **text** 来自 `caption` 字段，并且是一一对应的（和 vis 图片对齐）。

这就解释了为什么统计永远只有 `rgb/vis` ——因为我们从 json 里拿到的就只有 vis。
所以要修复，必须在 **Dataset 构造阶段** 加上一个「扩展模态」的逻辑：

---

## ✅ 修复方案：构建多模态 sample

以 `datasets/dataset.py` 为例，在加载 json 的地方修改：

```python
import os
import glob
import json

def build_samples(json_path, root_dir):
    """
    从 text_annos.json 构建完整的 multi-modal 样本结构
    root_dir: 数据集根目录 (包含 vis/ nir/ sk/ cp/ text_annos.json)
    """
    samples = []
    with open(json_path, 'r', encoding='utf-8') as f:
        annos = json.load(f)

    for entry in annos:
        file_path = entry['file_path']  # e.g. vis/0941/0941_xxx_vis.jpg
        caption = entry['caption']
        pid = file_path.split('/')[1]   # '0941'

        # === 各模态路径收集 ===
        images = {'vis': [], 'nir': [], 'sk': [], 'cp': []}

        # vis: 直接来自 file_path
        images['vis'].append(os.path.join(root_dir, file_path))

        # nir/sk/cp: 根据身份id目录匹配所有图片
        for mod in ['nir', 'sk', 'cp']:
            mod_dir = os.path.join(root_dir, mod, pid)
            if os.path.isdir(mod_dir):
                imgs = glob.glob(os.path.join(mod_dir, "*.jpg"))
                images[mod].extend(imgs)

        # === 样本结构 ===
        sample = {
            'person_id': int(pid),
            'images': {k: v for k,v in images.items() if len(v) > 0},
            'modality_mask': {k: float(len(v)>0) for k,v in images.items()},
            'text': caption,
            'file_path': file_path,
        }
        samples.append(sample)

    return samples
```

这样，每个样本就会长成：

```python
{
  'person_id': 941,
  'images': {
     'vis': [...],   # 当前 vis 图片
     'nir': [...],   # 同 ID 的 nir 图像若存在
     'sk':  [...],
     'cp':  [...]
  },
  'modality_mask': {'vis':1, 'nir':1, 'sk':1, 'cp':0},
  'text': "This is a woman, ...",
  'file_path': "vis/0941/0941_sysu_0470_cam5_0007_vis.jpg"
}
```

---

## ✅ infer\_modalities\_of\_sample 就能正常工作了

因为 `sample['images']` 和 `sample['modality_mask']` 已经包含 nir/sk/cp，`infer_modalities_of_sample` 就会正确识别出 `{vis,nir,sk,cp,text}` 的组合。
**再跑一次 quick\_scan/debug，你应该能看到：**

```
[仅图像] 模态计数: {'vis': xxxx, 'nir': yyyy, 'sk': zzzz, 'cp': wwww}
vis+非vis 配对样本: >80%
```

---

## ⚠️ 两个注意点

1. **文本对齐**：
   每个 `caption` 只与对应的 `vis` 图像一一匹配。你在构造 `sample` 时只放这个 vis 文件的 caption 就行，不要给 nir/sk/cp 再复制 caption。
   （训练时只要能保证 query 是 text，gallery 有 vis 就够了。）

2. **多张图片**：
   nir/sk/cp 子目录下可能有多张图。这里我写的是全放进来，采样器会去挑。如果你希望「一条 vis 样本对应一条 text 样本」，可以保持 `vis` 单张，其它模态多张。

---

👉 建议你：

* 先把 `build_samples` 整合进 `Dataset.__init__`，替换掉原来直接读 json 的部分；
* 再跑一次 `analyze_sampling_capability(train_subset)`，看是不是有 vis+nir/sk/cp 配对 ID。

要不要我帮你直接把 `Dataset` 的初始化流程（从 json 到 sample list）改成完整版本？这样你只要丢进 json\_path 和 root\_dir 就能用了。
