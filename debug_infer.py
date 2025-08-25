# 直接调试infer_modalities_of_sample函数
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, infer_modalities_of_sample

config = TrainingConfig()
full_dataset = MultiModalDataset(config, split='train')

print("调试infer_modalities_of_sample函数:")

sample = full_dataset[0]
print(f"直接获取样本0: modality_mask = {sample['modality_mask']}")

# 手动模拟infer_modalities_of_sample的逻辑
print("\n手动检测模态:")
mods = set()
for mod_name, mask_val in sample['modality_mask'].items():
    print(f"  检查{mod_name}: mask_val={mask_val}, type={type(mask_val)}")
    if isinstance(mask_val, float):
        if mask_val > 0.5:
            mods.add(mod_name)
            print(f"    -> 添加{mod_name}到模态集合")
        else:
            print(f"    -> {mod_name} mask值{mask_val} <= 0.5，跳过")
    else:
        print(f"    -> {mod_name} mask值类型不是float，跳过")

print(f"手动检测结果: {mods}")

# 调用原函数
result = infer_modalities_of_sample(full_dataset, 0)
print(f"infer_modalities_of_sample结果: {result}")

# 检查文本
if 'text_description' in sample:
    td = sample['text_description']
    if isinstance(td, list) and len(td) > 0:
        td = td[0]
    if isinstance(td, str) and td.strip():
        mods.add('text')
        print(f"添加text模态: '{td[:50]}...'")

print(f"最终结果（包含文本）: {mods}")