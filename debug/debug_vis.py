# 调试vis模态检测问题
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset

config = TrainingConfig()
full_dataset = MultiModalDataset(config, split='train')

sample = full_dataset[0]
print(f"样本0详细信息:")
print(f"  modality字段: {sample.get('modality', 'None')}")
print(f"  modality_mask keys: {list(sample['modality_mask'].keys())}")
print(f"  modality_mask values: {sample['modality_mask']}")

# 检查vis模态的mask值
vis_mask = sample['modality_mask'].get('vis', 'Not found')
print(f"  vis mask value: {vis_mask}, type: {type(vis_mask)}")

if isinstance(vis_mask, (int, float)):
    print(f"  vis mask > 0.5? {vis_mask > 0.5}")
elif hasattr(vis_mask, 'item'):
    print(f"  vis mask > 0.5? {vis_mask.item() > 0.5}")

# 检查images字段
if 'images' in sample:
    print(f"  images keys: {list(sample['images'].keys())}")
    for mod_name, img_tensor in sample['images'].items():
        if hasattr(img_tensor, 'abs') and hasattr(img_tensor, 'sum'):
            tensor_sum = float(img_tensor.abs().sum())
            print(f"  {mod_name} tensor sum: {tensor_sum}")