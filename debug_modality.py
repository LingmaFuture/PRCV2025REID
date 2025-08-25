# 调试模态检测问题
import torch
from configs.config import TrainingConfig
from datasets.dataset import MultiModalDataset, infer_modalities_of_sample
from sklearn.model_selection import train_test_split

def debug_modality_detection():
    """调试模态检测逻辑"""
    print("=== 调试模态检测 ===")
    
    config = TrainingConfig()
    
    # 创建数据集
    full_dataset = MultiModalDataset(config, split='train')
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 检查前几个样本的模态
    print("\n=== 前10个样本的模态检测结果 ===")
    for i in range(min(10, len(full_dataset))):
        # 直接从数据集获取样本
        sample = full_dataset[i]
        print(f"样本 {i}:")
        print(f"  person_id: {sample['person_id']}")
        print(f"  modality字段: {sample.get('modality', 'None')}")
        print(f"  modality_mask: {sample.get('modality_mask', {})}")
        
        # 使用infer_modalities_of_sample检测
        mods = infer_modalities_of_sample(full_dataset, i)
        print(f"  推断的模态: {mods}")
        
        # 检查images字段
        if 'images' in sample:
            available_image_mods = []
            for mod_name, img_tensor in sample['images'].items():
                if torch.is_tensor(img_tensor) and img_tensor.abs().sum() > 1e-6:
                    available_image_mods.append(mod_name)
            print(f"  可用图像模态: {available_image_mods}")
        
        print()
    
    # 统计整个数据集的模态分布
    print("=== 整个数据集模态统计 ===")
    modality_stats = {}
    vis_nonvis_pairs = 0
    
    for i in range(len(full_dataset)):
        mods = infer_modalities_of_sample(full_dataset, i)
        for mod in mods:
            modality_stats[mod] = modality_stats.get(mod, 0) + 1
        
        # 检查是否有vis+非vis组合
        has_vis = 'vis' in mods
        has_nonvis = any(m in mods for m in ['nir', 'sk', 'cp', 'text'])
        if has_vis and has_nonvis:
            vis_nonvis_pairs += 1
    
    print("模态出现次数:")
    for mod, count in modality_stats.items():
        print(f"  {mod}: {count}")
    
    print(f"有vis+非vis组合的样本数: {vis_nonvis_pairs}")
    print(f"比例: {vis_nonvis_pairs/len(full_dataset)*100:.1f}%")

if __name__ == "__main__":
    debug_modality_detection()