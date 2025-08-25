# 临时诊断脚本：分析采样器约束问题
import json
import os
from collections import defaultdict, Counter
from configs.config import TrainingConfig

def analyze_pairable_ids():
    """分析可配对ID统计"""
    config = TrainingConfig()
    
    print("=== 数据分布分析 ===")
    
    # 1. 加载JSON标注
    with open(config.json_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 2. 统计每个ID的模态分布
    id_modality_count = defaultdict(Counter)
    id_sample_count = Counter()
    
    for item in annotations:
        file_path = item['file_path']
        parts = file_path.split('/')
        if len(parts) >= 3 and parts[1].isdigit():
            person_id = int(parts[1])
            id_sample_count[person_id] += 1
            
            # vis模态直接从JSON获得
            id_modality_count[person_id]['vis'] += 1
            
            # 检查其他模态是否在文件系统中存在
            person_id_str = f"{person_id:04d}"
            for modality in ['nir', 'sk', 'cp']:
                folder_path = os.path.join(config.data_root, modality, person_id_str)
                if os.path.exists(folder_path):
                    images = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    id_modality_count[person_id][modality] = len(images)
            
            # 文本模态
            if item['caption'].strip():
                id_modality_count[person_id]['text'] += 1
    
    # 3. 分析可配对能力
    total_ids = len(id_modality_count)
    pairable_ids = []
    vis_only_ids = []
    nonvis_only_ids = []
    
    for pid, counts in id_modality_count.items():
        has_vis = counts['vis'] > 0
        has_nonvis = (counts['nir'] + counts['sk'] + counts['cp'] + counts['text']) > 0
        
        if has_vis and has_nonvis:
            pairable_ids.append(pid)
        elif has_vis:
            vis_only_ids.append(pid)
        elif has_nonvis:
            nonvis_only_ids.append(pid)
    
    print(f"总身份ID数: {total_ids}")
    print(f"可配对ID数 (vis+非vis): {len(pairable_ids)}")
    print(f"仅vis ID数: {len(vis_only_ids)}")
    print(f"仅非vis ID数: {len(nonvis_only_ids)}")
    print(f"可配对比例: {len(pairable_ids)/total_ids*100:.1f}%")
    
    # 4. 根据当前配置估算最大batch数
    Kmin = config.num_instances  # 2
    unique_id_per_batch = config.num_ids_per_batch  # 3
    
    print(f"\n=== 当前采样器配置 ===")
    print(f"每批次ID数 (P): {unique_id_per_batch}")
    print(f"每ID实例数 (K): {Kmin}")
    print(f"批次大小: {config.batch_size}")
    
    # 估算可配对ID能提供的最大batch数
    if len(pairable_ids) >= unique_id_per_batch:
        # 简化估算：假设每个ID都能提供Kmin个样本
        max_batches_from_pairable = len(pairable_ids) // unique_id_per_batch
        print(f"基于可配对ID的理论最大batch数: ~{max_batches_from_pairable}")
        
        # 更精确估算：考虑每个ID的实际样本数
        usable_samples_per_id = []
        for pid in pairable_ids:
            total_samples = sum(id_modality_count[pid].values())
            usable_samples_per_id.append(min(total_samples, 10))  # 假设最多用10个样本
        
        total_usable_samples = sum(usable_samples_per_id)
        estimated_max_batches = total_usable_samples // config.batch_size
        print(f"基于实际样本数的估算最大batch数: ~{estimated_max_batches}")
    else:
        print(f"❌ 可配对ID数({len(pairable_ids)}) < 每批次所需ID数({unique_id_per_batch})")
        print("   这会导致采样器无法生成任何batch!")
    
    # 5. 展示一些典型ID的模态分布
    print(f"\n=== 前10个可配对ID的模态分布示例 ===")
    for i, pid in enumerate(pairable_ids[:10]):
        counts = id_modality_count[pid]
        print(f"ID {pid:4d}: vis={counts['vis']:2d} nir={counts['nir']:2d} sk={counts['sk']:2d} cp={counts['cp']:2d} text={counts['text']:2d}")
    
    return {
        'total_ids': total_ids,
        'pairable_ids': len(pairable_ids),
        'estimated_max_batches': estimated_max_batches if len(pairable_ids) >= unique_id_per_batch else 0
    }

if __name__ == "__main__":
    stats = analyze_pairable_ids()