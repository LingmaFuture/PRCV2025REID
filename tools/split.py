# tools/split.py
"""
数据集划分工具
确保训练集和验证集按ID互斥划分，避免数据泄露
"""

import random
import logging
from typing import List, Set, Tuple


def split_ids(all_ids: List[int], val_ratio: float = 0.2, seed: int = 42) -> Tuple[Set[int], Set[int]]:
    """
    按ID划分数据集，确保训练集和验证集互斥
    
    Args:
        all_ids: 所有人员ID列表
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        train_ids: 训练集ID集合
        val_ids: 验证集ID集合
    """
    ids = sorted(list(set(all_ids)))
    rng = random.Random(seed)
    rng.shuffle(ids)
    
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    
    # 验证互斥性
    assert len(val_ids & train_ids) == 0, "训练集和验证集ID不互斥！"
    assert len(val_ids | train_ids) == len(ids), "ID划分不完整！"
    
    logging.info(f"数据集划分: 总ID数={len(ids)}, 训练集ID数={len(train_ids)}, 验证集ID数={len(val_ids)}")
    logging.info(f"验证集比例: {len(val_ids)/len(ids):.3f} (目标: {val_ratio:.3f})")
    
    return train_ids, val_ids


def create_split_datasets(full_dataset, train_ids: Set[int], val_ids: Set[int], config):
    """
    基于ID划分创建训练集和验证集 - 彻底重写版本
    
    Args:
        full_dataset: 完整数据集
        train_ids: 训练集ID集合
        val_ids: 验证集ID集合
        config: 配置对象
        
    Returns:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
    """
    import copy
    from datasets.dataset import MultiModalDataset
    
    logging.info(f"开始数据集划分: 训练ID={len(train_ids)}, 验证ID={len(val_ids)}")
    
    # 从原始数据集中分离训练和验证样本
    train_samples = []
    val_samples = []
    
    for item in full_dataset.data_list:
        person_id = item['person_id']
        if person_id in train_ids:
            train_samples.append(copy.deepcopy(item))
        elif person_id in val_ids:
            val_samples.append(copy.deepcopy(item))
        # 忽略不在任何集合中的样本
    
    logging.info(f"原始样本分配: 训练样本={len(train_samples)}, 验证样本={len(val_samples)}")
    
    # 创建全局label映射（包含所有ID，保证标签一致性）
    all_person_ids = sorted(list(train_ids | val_ids))
    global_pid2label = {pid: i for i, pid in enumerate(all_person_ids)}
    
    # 创建训练数据集实例
    train_dataset = copy.deepcopy(full_dataset)
    train_dataset.data_list = train_samples
    train_dataset.pid2label = global_pid2label
    train_dataset.is_training = True
    
    # 创建验证数据集实例
    val_dataset = copy.deepcopy(full_dataset) 
    val_dataset.data_list = val_samples
    val_dataset.pid2label = global_pid2label
    val_dataset.is_training = False  # 验证集不使用训练增强
    
    # 验证划分正确性
    train_person_ids = set(item['person_id'] for item in train_dataset.data_list)
    val_person_ids = set(item['person_id'] for item in val_dataset.data_list)
    
    # 严格验证
    assert len(train_person_ids & val_person_ids) == 0, f"训练集和验证集ID重叠: {train_person_ids & val_person_ids}"
    assert train_person_ids == train_ids, f"训练集ID不匹配: 期望{len(train_ids)}个, 实际{len(train_person_ids)}个"
    assert val_person_ids == val_ids, f"验证集ID不匹配: 期望{len(val_ids)}个, 实际{len(val_person_ids)}个"
    
    # 验证样本总数
    total_samples = len(train_dataset.data_list) + len(val_dataset.data_list)
    expected_samples = len([item for item in full_dataset.data_list if item['person_id'] in (train_ids | val_ids)])
    assert total_samples == expected_samples, f"样本数不匹配: 分割后{total_samples}, 期望{expected_samples}"
    
    logging.info(f"✅ 数据集划分完成:")
    logging.info(f"  训练集: {len(train_dataset.data_list)} 样本, {len(train_person_ids)} 个ID")
    logging.info(f"  验证集: {len(val_dataset.data_list)} 样本, {len(val_person_ids)} 个ID")
    logging.info(f"  划分比例: 训练{len(train_dataset.data_list)/total_samples:.1%}, 验证{len(val_dataset.data_list)/total_samples:.1%}")
    
    return train_dataset, val_dataset


def verify_split_integrity(train_dataset, val_dataset):
    """
    验证数据集划分的完整性
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
    """
    train_ids = set(item['person_id'] for item in train_dataset.data_list)
    val_ids = set(item['person_id'] for item in val_dataset.data_list)
    
    # 检查互斥性
    if not train_ids.isdisjoint(val_ids):
        common_ids = train_ids & val_ids
        raise ValueError(f"训练集和验证集存在共同ID: {common_ids}")
    
    # 检查完整性
    all_ids = train_ids | val_ids
    logging.info(f"数据集完整性验证通过:")
    logging.info(f"  训练集ID数: {len(train_ids)}")
    logging.info(f"  验证集ID数: {len(val_ids)}")
    logging.info(f"  总ID数: {len(all_ids)}")
    logging.info(f"  训练集样本数: {len(train_dataset.data_list)}")
    logging.info(f"  验证集样本数: {len(val_dataset.data_list)}")
    
    return True
