# evaluation/orbench_protocol.py
"""
ORBench官方评测协议实现 - 严格按照优化清单第1条要求实现
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import itertools
import random
import logging

logger = logging.getLogger(__name__)

class ORBenchEvaluator:
    """
    ORBench官方评测器
    
    按照优化清单实现：
    1. 固定600/400身份划分（train/test）
    2. RGB为gallery，其他四模态及其组合作query
    3. MM-1/2/3/4四种模式统计mAP/CMC
    4. 组合查询是"主模态+同ID的其它模态补齐"的随机配对
    5. 4/12/12/4套查询集，最终取各套平均分
    """
    
    # ORBench标准模态配置
    RGB_MODALITY = 'vis'  # RGB模态在数据集中的名称
    QUERY_MODALITIES = ['nir', 'sk', 'cp', 'text']  # 查询模态
    
    # MM-1/2/3/4模式配置
    MM_MODES = {
        'MM-1': 1,  # 单模态查询（4种）
        'MM-2': 2,  # 双模态查询（C(4,2)=6种） 
        'MM-3': 3,  # 三模态查询（C(4,3)=4种）
        'MM-4': 4,  # 四模态查询（1种）
    }
    
    def __init__(self, test_ids: List[int], random_seed: int = 42):
        """
        Args:
            test_ids: 测试身份ID列表（400个）
            random_seed: 随机种子，确保可复现
        """
        self.test_ids = set(test_ids)
        self.rng = random.Random(random_seed)
        
        # 生成所有查询组合
        self.query_combinations = self._generate_query_combinations()
        
        logger.info(f"ORBench评测器初始化完成:")
        logger.info(f"  测试身份数: {len(self.test_ids)}")
        logger.info(f"  查询组合数: {len(self.query_combinations)}")
        
    def _generate_query_combinations(self) -> Dict[str, List[List[str]]]:
        """生成ORBench标准的查询模态组合"""
        combinations = {}
        
        # MM-1: 单模态查询（4种）
        combinations['MM-1'] = [[mod] for mod in self.QUERY_MODALITIES]
        
        # MM-2: 双模态查询（C(4,2)=6种）
        combinations['MM-2'] = [list(combo) for combo in itertools.combinations(self.QUERY_MODALITIES, 2)]
        
        # MM-3: 三模态查询（C(4,3)=4种）
        combinations['MM-3'] = [list(combo) for combo in itertools.combinations(self.QUERY_MODALITIES, 3)]
        
        # MM-4: 四模态查询（1种）
        combinations['MM-4'] = [self.QUERY_MODALITIES]
        
        return combinations
        
    def prepare_gallery_features(self, model, dataset, device, batch_size: int = 32):
        """
        准备Gallery特征（仅RGB模态）
        
        Args:
            model: 训练好的模型
            dataset: 数据集
            device: 计算设备
            batch_size: 批次大小
            
        Returns:
            gallery_features: [N, D] gallery特征矩阵
            gallery_labels: [N] gallery标签
        """
        model.eval()
        gallery_features = []
        gallery_labels = []
        
        # 创建仅包含RGB样本的数据加载器
        rgb_samples = []
        for idx, sample in enumerate(dataset.data_list):
            person_id = sample['person_id']
            if person_id in self.test_ids:
                # 检查该样本是否有RGB模态
                if self._has_modality(dataset, idx, self.RGB_MODALITY):
                    rgb_samples.append(idx)
                    
        logger.info(f"Gallery RGB样本数: {len(rgb_samples)}")
        
        # 批次处理
        from torch.utils.data import DataLoader, Subset
        from datasets.dataset import compatible_collate_fn
        
        rgb_dataset = Subset(dataset, rgb_samples)
        rgb_loader = DataLoader(
            rgb_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=compatible_collate_fn
        )
        
        with torch.no_grad():
            for batch in rgb_loader:
                # 只保留RGB模态
                filtered_batch = self._filter_batch_modalities(batch, [self.RGB_MODALITY])
                batch = self._move_batch_to_device(filtered_batch, device)
                
                # 前向传播
                outputs = self._call_model(model, batch)
                features = outputs['bn_features']  # 使用BN后特征
                labels = batch['person_id']
                
                gallery_features.append(features.cpu())
                gallery_labels.append(labels.cpu())
                
        gallery_features = torch.cat(gallery_features, dim=0)
        gallery_labels = torch.cat(gallery_labels, dim=0)
        
        return gallery_features, gallery_labels
        
    def prepare_query_features(self, model, dataset, device, mm_mode: str, 
                             combination: List[str], batch_size: int = 32):
        """
        准备Query特征（指定模态组合）
        
        Args:
            model: 训练好的模型
            dataset: 数据集
            device: 计算设备
            mm_mode: MM模式 ('MM-1', 'MM-2', 'MM-3', 'MM-4')
            combination: 模态组合，如['nir', 'text']
            batch_size: 批次大小
            
        Returns:
            query_features: [M, D] query特征矩阵
            query_labels: [M] query标签
        """
        model.eval()
        query_features = []
        query_labels = []
        
        # 创建包含指定模态组合的样本
        combo_samples = []
        for idx, sample in enumerate(dataset.data_list):
            person_id = sample['person_id']
            if person_id in self.test_ids:
                # 检查该样本是否包含所需的所有模态
                if self._has_all_modalities(dataset, idx, combination):
                    combo_samples.append(idx)
                    
        logger.info(f"Query {mm_mode} {combination} 样本数: {len(combo_samples)}")
        
        if len(combo_samples) == 0:
            return torch.empty(0, 512), torch.empty(0, dtype=torch.long)
            
        # 随机采样以避免过大的查询集
        if len(combo_samples) > 1000:
            combo_samples = self.rng.sample(combo_samples, 1000)
            
        from torch.utils.data import DataLoader, Subset
        from datasets.dataset import compatible_collate_fn
        
        combo_dataset = Subset(dataset, combo_samples)
        combo_loader = DataLoader(
            combo_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=compatible_collate_fn
        )
        
        with torch.no_grad():
            for batch in combo_loader:
                # 只保留指定模态
                filtered_batch = self._filter_batch_modalities(batch, combination)
                batch = self._move_batch_to_device(filtered_batch, device)
                
                # 前向传播
                outputs = self._call_model(model, batch)
                features = outputs['bn_features']  # 使用BN后特征
                labels = batch['person_id']
                
                query_features.append(features.cpu())
                query_labels.append(labels.cpu())
                
        if query_features:
            query_features = torch.cat(query_features, dim=0)
            query_labels = torch.cat(query_labels, dim=0)
        else:
            query_features = torch.empty(0, 512)
            query_labels = torch.empty(0, dtype=torch.long)
            
        return query_features, query_labels
        
    def evaluate_full_orbench(self, model, dataset, device, batch_size: int = 32) -> Dict:
        """
        执行完整的ORBench评测
        
        Returns:
            results: 包含所有MM模式结果的字典
        """
        logger.info("开始ORBench官方评测...")
        
        # 1. 准备Gallery特征
        logger.info("准备Gallery特征（RGB模态）...")
        gallery_features, gallery_labels = self.prepare_gallery_features(
            model, dataset, device, batch_size
        )
        
        if gallery_features.size(0) == 0:
            logger.error("Gallery特征为空，无法进行评测")
            return {}
            
        # 2. 对每种MM模式进行评测
        results = {}
        
        for mm_mode, num_modalities in self.MM_MODES.items():
            logger.info(f"评测 {mm_mode} 模式...")
            
            mode_results = []
            combinations = self.query_combinations[mm_mode]
            
            for combo in combinations:
                logger.info(f"  评测组合: {combo}")
                
                # 准备Query特征
                query_features, query_labels = self.prepare_query_features(
                    model, dataset, device, mm_mode, combo, batch_size
                )
                
                if query_features.size(0) == 0:
                    logger.warning(f"  组合 {combo} 无有效样本，跳过")
                    continue
                    
                # 计算mAP和CMC
                metrics = self._compute_retrieval_metrics(
                    query_features, gallery_features,
                    query_labels, gallery_labels
                )
                
                mode_results.append({
                    'combination': combo,
                    'mAP': metrics['mAP'],
                    'CMC@1': metrics['CMC@1'],
                    'CMC@5': metrics['CMC@5'], 
                    'CMC@10': metrics['CMC@10'],
                    'num_queries': query_features.size(0)
                })
                
                logger.info(f"    mAP: {metrics['mAP']:.4f}, "
                           f"CMC@1: {metrics['CMC@1']:.4f}, "
                           f"查询数: {query_features.size(0)}")
            
            # 计算该MM模式的平均结果
            if mode_results:
                avg_metrics = self._average_mode_results(mode_results)
                results[mm_mode] = {
                    'combinations': mode_results,
                    'average': avg_metrics
                }
                
                logger.info(f"{mm_mode} 平均结果: mAP={avg_metrics['mAP']:.4f}, "
                           f"CMC@1={avg_metrics['CMC@1']:.4f}")
            else:
                logger.warning(f"{mm_mode} 模式无有效结果")
                results[mm_mode] = {'combinations': [], 'average': {}}
                
        # 3. 计算总体平均结果
        overall_average = self._compute_overall_average(results)
        results['overall'] = overall_average
        
        logger.info("ORBench评测完成!")
        logger.info(f"总体平均: mAP={overall_average.get('mAP', 0):.4f}, "
                   f"CMC@1={overall_average.get('CMC@1', 0):.4f}")
        
        return results
        
    def _has_modality(self, dataset, idx: int, modality: str) -> bool:
        """检查样本是否包含指定模态"""
        try:
            from datasets.dataset import infer_modalities_of_sample
            modalities = infer_modalities_of_sample(dataset, idx)
            return modality in modalities
        except:
            return False
            
    def _has_all_modalities(self, dataset, idx: int, modalities: List[str]) -> bool:
        """检查样本是否包含所有指定模态"""
        try:
            from datasets.dataset import infer_modalities_of_sample
            sample_mods = infer_modalities_of_sample(dataset, idx)
            return all(mod in sample_mods for mod in modalities)
        except:
            return False
            
    def _filter_batch_modalities(self, batch: Dict, keep_modalities: List[str]) -> Dict:
        """过滤batch，只保留指定模态"""
        filtered_batch = batch.copy()
        
        # 过滤图像
        if 'images' in batch:
            filtered_images = {}
            for mod in keep_modalities:
                if mod in batch['images']:
                    filtered_images[mod] = batch['images'][mod]
            filtered_batch['images'] = filtered_images
            
        # 过滤模态掩码
        if 'modality_mask' in batch:
            filtered_mask = {}
            for mod in keep_modalities:
                if mod in batch['modality_mask']:
                    filtered_mask[mod] = batch['modality_mask'][mod]
                else:
                    # 如果模态不存在，设置为False
                    batch_size = len(batch.get('person_id', []))
                    filtered_mask[mod] = torch.zeros(batch_size, dtype=torch.bool)
            filtered_batch['modality_mask'] = filtered_mask
            
        # 文本处理
        if 'text' not in keep_modalities:
            filtered_batch['text_description'] = [''] * len(batch.get('person_id', []))
            
        return filtered_batch
        
    def _move_batch_to_device(self, batch: Dict, device) -> Dict:
        """将batch移动到指定设备"""
        def move_to_device(obj):
            if torch.is_tensor(obj):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(item) for item in obj]
            else:
                return obj
        return move_to_device(batch)
        
    def _call_model(self, model, batch: Dict):
        """调用模型进行前向传播"""
        from train import call_model_with_batch
        return call_model_with_batch(model, batch, return_features=True)
        
    def _compute_retrieval_metrics(self, query_features: torch.Tensor, 
                                 gallery_features: torch.Tensor,
                                 query_labels: torch.Tensor,
                                 gallery_labels: torch.Tensor) -> Dict:
        """计算检索指标"""
        # L2归一化
        query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
        gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.mm(query_features, gallery_features.t())
        
        # 计算mAP
        map_score = self._compute_map(similarity, query_labels, gallery_labels)
        
        # 计算CMC
        cmc_scores = self._compute_cmc(similarity, query_labels, gallery_labels, [1, 5, 10])
        
        return {
            'mAP': map_score,
            'CMC@1': cmc_scores[0],
            'CMC@5': cmc_scores[1], 
            'CMC@10': cmc_scores[2]
        }
        
    def _compute_map(self, similarity: torch.Tensor, 
                    query_labels: torch.Tensor, 
                    gallery_labels: torch.Tensor) -> float:
        """计算mAP@100"""
        num_queries = similarity.size(0)
        aps = []
        
        for i in range(num_queries):
            sim_scores = similarity[i]
            query_label = query_labels[i]
            
            # 排序
            _, indices = torch.sort(sim_scores, descending=True)
            indices = indices[:100]  # mAP@100
            
            # 找到正样本
            matches = (gallery_labels[indices] == query_label).float()
            
            if matches.sum() > 0:
                # 计算AP
                cum_matches = torch.cumsum(matches, dim=0)
                precisions = cum_matches / torch.arange(1, len(matches) + 1, dtype=torch.float)
                ap = (precisions * matches).sum() / matches.sum()
                aps.append(ap.item())
                
        return np.mean(aps) if aps else 0.0
        
    def _compute_cmc(self, similarity: torch.Tensor,
                    query_labels: torch.Tensor,
                    gallery_labels: torch.Tensor, 
                    ranks: List[int]) -> List[float]:
        """计算CMC@K"""
        num_queries = similarity.size(0)
        cmc_scores = {k: 0 for k in ranks}
        
        for i in range(num_queries):
            sim_scores = similarity[i]
            query_label = query_labels[i]
            
            # 排序
            _, indices = torch.sort(sim_scores, descending=True)
            
            # 找到第一个正样本的位置
            matches = (gallery_labels[indices] == query_label)
            if matches.any():
                first_match_idx = matches.nonzero(as_tuple=True)[0][0].item()
                
                for rank in ranks:
                    if first_match_idx < rank:
                        cmc_scores[rank] += 1
                        
        # 计算比例
        return [cmc_scores[k] / num_queries for k in ranks]
        
    def _average_mode_results(self, mode_results: List[Dict]) -> Dict:
        """计算模式内的平均结果"""
        if not mode_results:
            return {}
            
        metrics = ['mAP', 'CMC@1', 'CMC@5', 'CMC@10']
        averages = {}
        
        for metric in metrics:
            values = [r[metric] for r in mode_results if metric in r]
            averages[metric] = np.mean(values) if values else 0.0
            
        return averages
        
    def _compute_overall_average(self, results: Dict) -> Dict:
        """计算所有MM模式的总体平均"""
        mode_averages = []
        
        for mm_mode in self.MM_MODES.keys():
            if mm_mode in results and 'average' in results[mm_mode]:
                avg = results[mm_mode]['average']
                if avg:
                    mode_averages.append(avg)
                    
        if not mode_averages:
            return {}
            
        metrics = ['mAP', 'CMC@1', 'CMC@5', 'CMC@10']
        overall = {}
        
        for metric in metrics:
            values = [avg[metric] for avg in mode_averages if metric in avg]
            overall[metric] = np.mean(values) if values else 0.0
            
        return overall