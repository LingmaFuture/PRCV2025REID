# loss.py - 损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from configs.config import TrainingConfig

class CenterLoss(nn.Module):
    """中心损失"""
    
    def __init__(self, num_classes, feature_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        self.register_buffer('centers', torch.randn(num_classes, feature_dim))
        self.centers.data.uniform_(-1, 1)
    
    def forward(self, features, labels):
        """
        Args:
            features: (batch_size, feature_dim)
            labels: (batch_size,)
        """
        batch_size = features.size(0)
        
        # 获取每个样本对应的中心
        centers_batch = self.centers[labels]  # (batch_size, feature_dim)
        
        # 计算中心损失
        center_loss = F.mse_loss(features, centers_batch.detach())
        
        # 更新中心（使用移动平均）
        with torch.no_grad():
            for i in range(batch_size):
                label = labels[i].item()
                if 0 <= label < self.num_classes:
                    center = self.centers[label]
                    feature = features[i]
                    
                    # 移动平均更新
                    self.centers[label] = center + self.alpha * (feature - center)
        
        return center_loss


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    
    def __init__(self, alpha=1, gamma=2, num_classes=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = -ce_loss
            focal_loss = at * (1 - pt) ** self.gamma * logpt
        else:
            focal_loss = (1 - pt) ** self.gamma * (-ce_loss)
        
        return focal_loss.mean()


class ImprovedCombinedLoss(nn.Module):
    """改进的组合损失函数"""
    
    def __init__(self, config: TrainingConfig):
        super(ImprovedCombinedLoss, self).__init__()
        
        self.config = config
        
        # 各种损失
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2, num_classes=config.num_classes)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
        self.center_loss = CenterLoss(config.num_classes, config.feature_dim)
        
        # 距离度量
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, outputs, labels):
        """计算组合损失"""
        logits = outputs['logits']
        features = outputs['features']
        modality_features = outputs['modality_features']
        modality_weights = outputs.get('modality_weights')
        
        # 1. 分类损失
        ce_loss = self.ce_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        
        # 2. 三元组损失
        triplet_loss = self._compute_batch_hard_triplet_loss(features, labels)
        
        # 3. 中心损失
        center_loss = self.center_loss(features, labels)
        
        # 4. 模态对齐损失
        alignment_loss = self._compute_modality_alignment_loss(modality_features, labels)
        
        # 5. 模态平衡损失
        balance_loss = self._compute_modality_balance_loss(modality_weights)
        
        # 6. 模态一致性损失
        consistency_loss = self._compute_modality_consistency_loss(modality_features)
        
        # 总损失
        total_loss = (
            self.config.ce_weight * (0.7 * ce_loss + 0.3 * focal_loss) +
            self.config.triplet_weight * triplet_loss +
            self.config.center_weight * center_loss +
            self.config.alignment_weight * alignment_loss +
            self.config.modality_balance_weight * balance_loss +
            0.1 * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'focal_loss': focal_loss,
            'triplet_loss': triplet_loss,
            'center_loss': center_loss,
            'alignment_loss': alignment_loss,
            'balance_loss': balance_loss,
            'consistency_loss': consistency_loss
        }
    
    def _compute_batch_hard_triplet_loss(self, features, labels):
        """批次难样本三元组损失"""
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算距离矩阵
        dist_mat = torch.cdist(features, features, p=2)
        
        # 创建标签矩阵
        labels_expanded = labels.view(-1, 1).expand(batch_size, batch_size)
        labels_equal = labels_expanded.eq(labels_expanded.t())
        
        triplet_losses = []
        
        for i in range(batch_size):
            # 正样本掩码（相同标签，但不是自己）
            pos_mask = labels_equal[i] & (torch.arange(batch_size, device=features.device) != i)
            # 负样本掩码（不同标签）
            neg_mask = ~labels_equal[i]
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # 最难正样本（最远距离）
                pos_dist = dist_mat[i][pos_mask]
                hardest_pos_dist = pos_dist.max()
                
                # 最难负样本（最近距离）
                neg_dist = dist_mat[i][neg_mask]
                hardest_neg_dist = neg_dist.min()
                
                # 三元组损失
                loss = F.relu(hardest_pos_dist - hardest_neg_dist + 0.3)
                triplet_losses.append(loss)
        
        return sum(triplet_losses) / max(len(triplet_losses), 1)
    
    def _compute_modality_alignment_loss(self, modality_features, labels):
        """模态对齐损失"""
        modalities = list(modality_features.keys())
        alignment_losses = []
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                feat1 = F.normalize(modality_features[modalities[i]], p=2, dim=1)
                feat2 = F.normalize(modality_features[modalities[j]], p=2, dim=1)
                
                # 计算余弦相似度
                similarity = self.cosine_similarity(feat1, feat2)
                
                # 对于相同ID，鼓励高相似度；对于不同ID，惩罚高相似度
                batch_size = feat1.size(0)
                same_id_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
                
                # 简化：使用对角线元素（自己与自己的比较）
                pos_loss = (1 - similarity).mean()
                alignment_losses.append(pos_loss)
        
        return sum(alignment_losses) / max(len(alignment_losses), 1)
    
    def _compute_modality_balance_loss(self, modality_weights):
        """模态平衡损失，鼓励权重均匀分布"""
        if modality_weights is None:
            return torch.tensor(0.0)
        
        # 计算权重熵，鼓励均匀分布
        eps = 1e-8
        log_weights = torch.log(modality_weights + eps)
        entropy = -(modality_weights * log_weights).sum(dim=1).mean()
        
        # 最大熵为 log(num_modalities)
        max_entropy = math.log(modality_weights.size(1))
        
        # 损失为负熵（鼓励高熵）
        balance_loss = max_entropy - entropy
        
        return balance_loss
    
    def _compute_modality_consistency_loss(self, modality_features):
        """模态一致性损失"""
        modalities = list(modality_features.keys())
        consistency_losses = []
        
        # 计算所有模态特征的平均
        stacked_features = torch.stack([modality_features[mod] for mod in modalities], dim=0)
        mean_features = stacked_features.mean(dim=0)
        
        # 每个模态与平均特征的MSE损失
        for modality in modalities:
            feat = modality_features[modality]
            consistency_loss = F.mse_loss(feat, mean_features.detach())
            consistency_losses.append(consistency_loss)
        
        return sum(consistency_losses) / len(consistency_losses)