# model.py - 模型架构
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import torchvision.models as models
import math


from configs.config import TrainingConfig

class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, feature_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0
        
        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 多头注意力
        Q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.feature_dim
        )
        
        output = self.output_projection(attention_output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + query)
        
        return output


class AdaptiveModalityFusion(nn.Module):
    """自适应模态融合"""
    
    def __init__(self, feature_dim, num_modalities=5):
        super(AdaptiveModalityFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # 模态权重网络
        self.modality_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 4, 1)
        )
        
        # 跨模态注意力
        self.cross_attention = CrossModalAttention(feature_dim)
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, modality_features, modality_mask=None):
        """
        Args:
            modality_features: List of tensors, each (batch_size, feature_dim)
            modality_mask: Dict of modality -> mask (batch_size,)
        """
        batch_size = modality_features[0].size(0)
        valid_features = []
        attention_weights = []
        
        # 计算每个模态的注意力权重
        for i, features in enumerate(modality_features):
            weight = self.modality_attention(features)  # (batch_size, 1)
            attention_weights.append(weight)
            valid_features.append(features)
        
        # 转换为张量并应用mask
        attention_weights = torch.cat(attention_weights, dim=1)  # (batch_size, num_modalities)
        
        if modality_mask is not None:
            mask_tensor = torch.stack([
                torch.tensor([modality_mask[mod].item() for mod in ['vis', 'nir', 'sk', 'cp', 'text']], 
                           device=attention_weights.device) 
                for _ in range(batch_size)
            ])
            attention_weights = attention_weights * mask_tensor
            attention_weights = F.softmax(attention_weights, dim=1)
        else:
            attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权融合
        weighted_features = []
        for i, features in enumerate(valid_features):
            weight = attention_weights[:, i:i+1]
            weighted_features.append(features * weight)
        
        # 基础融合
        basic_fusion = sum(weighted_features)
        
        # 跨模态注意力（如果有多个模态）
        if len(valid_features) > 1:
            stacked_features = torch.stack(valid_features, dim=1)  # (batch, num_modalities, feature_dim)
            attended_fusion = self.cross_attention(
                basic_fusion.unsqueeze(1),  # query
                stacked_features,  # key
                stacked_features   # value
            ).squeeze(1)
        else:
            attended_fusion = basic_fusion
        
        # 最终融合
        concatenated = torch.cat([basic_fusion, attended_fusion], dim=1)
        gate_weight = self.gate(torch.cat(valid_features, dim=1))
        
        final_features = self.fusion_network(concatenated)
        final_features = final_features * gate_weight + basic_fusion * (1 - gate_weight)
        
        return final_features, attention_weights


class ImprovedMultiModalReIDModel(nn.Module):
    """改进的多模态行人重识别模型"""
    
    def __init__(self, config: TrainingConfig):
        super(ImprovedMultiModalReIDModel, self).__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        self.feature_dim = config.feature_dim
        
        # 各模态编码器
        self.vis_encoder = self._build_image_encoder()
        self.nir_encoder = self._build_image_encoder()
        self.sk_encoder = self._build_image_encoder()
        self.cp_encoder = self._build_image_encoder()
        self.text_encoder = self._build_text_encoder()
        
        # 自适应模态融合
        self.fusion_module = AdaptiveModalityFusion(self.feature_dim, num_modalities=5)
        
        # 分类头
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        
        # BN层用于特征归一化
        self.feature_bn = nn.BatchNorm1d(self.feature_dim)
        self.feature_bn.bias.requires_grad_(False)
        
        # 初始化权重
        self._init_weights()
    
    def _build_image_encoder(self):
        """构建图像编码器"""
        if self.config.backbone == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone_feat_dim = 2048
        elif self.config.backbone == 'resnet101':
            backbone = models.resnet101(pretrained=True)
            backbone_feat_dim = 2048
        elif self.config.backbone == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            backbone_feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")
        
        # 去掉最后的分类层
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        return nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_feat_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate)
        )
    
    def _build_text_encoder(self):
        """构建文本编码器"""
        try:
            bert = AutoModel.from_pretrained(self.config.text_model)
            hidden_size = bert.config.hidden_size
        except:
            # 回退到基础BERT
            bert = BertModel.from_pretrained('bert-base-uncased')
            hidden_size = bert.config.hidden_size
        
        return nn.ModuleDict({
            'bert': bert,
            'projection': nn.Sequential(
                nn.Linear(hidden_size, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            )
        })
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch, return_features=False):
        """前向传播"""
        images = batch['images']
        text_descriptions = batch['text_description']
        modality_mask = batch.get('modality_mask', None)
        
        # 提取各模态特征
        vis_feat = self.vis_encoder(images['vis'])
        nir_feat = self.nir_encoder(images['nir'])
        sk_feat = self.sk_encoder(images['sk'])
        cp_feat = self.cp_encoder(images['cp'])
        
        # 文本特征提取
        if isinstance(text_descriptions, (list, tuple)) and any(text_descriptions):
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
            except:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # 过滤空描述
            valid_texts = [text if text else "[UNK]" for text in text_descriptions]
            
            encoded = tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(vis_feat.device)
            attention_mask = encoded['attention_mask'].to(vis_feat.device)
            
            bert_output = self.text_encoder['bert'](
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            text_feat = self.text_encoder['projection'](bert_output.pooler_output)
        else:
            text_feat = torch.zeros_like(vis_feat)
        
        # 模态融合
        modality_features = [vis_feat, nir_feat, sk_feat, cp_feat, text_feat]
        
        fused_features, modality_weights = self.fusion_module(modality_features, modality_mask)
        
        if return_features:
            # 返回用于检索的L2归一化特征
            normalized_features = F.normalize(self.feature_bn(fused_features), p=2, dim=1)
            return normalized_features
        
        # 分类预测
        cls_logits = self.classifier(fused_features)
        
        return {
            'logits': cls_logits,
            'features': fused_features,
            'modality_features': {
                'vis': vis_feat,
                'nir': nir_feat,
                'sk': sk_feat,
                'cp': cp_feat,
                'text': text_feat
            },
            'modality_weights': modality_weights
        }
