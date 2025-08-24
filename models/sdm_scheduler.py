# sdm_scheduler.py
"""
SDM权重和温度调度器
实现文档中推荐的"热身→放量"策略
"""

import math
import logging


class SDMWeightScheduler:
    """
    SDM权重调度器
    
    实现文档推荐的权重调度策略：
    - Epoch 0-2: λ_sdm = 0 (热身)
    - Epoch 3-5: λ_sdm = 0.5 (初始)
    - Epoch 6+: λ_sdm = 1.0 (稳定)
    - 可选: λ_sdm = 1.5 (上限)
    """
    
    def __init__(self, config):
        """
        Args:
            config: 训练配置对象
        """
        self.warmup_epochs = getattr(config, 'sdm_weight_warmup_epochs', 1)  # guide5.md
        self.weight_schedule = getattr(config, 'sdm_weight_schedule', [0.1, 0.3, 0.5])  # guide5.md: 渐进式
        self.initial_weight = getattr(config, 'sdm_weight_initial', 0.1)
        self.final_weight = getattr(config, 'sdm_weight_final', 0.5)
        self.max_weight = getattr(config, 'sdm_weight_max', 0.5)
        
        # 当前权重
        self.current_weight = 0.0
        
        logging.info(f"SDM权重调度器初始化:")
        logging.info(f"  热身epochs: {self.warmup_epochs}")
        logging.info(f"  初始权重: {self.initial_weight}")
        logging.info(f"  目标权重: {self.final_weight}")
        logging.info(f"  最大权重: {self.max_weight}")
    
    def get_weight(self, epoch):
        """
        获取当前epoch的SDM权重 - guide5.md: 实现渐进式调度0.1→0.3→0.5
        
        Args:
            epoch (int): 当前epoch (从1开始)
            
        Returns:
            float: SDM权重
        """
        if epoch <= self.warmup_epochs:
            # guide5.md: 热身阶段完全跳过SDM
            weight = 0.0
        else:
            # guide5.md: warmup后按渐进权重调度
            schedule_idx = min(epoch - self.warmup_epochs - 1, len(self.weight_schedule) - 1)
            if schedule_idx >= 0 and schedule_idx < len(self.weight_schedule):
                weight = self.weight_schedule[schedule_idx]
            else:
                weight = self.final_weight
        
        self.current_weight = weight
        return weight
    
    def can_increase_weight(self, epoch, train_metrics, val_metrics=None):
        """
        判断是否可以增加权重到最大值
        
        Args:
            epoch (int): 当前epoch
            train_metrics (dict): 训练指标
            val_metrics (dict, optional): 验证指标
            
        Returns:
            bool: 是否可以增加权重
        """
        if epoch < 10:  # 至少训练10个epoch
            return False
        
        # 检查训练稳定性
        stability_score = train_metrics.get('stability_score', 0.0)
        if stability_score < 0.8:  # 训练不够稳定
            return False
        
        # 检查验证性能（如果有）
        if val_metrics:
            current_map = val_metrics.get('map_avg2', 0.0)
            if current_map < 0.1:  # mAP太低
                return False
        
        return True
    
    def increase_to_max(self):
        """将权重增加到最大值"""
        if self.current_weight < self.max_weight:
            self.current_weight = self.max_weight
            logging.info(f"SDM权重增加到最大值: {self.max_weight}")
            return True
        return False
    
    def decrease_weight(self, reason=""):
        """降低权重（出现不稳定时）"""
        if self.current_weight > self.initial_weight:
            self.current_weight = self.initial_weight
            logging.warning(f"SDM权重降低到初始值: {self.initial_weight} (原因: {reason})")
            return True
        return False


class SDMTemperatureScheduler:
    """
    SDM温度调度器
    
    实现文档推荐的温度调度策略：
    - 起步：τ = 0.12
    - 3个epoch后稳定：τ = 0.10
    - 出现不稳定：τ = 0.15
    """
    
    def __init__(self, config):
        """
        Args:
            config: 训练配置对象
        """
        self.init_temp = getattr(config, 'sdm_init_temperature', 0.12)
        self.final_temp = getattr(config, 'sdm_final_temperature', 0.10)
        self.fallback_temp = getattr(config, 'sdm_fallback_temperature', 0.15)
        self.warmup_epochs = getattr(config, 'sdm_temp_warmup_epochs', 3)
        
        # 当前温度
        self.current_temp = self.init_temp
        self.use_fallback = False
        
        logging.info(f"SDM温度调度器初始化:")
        logging.info(f"  初始温度: {self.init_temp}")
        logging.info(f"  目标温度: {self.final_temp}")
        logging.info(f"  回退温度: {self.fallback_temp}")
        logging.info(f"  温度调整epochs: {self.warmup_epochs}")
    
    def get_temperature(self, epoch):
        """
        获取当前epoch的温度参数
        
        Args:
            epoch (int): 当前epoch (从1开始)
            
        Returns:
            float: 温度参数
        """
        if self.use_fallback:
            # 使用回退温度
            return self.fallback_temp
        
        if epoch <= self.warmup_epochs:
            # 初始阶段：τ = 0.12
            temp = self.init_temp
        else:
            # 稳定阶段：τ = 0.10
            temp = self.final_temp
        
        self.current_temp = temp
        return temp
    
    def check_stability(self, train_metrics):
        """
        检查训练稳定性，决定是否使用回退温度
        
        Args:
            train_metrics (dict): 训练指标
            
        Returns:
            bool: 是否需要使用回退温度
        """
        # 检查损失异常
        sdm_loss = train_metrics.get('sdm_loss', 0.0)
        if sdm_loss > 5.0 or sdm_loss < 0:
            self.use_fallback = True
            logging.warning(f"SDM损失异常 ({sdm_loss:.4f})，使用回退温度 {self.fallback_temp}")
            return True
        
        # 检查稳定性分数
        stability_score = train_metrics.get('stability_score', 0.0)
        if stability_score < 0.5:  # 降低阈值，避免过度使用回退温度
            self.use_fallback = True
            logging.warning(f"训练不稳定 (score={stability_score:.2f})，使用回退温度 {self.fallback_temp}")
            return True
        
        return False
    
    def reset_to_normal(self):
        """重置到正常温度"""
        if self.use_fallback:
            self.use_fallback = False
            logging.info(f"训练稳定，恢复正常温度调度")
            return True
        return False


class SDMScheduler:
    """
    SDM综合调度器
    
    统一管理权重和温度调度
    """
    
    def __init__(self, config):
        """
        Args:
            config: 训练配置对象
        """
        self.weight_scheduler = SDMWeightScheduler(config)
        self.temp_scheduler = SDMTemperatureScheduler(config)
        
    def get_parameters(self, epoch, train_metrics, val_metrics=None):
        """
        获取当前epoch的SDM参数
        
        Args:
            epoch (int): 当前epoch
            train_metrics (dict): 训练指标
            val_metrics (dict, optional): 验证指标
            
        Returns:
            tuple: (weight, temperature)
        """
        # ✅ 容错：如果指标为空或缺少关键字段，返回当前值，不做判定
        if not train_metrics or 'stability_score' not in train_metrics:
            current_weight = getattr(self.weight_scheduler, 'current_weight', self.weight_scheduler.initial_weight)
            current_temp = getattr(self.temp_scheduler, 'current_temp', self.temp_scheduler.init_temp)
            return current_weight, current_temp
            
        # 检查稳定性
        self.temp_scheduler.check_stability(train_metrics)
        
        # 获取权重和温度
        weight = self.weight_scheduler.get_weight(epoch)
        temperature = self.temp_scheduler.get_temperature(epoch)
        
        return weight, temperature
    
    def can_increase_weight(self, epoch, train_metrics, val_metrics=None):
        """判断是否可以增加权重"""
        return self.weight_scheduler.can_increase_weight(epoch, train_metrics, val_metrics)
    
    def increase_weight(self):
        """增加权重"""
        return self.weight_scheduler.increase_to_max()
    
    def decrease_weight(self, reason=""):
        """降低权重"""
        return self.weight_scheduler.decrease_weight(reason)
    
    def reset_temperature(self):
        """重置温度"""
        return self.temp_scheduler.reset_to_normal()
