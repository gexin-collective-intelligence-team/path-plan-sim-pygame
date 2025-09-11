"""
局部路径规划算法基类

提供统一的接口规范，确保所有局部算法的一致性
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseLocalPlanner(ABC):
    """局部路径规划算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.config = {}
    
    @abstractmethod
    def plan(self, 
             current_pos: np.ndarray,
             target_pos: np.ndarray,
             static_obstacles: List,  # 原始障碍物对象
             dynamic_obstacles: List,  # 原始动态障碍物对象（只包含新增的）
             velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算局部路径调整
        
        Args:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            static_obstacles: 静态障碍物列表（原始对象，不参与局部避障）
            dynamic_obstacles: 动态障碍物列表（原始对象，只包含新增的）
            velocity: 当前速度向量 [vx, vy]
            
        Returns:
            调整后的目标位置 [x, y]
        """
        pass
    
    def set_config(self, config: dict):
        """设置算法参数"""
        self.config.update(config)
    
    def get_config(self) -> dict:
        """获取当前配置"""
        return self.config.copy()
    
    def is_applicable(self, obstacle_type: str) -> bool:
        """
        判断算法是否适用于特定类型的障碍物
        
        Args:
            obstacle_type: 障碍物类型 ('static' 或 'dynamic')
            
        Returns:
            是否适用
        """
        return obstacle_type == 'dynamic'  # 默认只处理动态障碍物