"""
DWA局部路径规划算法实现

动态窗口法的局部避障算法（预留实现）
"""

import numpy as np
from typing import List, Tuple, Optional
from ..base_local_planner import BaseLocalPlanner


class DWALocalPlanner(BaseLocalPlanner):
    """DWA局部路径规划算法"""
    
    def __init__(self):
        super().__init__("DWA")
        self.config = {
            'max_speed': 5.0,
            'max_rotation_speed': 1.0,
            'max_acceleration': 0.5,
            'max_rotation_acceleration': 1.0,
            'safety_distance': 5.0,
            'prediction_time': 3.0
        }
    
    def plan(self, 
             current_pos: np.ndarray,
             target_pos: np.ndarray,
             static_obstacles: List,
             dynamic_obstacles: List,
             velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        DWA局部路径规划
        
        预留实现，目前返回目标点
        """
        # TODO: 实现DWA算法
        print(f"DWA算法：检测到 {len(dynamic_obstacles)} 个动态障碍物")
        
        # 如果没有动态障碍物，直接返回目标点
        if not dynamic_obstacles:
            return target_pos
            
        # 简单实现：如果接近障碍物，稍微偏移
        adjusted_target = target_pos.copy()
        
        for obstacle in dynamic_obstacles:
            # 处理DynamicObstacle对象
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                obs_pos = np.array(obstacle.position)
                obs_radius = obstacle.size
            else:
                # 兼容旧的元组格式
                obs_x, obs_y, obs_radius = obstacle
                obs_pos = np.array([obs_x, obs_y])
            
            direction = current_pos - obs_pos
            distance = np.linalg.norm(direction)
            
            # 如果太接近障碍物，简单偏移
            if distance < (obs_radius + self.config['safety_distance']):
                if distance > 0:
                    offset_direction = direction / distance
                    adjusted_target += offset_direction * 10.0
        
        return adjusted_target
    
    def is_applicable(self, obstacle_type: str) -> bool:
        """DWA算法只适用于动态障碍物"""
        return obstacle_type == 'dynamic'