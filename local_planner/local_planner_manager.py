"""
局部路径规划算法管理器

负责算法的注册、选择和调用
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Type
from .base_local_planner import BaseLocalPlanner


class LocalPlannerManager:
    """局部路径规划算法管理器"""
    
    def __init__(self):
        self._algorithms: Dict[str, Type[BaseLocalPlanner]] = {}
        self._instances: Dict[str, BaseLocalPlanner] = {}
        self._current_algorithm: Optional[str] = None
        
    def register_algorithm(self, name: str, algorithm_class: Type[BaseLocalPlanner]):
        """注册新的局部算法"""
        if not issubclass(algorithm_class, BaseLocalPlanner):
            raise ValueError(f"算法类必须继承自 BaseLocalPlanner")
        self._algorithms[name] = algorithm_class
        
    def get_algorithm(self, name: str) -> Optional[BaseLocalPlanner]:
        """获取算法实例"""
        if name not in self._algorithms:
            return None
            
        if name not in self._instances:
            self._instances[name] = self._algorithms[name]()
            
        return self._instances[name]
    
    def set_current_algorithm(self, name: str):
        """设置当前使用的算法"""
        if name not in self._algorithms and name != "无局部算法":
            raise ValueError(f"未注册的算法: {name}")
        self._current_algorithm = name
        
    def get_current_algorithm(self) -> Optional[str]:
        """获取当前算法名称"""
        return self._current_algorithm
    
    def list_algorithms(self) -> List[str]:
        """获取所有可用算法名称"""
        return list(self._algorithms.keys())
    
    def plan_local_path(self,
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       static_obstacles: List,
                       dynamic_obstacles: List,
                       velocity: Optional[np.ndarray] = None,
                       global_path: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        执行局部路径规划
        
        Args:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            static_obstacles: 静态障碍物列表（原始对象）
            dynamic_obstacles: 动态障碍物列表（原始对象，只包含新增的）
            velocity: 当前速度向量 [vx, vy]
            global_path: 全局路径点列表，用于前视目标点选择（可选）
            
        Returns:
            调整后的目标位置 [x, y]
        """
        if self._current_algorithm is None or self._current_algorithm == "无局部算法":
            return target_pos
            
        algorithm = self.get_algorithm(self._current_algorithm)
        if algorithm is None:
            return target_pos
            
        # 检查当前算法是否支持global_path参数
        import inspect
        algorithm_plan_params = inspect.signature(algorithm.plan).parameters
        
        # 根据算法是否支持global_path参数，选择不同的调用方式
        if 'global_path' in algorithm_plan_params:
            return algorithm.plan(current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity, global_path)
        else:
            # 对于不支持global_path参数的算法，不传递该参数
            return algorithm.plan(current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity)
    
    def is_algorithm_available(self, name: str) -> bool:
        """检查算法是否可用"""
        return name in self._algorithms