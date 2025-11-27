import numpy as np
from typing import List, Optional
from ..base_local_planner import BaseLocalPlanner


class APFLocalPlannerSimple(BaseLocalPlanner):
    """
    简化版APF局部路径规划器（原始人工势场法）
    仅保留基本的引力和斥力计算，无威胁指数和切向力改进
    """
    
    def __init__(self, name="人工势场法"):
        """
        初始化规划器配置
        
        Args:
            name: 算法名称
        """
        # 调用基类构造函数
        super().__init__(name)
        
        # 默认配置
        self.config = {
            # 引力参数
            'attractive_gain': 8.0,        # 引力增益系数
            'min_distance': 5.0,           # 到达目标的最小距离阈值
            
            # 斥力参数
            'repulsive_gain': 800.0,       # 斥力增益系数
            'safety_distance': 30.0,       # 安全距离
            'influence_distance': 100.0,    # 斥力影响距离
            
            # 移动参数
            'max_step': 15.0,              # 最大步长
        }
    
    def plan(self, 
             current_pos: np.ndarray,
             target_pos: np.ndarray,
             static_obstacles: List,
             dynamic_obstacles: List,
             velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        简化版APF路径规划（原始人工势场法）
        仅计算基本的引力和斥力，无威胁指数和切向力改进
        
        Args:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            static_obstacles: 静态障碍物列表（原始对象，不参与局部避障）
            dynamic_obstacles: 动态障碍物列表（原始对象，只包含新增的）
            velocity: 当前速度向量 [vx, vy]
            
        Returns:
            调整后的目标位置 [x, y]
        """
        # 转换为numpy数组
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 计算到目标的距离
        distance_to_target = np.linalg.norm(target_pos - current_pos)
        
        # 如果距离很小，直接返回目标点
        if distance_to_target < self.config['min_distance']:
            return target_pos
        
        # 1. 计算引力
        attractive_force = self._calculate_attractive_force(current_pos, target_pos)
        
        # 2. 计算斥力
        repulsive_force = self._calculate_repulsive_force(current_pos, static_obstacles, dynamic_obstacles)
        
        # 3. 计算合力
        total_force = attractive_force + repulsive_force
        
        # 4. 计算下一步位置
        if np.linalg.norm(total_force) > 0:
            direction = total_force / np.linalg.norm(total_force)
            step_size = min(self.config['max_step'], distance_to_target)
            next_pos = current_pos + direction * step_size
        else:
            # 无合力时，直接向目标点移动
            direction = (target_pos - current_pos) / distance_to_target
            next_pos = current_pos + direction * min(self.config['max_step'], distance_to_target)
        
        return next_pos
    
    def _calculate_attractive_force(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        计算引力
        
        Args:
            current_pos: 当前位置
            target_pos: 目标位置
            
        Returns:
            引力向量
        """
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            force_magnitude = self.config['attractive_gain']
            return force_magnitude * direction / distance
        return np.zeros(2)
    
    def _calculate_repulsive_force(self, current_pos: np.ndarray, static_obstacles: List, dynamic_obstacles: List) -> np.ndarray:
        """
        计算所有障碍物的斥力
        
        Args:
            current_pos: 当前位置
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            
        Returns:
            总斥力向量
        """
        total_repulsive = np.zeros(2)
        
        # 处理所有障碍物（静态和动态）
        all_obstacles = static_obstacles + dynamic_obstacles
        
        for obstacle in all_obstacles:
            try:
                # 尝试从对象格式获取障碍物信息
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    # 处理position可能是列表或元组的情况
                    pos = obstacle.position
                    if isinstance(pos, (list, tuple)):
                        obs_pos = np.array([float(pos[0]), float(pos[1])])
                    else:
                        obs_pos = np.array(pos)
                    
                    # 处理size可能是列表或元组的情况
                    size_val = obstacle.size
                    if isinstance(size_val, (list, tuple)):
                        obs_radius = float(size_val[0]) if len(size_val) > 0 else 10.0
                    else:
                        obs_radius = float(size_val)
                    
                    print(f"成功处理障碍物: 位置={obs_pos}, 半径={obs_radius}")
                else:
                    # 从数组格式获取障碍物信息 [x, y, radius]
                    obstacle_list = list(obstacle)
                    obs_pos = np.array([float(obstacle_list[0]), float(obstacle_list[1])])
                    obs_radius = float(obstacle_list[2]) if len(obstacle_list) > 2 else 10.0
                
                # 计算单个障碍物的斥力
                rep_force = self._calculate_single_obstacle_repulsion(current_pos, obs_pos, obs_radius)
                total_repulsive += rep_force
                
            except Exception as e:
                print(f"障碍物数据格式错误: {e}")
                print(f"障碍物类型: {type(obstacle)}")
                print(f"障碍物属性: {dir(obstacle) if hasattr(obstacle, '__dict__') else 'N/A'}")
                if hasattr(obstacle, 'position'):
                    print(f"position类型: {type(obstacle.position)}, 值: {obstacle.position}")
                if hasattr(obstacle, 'size'):
                    print(f"size类型: {type(obstacle.size)}, 值: {obstacle.size}")
                continue
        
        return total_repulsive
    
    def _calculate_single_obstacle_repulsion(self, current_pos: np.ndarray, obs_pos: np.ndarray, obs_radius: float) -> np.ndarray:
        """
        计算单个障碍物的斥力
        
        Args:
            current_pos: 当前位置
            obs_pos: 障碍物位置
            obs_radius: 障碍物半径
            
        Returns:
            斥力向量
        """
        direction = current_pos - obs_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            # 避免除零
            return np.array([1.0, 0.0])
        
        # 安全距离 = 障碍物半径 + 配置安全距离
        safe_distance = obs_radius + self.config['safety_distance']
        
        # 斥力影响范围
        max_influence_distance = safe_distance + self.config['influence_distance']
        
        if distance < safe_distance:
            # 在安全距离内，斥力随距离减小而增大
            force_magnitude = self.config['repulsive_gain'] * (1.0 / distance - 1.0 / safe_distance) / (distance ** 2)
            return direction / distance * force_magnitude
        elif distance < max_influence_distance:
            # 在影响范围内，斥力随距离增大而衰减
            decay_factor = (max_influence_distance - distance) / (max_influence_distance - safe_distance)
            force_magnitude = self.config['repulsive_gain'] * decay_factor / (distance ** 2)
            return direction / distance * force_magnitude
        else:
            # 超出影响范围，无斥力
            return np.zeros(2)