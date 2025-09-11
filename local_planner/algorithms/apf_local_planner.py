"""
APF局部路径规划算法实现

基于人工势场法的局部避障算法
"""

import numpy as np
from typing import List, Tuple, Optional
from ..base_local_planner import BaseLocalPlanner


class APFLocalPlanner(BaseLocalPlanner):
    """APF局部路径规划算法"""
    
    def __init__(self, config=None):
        """初始化APF本地规划器"""
        super().__init__(config)
        self.name = "APF算法"
        
        # 优化后的APF参数配置
        default_config = {
            'repulsive_gain': 400.0,      # 降低斥力强度：800→400
            'attractive_gain': 12.0,      # 增强引力：8→12
            'influence_distance': 50.0,  # 降低影响距离：200→180
            'safety_distance': 35.0,      # 降低安全距离：40→35
            'max_step': 10.0,             # 增加步长：8→10
            'damping_factor': 0.8,        # 降低阻尼：0.85→0.8
            'velocity_factor': 2.5,       # 降低速度因子：3→2.5
            'prediction_time': 5.0,       # 降低预测时间：6→5
            'emergency_factor': 2.5,      # 降低紧急因子：3→2.5
            'angular_factor': 5.0,        # 增强角度因子：4→5
            'lookahead_factor': 1.8,      # 降低前瞻因子：2→1.8
            'early_warning_factor': 3.5,  # 降低预警因子：4→3.5
            'proactive_factor': 5.0,      # 降低主动因子：6→5
            'min_distance': 2.0           # 最小距离阈值
        }
        
        if self.config is None:
            self.config = default_config
        else:
            self.config.update({k: v for k, v in default_config.items() if k not in self.config})
        self.config = {
            'repulsive_gain': 800.0,          # 降低斥力增益，避免过度反应
            'attractive_gain': 8,             # 增强引力，保持目标导向
            'influence_distance': 200.0,      # 适中影响距离，提前预警
            'safety_distance': 40.0,         # 适中安全距离
            'min_distance': 2.0,            # 最小安全距离
            'max_step': 8.0,                # 更大步长，快速响应
            'damping_factor': 0.85,          # 适中阻尼
            'velocity_factor': 3.0,         # 适中速度因子
            'prediction_time': 6.0,         # 适中预测时间
            'emergency_factor': 3.0,         # 适中紧急因子
            'angular_factor': 4.0,          # 增强角度因子
            'lookahead_factor': 2.0,          # 适中前瞻因子
            'early_warning_factor': 4.0,      # 适中提前预警
            'proactive_factor': 6.0         # 适中主动避障
        }
    
    def plan(self, current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity=None):
        """
        执行局部路径规划
        
        Args:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            velocity: 当前速度向量 [vx, vy]
            
        Returns:
            调整后的目标位置 [x, y]
        """
        # 🔥 调试信息：确认APF算法被调用
        print(f"🔥 APF算法被调用！当前位置: {current_pos}, 目标位置: {target_pos}, 动态障碍物数量: {len(dynamic_obstacles)}")
        
        # 转换为numpy数组
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        if velocity is None:
            velocity = np.zeros(2)
        else:
            velocity = np.array(velocity)
        
        # 保存当前速度用于预测计算
        self.current_velocity = velocity if velocity is not None else np.zeros(2)
        
        # 计算到目标的距离
        distance_to_target = np.linalg.norm(target_pos - current_pos)
        
        # 如果距离很小，直接返回目标点避免震荡
        if distance_to_target < self.config['min_distance']:
            print(f"⚠️ 距离太近({distance_to_target:.2f} < {self.config['min_distance']})，跳过斥力计算")
            return target_pos
            
        # 计算引力（朝向目标点）
        attractive_force = self._calculate_attractive_force(current_pos, target_pos)
        
        # 计算斥力（增强版）
        repulsive_force = self._calculate_enhanced_repulsive_force(current_pos, dynamic_obstacles)
        
        # 计算合力
        total_force = attractive_force + repulsive_force
        print("F_attr:", attractive_force, "F_rep:", repulsive_force, "F_total:", total_force)

        # 自适应步长：距离越近，步长越小
        if np.linalg.norm(total_force) > 0:
            # 根据距离调整步长
            adaptive_step = min(self.config['max_step'], 
                              max(0.5, distance_to_target * 0.1))
            
            # 添加阻尼因子减少震荡
            damping = self.config['damping_factor']
            
            adjusted_target = current_pos + total_force * adaptive_step * damping
            
            # 确保不会越过目标点
            direction_to_target = target_pos - current_pos
            if np.dot(adjusted_target - current_pos, direction_to_target) < 0:
                return target_pos
                
            return adjusted_target
        
        return target_pos
    
    def _calculate_attractive_force(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """计算引力"""
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # 限制引力大小，避免在接近目标时过大
            max_force = 2.0
            force_magnitude = min(self.config['attractive_gain'], max_force)
            return force_magnitude * direction / distance
        return np.zeros(2)
    
    def _calculate_enhanced_repulsive_force(self, current_pos: np.ndarray, obstacles: List) -> np.ndarray:
        """增强版斥力计算（多时间点预测+速度加权+角度优化）"""
        total_repulsive = np.zeros(2)
        ship_velocity = getattr(self, 'current_velocity', np.zeros(2))
        
        print(f"🎯 开始计算斥力，障碍物数量: {len(obstacles)}")
        
        for i, obstacle in enumerate(obstacles):
            # 处理DynamicObstacle对象
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                obs_pos = np.array(obstacle.position)
                obs_radius = obstacle.size
                obs_velocity = getattr(obstacle, 'velocity', np.zeros(2))
            else:
                # 兼容旧的元组格式
                if len(obstacle) >= 5:
                    obs_x, obs_y, obs_radius, vel_x, vel_y = obstacle[:5]
                    obs_pos = np.array([obs_x, obs_y])
                    obs_velocity = np.array([vel_x, vel_y])
                else:
                    obs_x, obs_y, obs_radius = obstacle
                    obs_pos = np.array([obs_x, obs_y])
                    obs_velocity = np.zeros(2)
            
            # 计算相对速度和相对位置
            relative_velocity = obs_velocity - ship_velocity
            relative_pos = obs_pos - current_pos
            current_distance = np.linalg.norm(relative_pos) - obs_radius
            
            print(f"📍 障碍物 {i+1}: 位置={obs_pos}, 半径={obs_radius}, 距离={current_distance:.2f}")
            
            # 存储障碍物速度供风险计算使用
            self.current_obs_velocity = obs_velocity
            
            # 预测碰撞时间和位置
            collision_time = self._predict_collision_time(current_pos, ship_velocity, obs_pos, obs_velocity)
            
            print(f"⏰ 碰撞时间: {collision_time:.2f}")
            
            # 超密集时间点预测，确保早期发现
            future_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                           1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 2.9, 3.2, 
                           3.5, 3.8, 4.1, 4.4, 4.7, 5.0]
            weighted_repulsive = np.zeros(2)
            total_weight = 0.0
            
            for time in future_times:
                # 计算未来预测位置
                future_pos = obs_pos + obs_velocity * time
                safe_radius = obs_radius + self.config['safety_distance']
                
                # 计算到预测位置的距离
                direction = current_pos - future_pos
                distance = np.linalg.norm(direction)
                min_distance = distance - safe_radius
                
                # 碰撞风险权重
                collision_risk = self._calculate_collision_risk(current_pos, future_pos, safe_radius, collision_time, time)
                
                # 速度影响权重
                velocity_magnitude = np.linalg.norm(relative_velocity)
                velocity_factor = 1.0 + (velocity_magnitude / 12.0) * self.config['velocity_factor']
                influence_distance = self.config['influence_distance'] * velocity_factor * self.config['lookahead_factor']
                
                print(f"🔍 时间={time:.1f}, 最小距离={min_distance:.2f}, 影响距离={influence_distance:.2f}, 风险={collision_risk:.2f}")
                
                # 检查是否满足斥力计算条件 - 即使船舶在障碍物内部也要触发斥力
                if min_distance < influence_distance:  # 移除min_distance > 0的限制
                    # 非线性斥力函数
                    influence_ratio = min_distance / influence_distance
                    strength = self.config['repulsive_gain'] * (1.0 - influence_ratio**2) * (1.0 + collision_risk)
                    
                    print(f"⚡ 触发斥力！强度={strength:.2f}")
                    
                    # 考虑角度的斥力方向调整
                    if np.linalg.norm(direction) > 0:
                        base_direction = direction / np.linalg.norm(direction)
                        
                        # 增强切向分量避免碰撞
                        tangent = np.array([-base_direction[1], base_direction[0]])
                        # 根据障碍物位置动态调整切向强度
                        if abs(base_direction[0]) > abs(base_direction[1]):  # 主要是水平避障
                            tangent_strength = 0.8 * strength  # 增强横向分量
                        else:  # 主要是垂直避障
                            tangent_strength = 0.8 * strength  # 增强纵向分量
                        
                        # 添加智能切向选择：根据船舶与障碍物的相对位置
                        relative_pos = future_pos - current_pos
                        if relative_pos[0] > 0:  # 障碍物在右侧
                            tangent_direction = np.array([-base_direction[1], base_direction[0]])
                        else:  # 障碍物在左侧
                            tangent_direction = np.array([base_direction[1], -base_direction[0]])
                        
                        # 组合斥力：更强的切向避障
                        rep_force = strength * base_direction + tangent_strength * tangent_direction
                        
                        weighted_repulsive += rep_force
                        total_weight += 1.0
                        
                        print(f"🎯 斥力向量: {rep_force}")
            
            # 平均加权斥力
            if total_weight > 0:
                weighted_repulsive /= total_weight
                total_repulsive += weighted_repulsive
                print(f"✅ 障碍物 {i+1} 最终斥力: {weighted_repulsive}")
            else:
                print(f"❌ 障碍物 {i+1} 未产生斥力（距离过远）")
        
        print(f"🚀 总斥力: {total_repulsive}")
        return total_repulsive
    
    def _predict_collision_time(self, ship_pos: np.ndarray, ship_vel: np.ndarray, 
                               obs_pos: np.ndarray, obs_vel: np.ndarray) -> float:
        """预测碰撞时间"""
        relative_pos = obs_pos - ship_pos
        relative_vel = obs_vel - ship_vel
        
        # 计算相对速度的平方
        vel_sq = np.dot(relative_vel, relative_vel)
        if vel_sq < 1e-6:  # 相对速度很小
            return -1.0
            
        # 计算相对位置与相对速度的点积
        pos_dot_vel = np.dot(relative_pos, relative_vel)
        
        # 计算最近距离的时间
        t_closest = -pos_dot_vel / vel_sq
        
        # 检查这个时间点是否在合理范围内
        if t_closest < 0 or t_closest > self.config['prediction_time']:
            return -1.0
            
        # 计算最近距离
        closest_pos = ship_pos + ship_vel * t_closest
        closest_obs_pos = obs_pos + obs_vel * t_closest
        closest_distance = np.linalg.norm(closest_pos - closest_obs_pos)
        
        # 如果最近距离大于安全距离，不会碰撞
        if closest_distance > self.config['safety_distance'] + 10.0:
            return -1.0
            
        return max(0.0, t_closest)
    
    def _calculate_collision_risk(self, ship_pos: np.ndarray, obs_pos: np.ndarray, 
                                 safe_radius: float, collision_time: float, 
                                 prediction_time: float) -> float:
        """计算碰撞风险系数（增强提前预警）"""
        distance = np.linalg.norm(ship_pos - obs_pos) - safe_radius
        if distance <= 0:
            return 5.0  # 已经碰撞，最高风险
            
        # 极端扩展风险计算范围，超早触发避障
        extended_safe_radius = safe_radius * 4.0  # 超大检测范围，提前150像素开始避障
        
        # 超敏感基础风险计算
        if distance <= extended_safe_radius:
            # 超早期风险检测，在更远距离就开始响应
            risk_ratio = max(0.0, (extended_safe_radius - distance) / extended_safe_radius)
            # 超敏感非线性放大
            base_risk = risk_ratio ** 0.3  # 更激进的早期风险放大
            # 距离越近，风险增长越快
            if distance < safe_radius * 1.5:
                base_risk *= 2.5  # 近距离风险倍增
        else:
            base_risk = 0.0
        
        # 时间风险（更早触发）
        time_risk = 0.0
        if collision_time > 0:
            # 更早的时间预警
            early_warning_time = min(collision_time, self.config['prediction_time'] * 1.2)
            time_risk = max(0.0, 1.0 - early_warning_time / (self.config['prediction_time'] * 1.5))
        
        # 速度风险（基于相对速度）
        ship_velocity = getattr(self, 'current_velocity', np.zeros(2))
        obs_velocity = getattr(self, 'current_obs_velocity', np.zeros(2))
        relative_velocity = obs_velocity - ship_velocity
        velocity_risk = min(2.0, np.linalg.norm(relative_velocity) / 20.0)
        
        # 超强组合风险（极端早期预警）
        total_risk = base_risk * 4.0 + time_risk * 3.5 + velocity_risk * 1.0
        # 主动避障增强
        proactive_boost = self.config['proactive_factor'] * max(0, 1.0 - distance / (safe_radius * 3.0))
        total_risk += proactive_boost
        return min(8.0, total_risk)
    
    def _calculate_angle_weight(self, ship_pos: np.ndarray, ship_vel: np.ndarray,
                               obs_pos: np.ndarray, obs_vel: np.ndarray) -> float:
        """计算角度权重，考虑运动方向的相对角度"""
        if np.linalg.norm(ship_vel) < 1e-6:
            return 1.0
            
        # 船舶到障碍物的方向
        to_obstacle = obs_pos - ship_pos
        if np.linalg.norm(to_obstacle) < 1e-6:
            return 1.0
            
        # 归一化方向向量
        ship_dir = ship_vel / np.linalg.norm(ship_vel)
        obs_dir = to_obstacle / np.linalg.norm(to_obstacle)
        
        # 计算角度余弦值
        cos_angle = np.dot(ship_dir, obs_dir)
        
        # 如果障碍物在正前方，权重更高
        if cos_angle > 0.7:  # 约45度以内
            return 2.0
        elif cos_angle > 0:  # 前侧方
            return 1.5
        elif cos_angle > -0.7:  # 后侧方
            return 1.0
        else:  # 正后方
            return 0.5
    
    def is_applicable(self, obstacle_type: str) -> bool:
        """APF算法只适用于动态障碍物"""
        return obstacle_type == 'dynamic'