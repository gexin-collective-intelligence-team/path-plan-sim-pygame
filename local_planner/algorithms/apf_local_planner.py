"""
APF局部路径规划算法实现

基于人工势场法的局部避障算法
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from ..base_local_planner import BaseLocalPlanner


class APFLocalPlanner(BaseLocalPlanner):
    """APF局部路径规划算法"""
    
    def __init__(self, config=None):
        """初始化APF本地规划器（抗振荡优化版）"""
        super().__init__(config)
        self.name = "APF算法"
        
        # 智能动态避障APF参数配置 - 抗震荡优化版（解决循环和碰撞问题）
        self.config = {
            'repulsive_gain': 800.0,         # 优化：1500→800（降低斥力，减少过度反应）
            'attractive_gain': 8.0,          # 优化：4.0→8.0（增加引力，更好地引导向目标）
            'influence_distance': 120.0,     # 优化：150→120（缩小影响范围，减少不必要的避障）
            'safety_distance': 15.0,         # 优化：25→15（减小安全距离，缩小碰撞半径）
            'min_distance': 3.0,             # 保持：3.0（目标接近阈值）
            'max_step': 8.0,                 # 优化：10.0→8.0（减小步长，增加稳定性）
            'damping_factor': 0.95,          # 优化：0.9→0.95（增加阻尼，减少震荡）
            'velocity_factor': 1.2,          # 优化：1.5→1.2（降低速度对斥力的影响）
            'prediction_time': 2.5,          # 优化：3.5→2.5（减少预测时间，避免过度反应）
            'angular_factor': 1.2,           # 优化：1.5→1.2（降低角度敏感度）
            'lookahead_factor': 1.2,         # 优化：1.5→1.2（调整前瞻因子）
            'early_warning_factor': 1.2,     # 优化：1.5→1.2（降低预警敏感度）
            'proactive_factor': 1.5,         # 优化：2.0→1.5（降低主动避障强度）
            'history_weight': 0.3,           # 优化：0.2→0.3（增加历史权重，提高稳定性）
            'noise_filter': 0.7,             # 低通滤波器系数，用于斥力平滑过渡
            'escape_threshold': 5,           # 优化：8→5（减少循环检测步数，更早触发逃离）
            'escape_force': 120.0,           # 优化：80→120（增加逃离力，确保能跳出循环）
            'random_factor': 0.04,           # 优化：0.03→0.04（增加随机扰动，增加逃离多样性）
            'max_angular_velocity': np.pi/3, # 优化：π/2→π/3（减小最大角速度至60度/秒，增加稳定性）
            'lookahead_distance': 60.0,      # 调整：100.0→60.0（减小前视距离，避免震荡）
            'lookahead_angle_threshold': np.pi/6  # 新增：前视角度阈值，用于过滤偏离当前方向的路径点
        }
        
        # 初始化抗振荡状态变量
        self.position_history = deque(maxlen=20)  # 存储最近的位置历史
        self.force_history = deque(maxlen=20)     # 存储最近的合力历史
        self.target_pos = None                   # 当前目标位置，用于循环检测
        self.current_velocity = np.zeros(2)      # 当前速度向量
        self.is_looping = False                  # 循环状态标志
        self._last_escape_direction = np.array([1.0, 0.0])  # 初始化逃离方向，避免重复逃离
        
        # 初始化前视目标点选择相关变量
        self.global_path = None          # 全局路径缓存
        self.current_path_index = 0      # 当前路径索引
        
        # 抗振荡状态变量
        self.previous_force = np.zeros(2)
        self.force_history = []  # 记录历史力向量
        
        # 状态跟踪（用于检测和解决循环）
        self.position_history = []
        self.max_history_length = 10
        self.history_length = 10  # 添加历史记录长度变量
        self.escape_mode = False
        self.escape_direction = None
        self.collision_occurred = False  # 添加碰撞标志
        
        # 路径相关状态变量
        self.global_path = None  # 全局路径缓存
        self.current_path_index = 0  # 当前路径点索引
    
    def _update_position_history(self, current_pos: np.ndarray):
        """更新位置历史，用于循环检测"""
        # 确保position_history是deque对象
        if not isinstance(self.position_history, deque):
            self.position_history = deque(self.position_history, maxlen=15)
        
        self.position_history.append(current_pos.copy())
        # 使用deque的popleft方法而不是pop(0)
        max_history = getattr(self, 'max_history_length', 15)
        if len(self.position_history) > max_history:
            self.position_history.popleft()
    
    def _detect_loop(self) -> bool:
        """增强循环检测：提高敏感度，更早发现循环行为"""
        # 如果历史数据不足，无法检测循环
        if len(self.position_history) < 5:  # 进一步减少阈值，更早检测循环
            return False
        
        # 1. 检查位置是否在一个极小范围内循环（使用最近的几个点）
        recent_positions = np.array(list(self.position_history)[-5:])  # 最近5个位置
        max_distance = np.max(np.linalg.norm(recent_positions - np.mean(recent_positions, axis=0), axis=1))
        
        # 提高循环检测敏感度：更小的阈值
        is_position_circling = max_distance < 3.5  # 进一步减小阈值至3.5
        
        # 2. 检查是否在两个点之间反复移动（最常见的循环形式）
        is_point_cycling = False
        if len(self.position_history) >= 6:  # 降低要求，更容易检测
            # 检查最近的位置是否与之前的位置重复
            for i in range(len(self.position_history) - 3):
                # 检查是否回到了几步前的位置，使用更小的距离阈值
                if np.linalg.norm(np.array(self.position_history[-1]) - np.array(self.position_history[i])) < 2.0:
                    # 再检查是否也接近另一个点
                    for j in range(min(i+1, len(self.position_history) - 2), len(self.position_history) - 1):
                        if j != i and np.linalg.norm(np.array(self.position_history[-2]) - np.array(self.position_history[j])) < 2.0:
                            is_point_cycling = True
                            print(f"🔄 检测到两点间循环: 最近点与{i}步前点距离<2.0, 次近点与{j}步前点距离<2.0")
                            break
                    if is_point_cycling:
                        break
        
        # 3. 检查连续位置变化趋势
        is_stopping = False
        if len(self.position_history) >= 4:
            # 计算最近几次移动的距离
            move_distances = []
            for i in range(1, len(self.position_history)):
                move_dist = np.linalg.norm(np.array(self.position_history[i]) - np.array(self.position_history[i-1]))
                move_distances.append(move_dist)
            
            # 如果最近几次移动距离都很小，认为可能在停止或循环
            recent_moves = move_distances[-3:] if len(move_distances) >= 3 else move_distances
            if len(recent_moves) >= 3:
                avg_move = np.mean(recent_moves)
                if avg_move < 1.0:  # 平均移动距离小于1.0认为可能在停止或循环
                    is_stopping = True
                    print(f"🔄 检测到移动停滞: 平均移动距离={avg_move:.2f}")
        
        # 4. 检查合力是否在一个合理范围内但船舶没有前进
        force_check = False
        if len(self.force_history) >= 4:  # 降低要求
            recent_forces = np.array(list(self.force_history)[-4:])
            avg_force_magnitude = np.mean(np.linalg.norm(recent_forces, axis=1))
            
            # 放宽合力范围判断
            if 2.0 < avg_force_magnitude < 15.0:  # 更宽的范围
                force_check = True
        
        # 综合判断：更敏感的循环检测条件
        # 位置环绕+力检查 或者 点循环 或者 移动停滞，任一条件满足即认为在循环
        is_looping = (is_position_circling and force_check) or is_point_cycling or is_stopping
        
        # 打印检测结果
        if is_looping:
            print(f"🔄 循环检测: 检测到循环行为! 位置变化范围={max_distance:.1f}, "
                  f"环绕状态={is_position_circling}, 点循环={is_point_cycling}, "
                  f"移动停滞={is_stopping}")
        else:
            print(f"🔍 循环检测: 正常状态, 位置变化范围={max_distance:.1f}")
        
        return is_looping

    def _calculate_escape_force(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """优化逃离力：更智能的逃离方向选择，避免长时间循环"""
        # 计算从当前位置到目标位置的向量
        to_target = target_pos - current_pos
        distance_to_target = np.linalg.norm(to_target)
        
        # 记录目标位置（供循环检测使用）
        self.target_pos = target_pos.copy()
        
        # 确保_last_escape_direction存在且已初始化
        if not hasattr(self, '_last_escape_direction') or self._last_escape_direction is None:
            self._last_escape_direction = np.array([1.0, 0.0])
        
        if distance_to_target > 0:
            # 目标方向的垂直方向
            to_target_normalized = to_target / distance_to_target
            
            # 改进：随机选择左侧或右侧的垂直方向，增加逃离的多样性
            if np.random.random() > 0.5:
                escape_dir = np.array([-to_target_normalized[1], to_target_normalized[0]])  # 垂直于目标方向（左侧）
            else:
                escape_dir = np.array([to_target_normalized[1], -to_target_normalized[0]])  # 垂直于目标方向（右侧）
            
            # 如果之前有逃离方向，尝试不同方向
            if self._last_escape_direction is not None:
                # 计算与上次逃离方向的夹角
                angle_diff = np.arccos(np.clip(np.dot(escape_dir, self._last_escape_direction), -1.0, 1.0))
                # 如果夹角小于60度，尝试相反方向
                if angle_diff < np.pi/3:
                    escape_dir = -escape_dir
                    print("🔄 调整逃离方向：尝试相反方向")
        else:
            # 如果已经在目标位置，随机选择一个方向
            angle = np.random.uniform(0, 2*np.pi)
            escape_dir = np.array([np.cos(angle), np.sin(angle)])
        
        # 增强逃离力效果：增加随机扰动并更偏向目标方向
        random_perturbation = np.random.normal(0, self.config['random_factor'] * 2, 2)  # 增加扰动幅度
        # 逃离方向 = 垂直方向 + 扰动 + 30%目标方向（更偏向目标）
        if distance_to_target > 0:
            escape_dir = escape_dir + random_perturbation + 0.3 * to_target_normalized
        else:
            escape_dir = escape_dir + random_perturbation
        
        escape_dir = escape_dir / np.linalg.norm(escape_dir)
        
        # 记录逃离方向
        self._last_escape_direction = escape_dir.copy()
        
        # 改进：使用更激进的逃离力设置，确保能够有效跳出循环
        # 循环状态下始终使用较大的逃离力
        adjusted_escape_force = self.config['escape_force'] * 1.2  # 直接使用较大的逃离力
        
        print(f"🚀 生成增强逃离力: 大小={adjusted_escape_force:.1f}, 方向={escape_dir}")
        
        return escape_dir * adjusted_escape_force

    def plan(self, current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity=None, global_path=None):
        """
        统一版APF路径规划（使用增强斥力计算）
        
        Args:
            current_pos: 当前位置 [x, y]
            target_pos: 目标位置 [x, y]
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            velocity: 当前速度向量 [vx, vy]
            global_path: 全局路径点列表，用于前视目标点选择
            
        Returns:
            调整后的目标位置 [x, y]
        """
        # 转换为numpy数组
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 初始化状态变量
        if not hasattr(self, 'position_history') or not isinstance(self.position_history, deque):
            self.position_history = deque(maxlen=15)
        if not hasattr(self, 'force_history') or not isinstance(self.force_history, deque):
            self.force_history = deque(maxlen=15)
        
        # 更新位置历史
        self._update_position_history(current_pos)
        
        # 新增：使用前视距离选择动态局部目标点
        if global_path is not None and len(global_path) > 0:
            # 在全局路径上查找前视距离内最远的可行局部目标点
            dynamic_local_target = self.find_lookahead_waypoint(
                current_pos, global_path, static_obstacles, dynamic_obstacles, velocity
            )
            # 使用动态选择的局部目标点替代原始目标点
            local_target = dynamic_local_target
        else:
            # 如果没有全局路径，使用原始目标点
            local_target = target_pos
        
        # 存储当前速度供增强版斥力使用
        self.current_velocity = np.array(velocity) if velocity is not None else np.zeros(2)
        current_speed = np.linalg.norm(self.current_velocity)
        
        # 计算到目标的距离（使用局部目标点）
        distance_to_target = np.linalg.norm(local_target - current_pos)
        
        # 如果距离很小，直接返回目标点
        if distance_to_target < self.config['min_distance']:
            # 重置循环状态
            self.position_history.clear()
            self.force_history.clear()
            return target_pos
        
        # 增强版循环检测：检查是否在两个点之间反复移动
        is_looping = self._detect_loop()
        # 保存循环状态供find_lookahead_waypoint方法使用
        self.is_looping = is_looping
        
        # 新增：简单的位置循环检测（检查是否回到了最近访问过的位置）
        if len(self.position_history) > 4:  # 降低阈值，更早检测
            for prev_pos in list(self.position_history)[-4:]:
                if np.linalg.norm(current_pos - prev_pos) < 2.5:  # 降低距离阈值，提高敏感度
                    is_looping = True
                    print("⚠️ 检测到位置重复，疑似循环")
                    break
        
        # 1. 计算各力（引力+增强斥力+边界斥力+紧急避障力）
        # 使用动态选择的局部目标点计算引力
        attractive_force = self._calculate_attractive_force(current_pos, local_target)
        
        # 修复：直接调用实时斥力计算方法，因为之前的方法名错误
        repulsive_force = self._calculate_realtime_repulsion(current_pos, static_obstacles, dynamic_obstacles)
        
        boundary_force = self._calculate_boundary_repulsion(current_pos)
        
        # 新增：计算紧急避障力
        emergency_avoidance_force = self._calculate_emergency_avoidance_force(current_pos, static_obstacles + dynamic_obstacles)
        
        # 新增：检查是否发生碰撞，如果碰撞则停止运动
        if self._check_collision(current_pos, static_obstacles + dynamic_obstacles):
            print("🚨 碰撞检测：检测到与障碍物碰撞！停止运动")
            # 设置标志表示已碰撞
            self.collision_occurred = True
            # 清除所有力，确保完全停止
            self.attractive_force_history = deque(maxlen=self.history_length)
            self.repulsive_force_history = deque(maxlen=self.history_length)
            return current_pos  # 返回当前位置表示停止
        
        # 2. 计算合力（加入循环逃离力）
        total_force = attractive_force + repulsive_force + boundary_force + emergency_avoidance_force
        
        # 记录力历史（用于循环检测）
        # 确保force_history是deque对象
        if not isinstance(self.force_history, deque):
            self.force_history = deque(self.force_history, maxlen=15)
        self.force_history.append(total_force.copy())
        if len(self.force_history) > self.config.get('max_history_length', 15):
            # 由于deque有maxlen限制，通常不需要手动popleft，但保留以确保
            if hasattr(self.force_history, 'popleft'):
                self.force_history.popleft()
        
        # 若检测到循环，添加逃离力
        if is_looping:
            # 设置循环标志
            self.is_looping = True
            # 使用智能逃离策略：调用专门的逃离力计算方法
            escape_force = self._calculate_escape_force(current_pos, local_target)
            # 增强逃离力效果：在循环情况下使用更大的逃离力
            enhanced_escape_force = escape_force * 1.5  # 增加50%的逃离力
            total_force += enhanced_escape_force
            print(f"🔄 检测到循环，添加增强逃离力: {np.linalg.norm(enhanced_escape_force):.1f}")
            
            # 重置历史记录，避免重复检测同一循环
            self.position_history = deque(list(self.position_history)[-3:], maxlen=20)
        else:
            # 清除循环标志
            self.is_looping = False
        
        # 3. 计算下一步位置（动态步长+转向约束）
        if np.linalg.norm(total_force) > 0:
            direction = total_force / np.linalg.norm(total_force)
            
            # 打印合力方向和大小（用于调试）
            force_magnitude = np.linalg.norm(total_force)
            print(f"🧭 当前合力方向: {direction}, 合力大小: {force_magnitude:.1f}")
            print(f"引力大小: {np.linalg.norm(attractive_force):.1f}, 斥力大小: {np.linalg.norm(repulsive_force):.1f}")
            
            # 优化步长计算：根据状态调整步长
            if is_looping:
                # 在循环状态下使用更大的步长跳出循环
                step_size = min(self.config['max_step'] * 1.5, distance_to_target * 0.4)
                print(f"🔄 循环状态下使用更大步长: {step_size:.2f}")
            else:
                # 正常状态下使用稳定步长
                step_size = min(self.config['max_step'] * 1.2, distance_to_target * 0.3)
            
            # 转向约束：简化版，避免过度复杂的计算
            if current_speed > 1e-6:
                # 简化的转向限制
                max_turn_angle = self.config['max_angular_velocity'] * 0.1  # 假设0.1秒时间步
                
                # 计算当前方向与目标方向的角度差
                current_angle = np.arctan2(self.current_velocity[1], self.current_velocity[0])
                target_angle = np.arctan2(direction[1], direction[0])
                angle_diff = target_angle - current_angle
                
                # 处理角度跨越±π的情况
                angle_diff = angle_diff - 2 * np.pi if angle_diff > np.pi else angle_diff
                angle_diff = angle_diff + 2 * np.pi if angle_diff < -np.pi else angle_diff
                
                # 应用转向限制
                limited_angle_diff = np.clip(angle_diff, -max_turn_angle, max_turn_angle)
                limited_angle = current_angle + limited_angle_diff
                direction = np.array([np.cos(limited_angle), np.sin(limited_angle)])
                
                print(f"🔄 应用转向约束: 当前角度={np.degrees(current_angle):.1f}°, 目标角度={np.degrees(target_angle):.1f}°, 限制后角度={np.degrees(limited_angle):.1f}°")
            
            # 计算下一位置
            next_pos = current_pos + direction * step_size
            
            # 打印位置变化信息
            move_distance = np.linalg.norm(next_pos - current_pos)
            print(f"📍 当前位置: {current_pos}, 下一位置: {next_pos}, 移动距离: {move_distance:.2f}")
            print(f"📏 步长: {step_size:.2f}")
        else:
            # 无合力时，直接向局部目标点移动
            direction = (local_target - current_pos) / distance_to_target
            next_pos = current_pos + direction * min(self.config['max_step'], distance_to_target)
        
        # 4. 应用边界约束
        next_pos = self._apply_boundary_constraints(next_pos)
        
        return next_pos
    
    def _calculate_attractive_force(self, current_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """计算引力（修复：进一步降低引力，避免主导合力方向）"""
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # 距离自适应引力优化：极远距离引力更小，近距离引力适当增加
            if distance > 100:  # 极远距离
                force_magnitude = self.config['attractive_gain'] * 0.3  # 引力降低至30%
            elif distance > 50:  # 远距离
                force_magnitude = self.config['attractive_gain'] * 0.5  # 引力降低至50%
            elif distance < 20:  # 近距离
                force_magnitude = self.config['attractive_gain'] * 1.5  # 引力增加至150%
            else:  # 中等距离
                force_magnitude = self.config['attractive_gain']
            return force_magnitude * direction / distance
        return np.zeros(2)
    
    def _calculate_collision_time(self, ship_pos: np.ndarray, ship_vel: np.ndarray, 
                                 obs_pos: np.ndarray, obs_vel: np.ndarray, obs_radius: float) -> float:
        """计算碰撞时间预测"""
        relative_pos = obs_pos - ship_pos
        relative_vel = obs_vel - ship_vel
        
        # 计算相对速度的平方
        vel_sq = np.dot(relative_vel, relative_vel)
        
        # 放宽相对速度阈值，更好地检测低速接近
        if vel_sq < 0.1:  # 从1e-6增加到0.1，更好地检测低速接近
            # 即使相对速度很小，如果距离很近也应视为潜在碰撞
            current_distance = np.linalg.norm(relative_pos)
            # 安全距离（增加船舶半径的考虑）
            safe_distance = self.config['safety_distance'] + 20.0  # 障碍物半径通常在代码其他部分处理
            if current_distance < safe_distance * 1.5:  # 较近范围内的低速目标也需要考虑
                return 0.5  # 返回一个较短的预测时间，表示需要立即响应
            return -1.0
            
        # 计算相对位置与相对速度的点积
        pos_dot_vel = np.dot(relative_pos, relative_vel)
        
        # 计算最近距离的时间
        t_closest = -pos_dot_vel / vel_sq
        
        # 扩展预测时间窗口，提前更多时间检测碰撞
        max_pred_time = self.config['prediction_time'] * 1.5  # 扩展50%的预测时间
        if t_closest < 0 or t_closest > max_pred_time:
            # 检查是否在未来较短时间内（如2秒）有碰撞风险
            short_term_pred = 2.0
            if t_closest < short_term_pred and pos_dot_vel < 0:  # 正在接近
                return t_closest if t_closest > 0 else 0.1  # 确保返回正值
            return -1.0
            
        # 计算最近距离
        closest_pos = ship_pos + ship_vel * t_closest
        closest_obs_pos = obs_pos + obs_vel * t_closest
        closest_distance = np.linalg.norm(closest_pos - closest_obs_pos)
        
        # 更严格的安全距离判断，考虑船舶自身大小
        # 安全距离 = 配置安全距离 + 额外余量
        required_safety_distance = self.config['safety_distance'] + 15.0
        
        # 如果最近距离小于安全距离，返回预测时间
        if closest_distance < required_safety_distance:
            # 额外检查：即使最近距离足够，但如果两者运动方向夹角很小也应视为风险
            relative_dir = relative_pos / np.linalg.norm(relative_pos) if np.linalg.norm(relative_pos) > 0 else np.zeros(2)
            vel_dir = relative_vel / np.linalg.norm(relative_vel)
            angle_cos = np.dot(relative_dir, vel_dir)
            
            # 如果夹角小于45度（cos>0.7），说明是正面接近，风险更高
            if angle_cos > 0.7 and pos_dot_vel < 0:  # pos_dot_vel<0表示正在接近
                return min(t_closest, max_pred_time)
            
            return max(0.0, t_closest)
        
        # 额外检查：即使最近距离足够，但如果当前距离已经非常近也应视为风险
        current_distance = np.linalg.norm(relative_pos)
        if current_distance < required_safety_distance * 1.2:  # 当前距离较近
            return 0.1  # 返回一个很小的时间值，表示需要立即响应
        
        return -1.0
        
    def _calculate_collision_risk(self, ship_pos: np.ndarray, obs_pos: np.ndarray, 
                                 safe_radius: float, collision_time: float, 
                                 prediction_time: float) -> float:
        """计算碰撞风险系数（增强版：优化碰撞风险评估和响应）"""
        distance = np.linalg.norm(ship_pos - obs_pos) - safe_radius
        if distance <= 0:
            return 3.0  # 已经碰撞，提高风险等级（从2.0→3.0）
            
        # 扩展风险计算范围，更早触发避障
        extended_safe_radius = safe_radius * 3.0  # 从2.5→3.0，进一步扩大检测范围
        
        # 更敏感的基础风险计算
        if distance <= extended_safe_radius:
            # 超早期风险检测，在更远距离就开始响应
            risk_ratio = max(0.0, (extended_safe_radius - distance) / extended_safe_radius)
            # 更激进的早期风险放大
            base_risk = risk_ratio ** 0.3  # 从0.5→0.3，使风险增长更快
            # 距离越近，风险增长越快
            if distance < safe_radius * 1.5:
                base_risk *= 2.0  # 从1.5→2.0，大幅提高近距离风险
        else:
            base_risk = 0.0
        
        # 时间风险（优化：即使碰撞时间为负，也考虑当前接近状态）
        time_risk = 0.0
        if collision_time > 0:
            # 更早的时间预警
            early_warning_time = min(collision_time, self.config['prediction_time'] * 1.2)
            time_risk = max(0.0, 1.0 - early_warning_time / (self.config['prediction_time'] * 1.5)) * 2.0  # 从1.5→2.0
        else:
            # 对于碰撞时间为负的情况，检查是否正在接近
            ship_velocity = getattr(self, 'current_velocity', np.zeros(2))
            obs_velocity = getattr(self, 'current_obs_velocity', np.zeros(2))
            relative_vel = obs_velocity - ship_velocity
            relative_pos = obs_pos - ship_pos
            
            # 计算相对运动方向与位置方向的夹角
            if np.linalg.norm(relative_pos) > 0 and np.linalg.norm(relative_vel) > 0:
                pos_dir = relative_pos / np.linalg.norm(relative_pos)
                vel_dir = relative_vel / np.linalg.norm(relative_vel)
                dot_product = np.dot(pos_dir, vel_dir)
                
                # 如果点积为负，表示两者正在靠近
                if dot_product < 0 and distance < extended_safe_radius:
                    # 根据距离和接近速度计算时间风险
                    approach_speed = np.linalg.norm(relative_vel) * abs(dot_product)
                    if approach_speed > 0.5:  # 有明显的接近速度
                        time_risk = min(1.5, (extended_safe_radius - distance) / (approach_speed * 2))
        
        # 恢复速度风险上限（从1.0→1.5），提高速度对斥力的影响
        ship_velocity = getattr(self, 'current_velocity', np.zeros(2))
        obs_velocity = getattr(self, 'current_obs_velocity', np.zeros(2))
        relative_velocity = obs_velocity - ship_velocity
        velocity_risk = min(1.5, np.linalg.norm(relative_velocity) / 15.0)  # 从20→15，对速度更敏感
        
        # 提高组合风险放大倍数（原1.5→2.0，1.0→1.5）
        total_risk = base_risk * 2.0 + time_risk * 1.5 + velocity_risk * 1.5
        # 提高主动避障增强（从1.5→2.0）
        proactive_boost = self.config['proactive_factor'] * max(0, 1.0 - distance / (safe_radius * 3.0)) * 2.0
        total_risk += proactive_boost
        
        # 提高风险上限（从3.0→5.0）
        return min(5.0, total_risk)
    
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

    def _calculate_realtime_repulsion(self, current_pos: np.ndarray, static_obstacles: List, dynamic_obstacles: List) -> np.ndarray:
        """实时计算所有障碍物的斥力"""
        total_repulsive = np.zeros(2)
        
        # 增加动态障碍物数量日志
        print(f"动态障碍物数量: {len(dynamic_obstacles)}")
        
        # 处理动态障碍物
        for i, obstacle in enumerate(dynamic_obstacles):
            try:
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    obs_pos = np.array(obstacle.position)
                    obs_radius = float(obstacle.size) if hasattr(obstacle.size, '__float__') else 10.0
                    # 修复：正确获取动态障碍物的速度信息（从direction和speed属性计算）
                    if hasattr(obstacle, 'direction') and hasattr(obstacle, 'speed'):
                        direction = np.array(obstacle.direction)
                        speed = float(obstacle.speed)
                        obs_velocity = direction * speed
                        print(f"动态障碍物 {i+1}: 位置={obs_pos}, 速度向量={obs_velocity}, 速度大小={np.linalg.norm(obs_velocity):.2f}")
                    else:
                        obs_velocity = np.array(getattr(obstacle, 'velocity', np.zeros(2)))
                        print(f"动态障碍物 {i+1}: 位置={obs_pos}, 直接获取速度={obs_velocity}, 速度大小={np.linalg.norm(obs_velocity):.2f}")
                else:
                    # 安全提取各种格式的障碍物数据
                    def safe_extract(val, idx, default):
                        try:
                            if isinstance(val, (list, tuple)) and len(val) > idx:
                                v = val[idx]
                                return float(v[0] if isinstance(v, (list, tuple)) else v)
                            return float(val[idx]) if len(val) > idx else float(default)
                        except (IndexError, TypeError, ValueError):
                            return float(default)
                    
                    obstacle_list = list(obstacle) if hasattr(obstacle, '__iter__') else [obstacle]
                    
                    obs_pos = np.array([safe_extract(obstacle_list, 0, 0), safe_extract(obstacle_list, 1, 0)])
                    obs_radius = safe_extract(obstacle_list, 2, 10.0)
                    obs_velocity = np.array([
                        safe_extract(obstacle_list, 3, 0.0),
                        safe_extract(obstacle_list, 4, 0.0)
                    ])
                    print(f"动态障碍物 {i+1} (非对象格式): 位置={obs_pos}, 速度={obs_velocity}, 速度大小={np.linalg.norm(obs_velocity):.2f}")
            except Exception as e:
                print(f"动态障碍物数据格式错误: {e}, obstacle={obstacle}")
                continue
            
            # 计算距离信息
            distance = np.linalg.norm(current_pos - obs_pos)
            print(f"动态障碍物 {i+1}: 与当前位置距离={distance:.2f}")
            
            repulsive = self._calculate_single_obstacle_repulsion(current_pos, obs_pos, obs_radius, obs_velocity)
            repulsive_magnitude = np.linalg.norm(repulsive)
            total_repulsive += repulsive
            
            # 添加动态障碍物斥力日志（降低阈值）
            print(f"🚨 动态障碍物 {i+1} 斥力: {repulsive_magnitude:.1f}, 速度: {np.linalg.norm(obs_velocity):.1f}, 距离: {distance:.1f}")
        
        print(f"总动态障碍物斥力: {np.linalg.norm(total_repulsive):.1f}")
        return total_repulsive

    def _calculate_single_obstacle_repulsion(self, current_pos: np.ndarray, obs_pos: np.ndarray, 
                                           obs_radius: float, obs_velocity: np.ndarray) -> np.ndarray:
        """计算单个障碍物的斥力（抗震荡优化）"""
        direction = current_pos - obs_pos
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            print("距离过小，返回默认斥力")
            return np.array([1, 0])  # 避免除零
        
        safe_distance = obs_radius + self.config['safety_distance']
        
        # 紧急避障机制：当距离非常接近时，直接施加极大斥力
        emergency_range = obs_radius + 6.0  # 从12减小到6，进一步缩小紧急避障范围
        if distance < emergency_range:  # 非常接近碰撞
            print(f"🚨 紧急避障：距离={distance}, 安全距离={emergency_range}")
            # 施加极大斥力，优先级高于其他计算
            emergency_strength = self.config['repulsive_gain'] * 2.5  # 从3.5降低到2.5，减少紧急避障强度
            return direction / distance * emergency_strength
        
        # 优化斥力影响范围：当距离超出safe_distance+influence_distance时，平滑衰减
        max_influence_distance = safe_distance + self.config['influence_distance']
        if distance > max_influence_distance:
            # 使用平滑衰减而不是立即为0
            decay_factor = max(0, 1 - (distance - max_influence_distance) / 60.0)  # 额外60距离的平滑衰减区，增加过渡长度
            if decay_factor > 0:
                strength = self.config['repulsive_gain'] * decay_factor * 0.2  # 外围衰减区斥力降低到0.2
                print(f"平滑衰减区斥力: 距离={distance}, 衰减因子={decay_factor}, 强度={strength}")
                direction = direction / distance
                return direction * strength
            print(f"超出影响范围: 距离={distance}, 安全距离+影响范围={max_influence_distance}")
            return np.zeros(2)
        
        # 平滑过渡的斥力计算 - 更平缓的斥力曲线
        if distance < safe_distance:
            # 非线性平滑斥力，避免突变
            distance_ratio = max(0, (safe_distance - distance) / safe_distance)
            # 使用更平滑的曲线：先缓慢增长，然后逐渐加快
            strength = self.config['repulsive_gain'] * (distance_ratio ** 2) * 0.9  # 从1.2降低到0.9，减少近距离斥力强度
            print(f"安全距离内斥力: 距离={distance}, 安全距离={safe_distance}, 强度={strength}")
        else:
            influence_ratio = max(0, (self.config['influence_distance'] - (distance - safe_distance)) / self.config['influence_distance'])
            # 平滑过渡的斥力曲线
            strength = self.config['repulsive_gain'] * (influence_ratio ** 1.5) * 0.7  # 从0.8降低到0.7，并使用指数衰减
            print(f"影响范围内斥力: 距离={distance}, 安全距离={safe_distance}, 强度={strength}")
        
        # 归一化方向
        direction = direction / distance
        
        # 平滑的障碍物速度影响 - 减少对速度的过度敏感
        velocity_magnitude = np.linalg.norm(obs_velocity)
        if velocity_magnitude > 0:
            # 降低速度影响因子
            velocity_factor = 1.0 + min(velocity_magnitude / 6.0, 1.5)  # 从4.0增加到6.0，上限从2.5降低到1.5
            strength *= velocity_factor
            print(f"速度影响: 障碍物速度={velocity_magnitude}, 速度因子={velocity_factor}, 增强后强度={strength}")
            
            # 额外添加速度方向相关的斥力分量 - 更平滑的过渡
            relative_velocity = self.current_velocity - obs_velocity
            relative_velocity_magnitude = np.linalg.norm(relative_velocity)
            if relative_velocity_magnitude > 0:
                # 如果障碍物靠近，增加额外斥力 - 更平滑的增强
                dot_product = np.dot(direction, relative_velocity)
                if dot_product < 0:
                    # 根据接近程度平滑增强斥力
                    approach_factor = 1.0 + min(abs(dot_product) / relative_velocity_magnitude, 1.0) * 0.8  # 从2.2降低到最多1.8倍
                    strength *= approach_factor
                    print(f"靠近检测: 点积={dot_product}, 额外增强后强度={strength}")
        
        # 添加切向分量避免局部最小值 - 更平滑的切向力
        tangent = np.array([-direction[1], direction[0]])
        # 根据距离动态调整切向力大小：离障碍物越近，切向力越小，避免过度转向
        tangent_strength = strength * 0.4 * min(1.0, distance / (safe_distance * 0.5))  # 从0.6降低到0.4，并随距离调整
        
        final_force = direction * strength + tangent * tangent_strength
        print(f"最终斥力向量: {final_force}, 大小: {np.linalg.norm(final_force)}")
        
        # 额外的斥力平滑：记忆上一次对该障碍物的斥力，平滑过渡
        obstacle_id = hash(str(obs_pos) + str(obs_radius))
        if not hasattr(self, '_previous_repulsions'):
            self._previous_repulsions = {}
        
        if obstacle_id in self._previous_repulsions:
            # 使用低通滤波器平滑斥力变化
            smoothed_force = self.config['noise_filter'] * self._previous_repulsions[obstacle_id] + \
                           (1 - self.config['noise_filter']) * final_force
            self._previous_repulsions[obstacle_id] = final_force
            print(f"平滑后斥力向量: {smoothed_force}, 大小: {np.linalg.norm(smoothed_force)}")
            return smoothed_force
        else:
            self._previous_repulsions[obstacle_id] = final_force
            return final_force

    def _calculate_emergency_avoidance_force(self, current_pos: np.ndarray, obstacles: List) -> np.ndarray:
        """新增紧急避障机制：当检测到即将发生碰撞时，提供额外的避障力"""
        emergency_force = np.zeros(2)
        
        # 紧急避障的阈值
        collision_threshold = 20.0  # 距离障碍物小于20像素时触发紧急避障
        
        for obstacle in obstacles:
            try:
                # 提取障碍物信息
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    obs_pos = np.array(obstacle.position)
                    obs_radius = float(obstacle.size) if hasattr(obstacle.size, '__float__') else 10.0
                    # 获取障碍物速度
                    if hasattr(obstacle, 'direction') and hasattr(obstacle, 'speed'):
                        direction = np.array(obstacle.direction)
                        speed = float(obstacle.speed)
                        obs_velocity = direction * speed
                    else:
                        obs_velocity = np.array(getattr(obstacle, 'velocity', np.zeros(2)))
                else:
                    # 简化处理，跳过非标准格式的障碍物
                    continue
                
                # 计算距离
                distance = np.linalg.norm(current_pos - obs_pos)
                
                # 检测是否需要紧急避障
                if distance < obs_radius + collision_threshold:
                    # 计算到障碍物的方向（远离障碍物）
                    avoidance_direction = current_pos - obs_pos
                    if np.linalg.norm(avoidance_direction) > 0:
                        avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
                    
                    # 计算紧急避障力大小（距离越近，力越大）
                    emergency_strength = self.config['repulsive_gain'] * 2.0 * ((obs_radius + collision_threshold - distance) / collision_threshold)
                    
                    # 如果障碍物是动态的，根据其速度方向调整避障方向
                    if hasattr(obstacle, 'velocity') or hasattr(obstacle, 'speed'):
                        # 如果障碍物速度不为零，稍微调整避障方向以应对障碍物的移动
                        obs_vel_mag = np.linalg.norm(obs_velocity)
                        if obs_vel_mag > 0:
                            # 预测障碍物未来位置（简单预测）
                            pred_obs_pos = obs_pos + obs_velocity * 0.5  # 预测0.5秒后的位置
                            # 计算到预测位置的方向
                            pred_direction = current_pos - pred_obs_pos
                            if np.linalg.norm(pred_direction) > 0:
                                pred_direction = pred_direction / np.linalg.norm(pred_direction)
                                # 混合当前方向和预测方向
                                avoidance_direction = 0.7 * avoidance_direction + 0.3 * pred_direction
                    
                    # 添加到总紧急避障力
                    emergency_force += avoidance_direction * emergency_strength
                    print(f"🚨 紧急避障力: 大小={emergency_strength:.1f}, 方向={avoidance_direction}, 距离={distance:.1f}")
            except Exception as e:
                print(f"紧急避障计算错误: {e}")
                continue
        
        return emergency_force

    def _calculate_boundary_repulsion(self, current_pos: np.ndarray) -> np.ndarray:
        """计算边界斥力"""
        x, y = current_pos[0], current_pos[1]
        boundary_thresh = 80.0  # 从arithmetic/APF/apf.py中参考
        boundary_gain = 600.0   # 从arithmetic/APF/apf.py中参考
        
        # 假设边界尺寸（这里使用默认值，实际应用中可能需要从配置获取）
        width, height = 1000, 600  # 假设的地图尺寸
        
        # 到各墙的距离
        d_left = max(1e-6, x)
        d_right = max(1e-6, width - x)
        d_bottom = max(1e-6, y)
        d_top = max(1e-6, height - y)
        
        repulsion_force = np.zeros(2)
        
        # 计算左边墙的斥力
        if d_left < boundary_thresh:
            force_mag = boundary_gain * (1.0 / d_left - 1.0 / boundary_thresh) / (d_left ** 2)
            repulsion_force[0] += force_mag
        
        # 计算右边墙的斥力
        if d_right < boundary_thresh:
            force_mag = boundary_gain * (1.0 / d_right - 1.0 / boundary_thresh) / (d_right ** 2)
            repulsion_force[0] -= force_mag
        
        # 计算下边墙的斥力
        if d_bottom < boundary_thresh:
            force_mag = boundary_gain * (1.0 / d_bottom - 1.0 / boundary_thresh) / (d_bottom ** 2)
            repulsion_force[1] += force_mag
        
        # 计算上边墙的斥力
        if d_top < boundary_thresh:
            force_mag = boundary_gain * (1.0 / d_top - 1.0 / boundary_thresh) / (d_top ** 2)
            repulsion_force[1] -= force_mag
        
        # 限制斥力大小
        max_repulsion = 500.0
        if np.linalg.norm(repulsion_force) > max_repulsion:
            repulsion_force = repulsion_force / np.linalg.norm(repulsion_force) * max_repulsion
        
        return repulsion_force
    
    def _check_collision(self, current_pos: np.ndarray, obstacles: List) -> bool:
        """检查是否与任何障碍物发生碰撞"""
        for obstacle in obstacles:
            try:
                # 处理不同类型的障碍物数据结构
                if isinstance(obstacle, dict):
                    # 字典类型障碍物
                    obs_pos = np.array([obstacle.get('x', 0), obstacle.get('y', 0)])
                    obs_radius = obstacle.get('radius', 20)
                elif isinstance(obstacle, (list, tuple)) and len(obstacle) >= 2:
                    # 列表或元组类型障碍物
                    obs_pos = np.array([obstacle[0], obstacle[1]])
                    obs_radius = obstacle[2] if len(obstacle) > 2 else 20
                else:
                    continue
                
                # 确保半径是数值类型
                if not isinstance(obs_radius, (int, float)):
                    continue
                
                # 计算距离并检查碰撞（考虑机器人自身半径约10）
                distance = np.linalg.norm(current_pos - obs_pos)
                if distance < (obs_radius + 10):
                    return True
            except Exception as e:
                # 忽略处理单个障碍物时的错误
                print(f"碰撞检测错误: {e}")
                continue
        return False
    
    def _check_path_segment(self, start_pos: np.ndarray, end_pos: np.ndarray, obstacles: List) -> bool:
        """检查两点之间的路径段是否可行（无碰撞）"""
        # 采样检查路径段
        num_samples = 10  # 采样点数
        for i in range(num_samples + 1):
            t = i / num_samples
            check_pos = start_pos + t * (end_pos - start_pos)
            if self._check_collision(check_pos, obstacles):
                return False  # 路径段不可行
        return True  # 路径段可行
    
    def find_lookahead_waypoint(self, current_pos: np.ndarray, global_path: List[np.ndarray], 
                               static_obstacles: List, dynamic_obstacles: List, 
                               velocity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        严格按照前视距离定义查找局部目标点：
        从当前位置出发，在全局路径上寻找距离当前点lookahead_distance以内，
        且无障碍遮挡（可见性良好）的最远路径点。
        
        Args:
            current_pos: 当前位置
            global_path: 全局路径点列表
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            velocity: 当前速度向量
            
        Returns:
            前视距离内最远的可行局部目标点
        """
        # 更新全局路径缓存
        if global_path is not None and len(global_path) > 0:
            self.global_path = [np.array(point) for point in global_path]
        
        # 如果全局路径无效，返回当前位置或终点
        if self.global_path is None or len(self.global_path) == 0:
            return current_pos
        
        if len(self.global_path) == 1:
            return self.global_path[0]
        
        # 获取前视距离参数
        lookahead_distance = self.config.get('lookahead_distance', 60.0)
        
        # 找到距离当前位置最近的路径点作为起始点
        distances = [np.linalg.norm(current_pos - point) for point in self.global_path]
        closest_index = np.argmin(distances)
        
        # 合并所有障碍物用于可见性检查
        all_obstacles = static_obstacles + dynamic_obstacles
        
        # 检查循环状态并调整目标点选择策略
        is_looping = getattr(self, 'is_looping', False)
        
        # 循环状态下的特殊处理
        if is_looping:
            # 循环状态下临时增加前视距离
            adjusted_lookahead = lookahead_distance * 1.5  # 增加50%
            # 循环状态下跳过一些路径点，选择更远的起始点
            skip_points = min(3, len(self.global_path) - closest_index - 1)
            search_start_index = min(closest_index + skip_points, len(self.global_path) - 1)
            print(f"🔄 循环状态：前视距离增加至{adjusted_lookahead}，跳过{skip_points}个点，从索引{search_start_index}开始搜索")
        else:
            adjusted_lookahead = lookahead_distance
            search_start_index = closest_index
        
        # 从调整后的起始点开始，向前查找最远的可行点
        best_index = closest_index  # 默认为最近点，确保至少有一个点
        best_point = self.global_path[closest_index]
        max_distance = 0.0
        best_direction_score = -float('inf')
        
        # 计算当前运动方向
        current_direction = None
        if velocity is not None and np.linalg.norm(velocity) > 0.1:
            current_direction = velocity / np.linalg.norm(velocity)
        
        # 遍历从起始点到路径末尾的所有点
        for i in range(search_start_index, len(self.global_path)):
            path_point = self.global_path[i]
            distance = np.linalg.norm(current_pos - path_point)
            
            # 如果超出调整后的前视距离，停止搜索
            if distance > adjusted_lookahead:
                break
            
            # 检查从当前位置到该路径点是否有障碍物遮挡（可见性检查）
            if self._check_path_segment(current_pos, path_point, all_obstacles):
                # 计算方向分数（考虑运动方向的一致性）
                direction_score = distance  # 基础分数为距离
                if current_direction is not None:
                    to_point_dir = (path_point - current_pos) / distance
                    # 方向一致性加分（与当前运动方向夹角越小，分数越高）
                    direction_bonus = np.dot(current_direction, to_point_dir) * distance * 0.5
                    direction_score += direction_bonus
                
                # 如果路径段可行且综合分数更高，则更新最佳点
                if direction_score > best_direction_score:
                    max_distance = distance
                    best_index = i
                    best_point = path_point
                    best_direction_score = direction_score
        
        # 如果在循环状态下但没有找到更远的点，强制选择路径上更远的点
        if is_looping and best_index <= closest_index + 1 and closest_index < len(self.global_path) - 3:
            force_index = min(closest_index + 3, len(self.global_path) - 1)
            force_point = self.global_path[force_index]
            # 只在该点可见或路径上到该点的点都可见时才强制选择
            path_clear = True
            for j in range(closest_index, force_index + 1):
                if not self._check_path_segment(current_pos, self.global_path[j], all_obstacles):
                    path_clear = False
                    break
            
            if path_clear:
                best_index = force_index
                best_point = force_point
                print(f"🚨 循环强制策略：选择更远的点[索引{best_index}]")
        
        # 记录并输出选择的目标点信息
        distance_to_target = np.linalg.norm(current_pos - best_point)
        print(f"🎯 前视距离选择: 目标点[索引{best_index}]，距离{distance_to_target:.2f}，前视距离{adjusted_lookahead}，循环状态={is_looping}")
        
        # 保存最后选择的目标点（用于调试）
        self.last_target_point = best_point.copy()
        
        return best_point
        
    def _apply_boundary_constraints(self, position: np.ndarray) -> np.ndarray:
        """应用边界约束，确保位置不会超出地图边界"""
        # 假设边界尺寸（实际应用中可能需要从配置获取）
        width, height = 1000, 600  # 假设的地图尺寸
        
        # 确保位置在边界内
        constrained_pos = np.copy(position)
        constrained_pos[0] = max(0, min(constrained_pos[0], width))
        constrained_pos[1] = max(0, min(constrained_pos[1], height))
        
        return constrained_pos
    
    def _check_segment_path(self, start_pos: np.ndarray, end_pos: np.ndarray, obstacles: List) -> bool:
        """检查路径段是否无碰撞（_check_path_segment的别名方法）"""
        return self._check_path_segment(start_pos, end_pos, obstacles)
    
    def discretize_path(self, full_path: List[np.ndarray], distance_threshold: float = 50.0) -> List[np.ndarray]:
        """
        将完整路径稀疏化，每隔一段距离保留一个点作为局部子目标点
        
        Args:
            full_path: 完整路径点列表
            distance_threshold: 距离阈值，超过这个距离才会保留新的子目标点
            
        Returns:
            离散化后的局部子目标点列表
        """
        if not full_path or len(full_path) < 2:
            return full_path
        
        # 确保路径点是numpy数组
        path_np = [np.array(point) for point in full_path]
        
        # 初始化离散化路径，包含起点
        discrete_path = [path_np[0]]
        
        # 上一个保留点
        last_point = path_np[0]
        
        # 遍历路径点，根据距离阈值保留点
        for point in path_np[1:]:
            distance = np.linalg.norm(point - last_point)
            # 如果距离超过阈值，保留该点
            if distance >= distance_threshold:
                discrete_path.append(point)
                last_point = point
        
        # 确保终点总是被包含
        if len(discrete_path) > 0 and not np.array_equal(discrete_path[-1], path_np[-1]):
            discrete_path.append(path_np[-1])
        
        # 特殊情况处理：如果离散化后只有起点和终点且距离很远，中间补一个点
        if len(discrete_path) == 2:
            start, end = discrete_path
            mid_point = (start + end) / 2
            discrete_path.insert(1, mid_point)
        
        print(f"📏 路径稀疏化完成：{len(full_path)} 个点 → {len(discrete_path)} 个局部子目标点")
        
        return discrete_path