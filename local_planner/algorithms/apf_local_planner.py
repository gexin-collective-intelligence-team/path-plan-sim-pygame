import numpy as np
from typing import List, Optional
from ..base_local_planner import BaseLocalPlanner
from shapely.geometry import Point, Polygon, LineString
from shapely import affinity
import math


class APFLocalPlanner(BaseLocalPlanner):
    """
    简化版APF局部路径规划器
    仅保留基本的引力和斥力计算
    """
    
    def __init__(self, name="APF算法"):
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
            'attractive_gain': 3.0,        # 引力增益系数
            'min_distance': 5.0,           # 到达目标的最小距离阈值
            
            # 斥力参数
            'repulsive_gain': 3000.0,      # 斥力增益系数（进一步增强）
            'static_safety_distance': 30.0, # 静态障碍物安全距离
            'dynamic_safety_distance': 25.0, # 动态障碍物安全距离（进一步缩短）
            'static_influence_distance': 100.0, # 静态障碍物影响距离
            'dynamic_influence_distance': 200.0, # 动态障碍物影响距离（大幅扩大）
            'prediction_time': 1.0,        # 动态障碍物预测时间（缩短为1秒）
            
            # 威胁权重 - 调整为更合理的分配
            'threat_distance_weight': 5.0,  # 距离权重（增强）
            'threat_speed_weight': 3.0,     # 速度权重（增强）
            'threat_direction_weight': 4.0, # 方向权重（增强）
            
            # 移动参数
            'max_step': 30.0,              # 最大步长（增加以提高USV速度）
            
            # 振荡检测参数
            'oscillation_threshold': 3,    # 检测振荡的历史步数
            'oscillation_distance': 10.0,  # 判断振荡的距离阈值
            'oscillation_angle': 120.0,    # 判断振荡的角度阈值（度）
            
            # 临时目标点参数
            'temp_goal_distance': 50.0,    # 临时目标点距离
            'temp_goal_angles': [45, 90, 135, -45, -90, -135],  # 临时目标点候选角度
        }
        
        # 历史路径点，用于检测振荡
        self.history_positions = []
        
        # 临时目标点标志
        self.use_temp_goal = False
        self.temp_goal = None
        self.temp_goal_count = 0
        self.max_temp_goal_steps = 10  # 临时目标点最大使用步数
    def reset_planner(self):
        """手动重置规划器状态，确保到达目标点后立即重置所有相关参数"""
        # 直接设置属性而非删除，确保一致的初始化状态
        self.current_target_idx = 0
        self.is_initialized = False
        self.last_target_pos = None
        self.stuck_counter = 0
        self.last_distance_to_target = float('inf')
        self.goal_reached = False  # 明确设置为False，确保重新开始规划
        self.history_positions = []  # 清空历史位置
        self.use_temp_goal = False
        self.temp_goal = None
        self.temp_goal_count = 0
        self._goal_reached_logged = False  # 重置日志标志，允许下次到达时再次打印
        print("APF 规划器状态已完全重置，可重新开始路径规划。")
    def plan(self, current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity=None, global_path=None):
        """
        改进版APF路径规划（增强版）
        保留振荡检测与临时目标点机制
        新增目标点可达性检测与自动跳过机制
        """

        # ---------- 基本初始化 ----------
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 检查是否已到达目标点，如果是则直接返回当前位置，不再进行任何计算
        if hasattr(self, "goal_reached") and self.goal_reached:
            # 只在第一次到达目标点时打印日志，避免重复打印
            if not getattr(self, "_goal_reached_logged", False):
                print("已到达目标点，保持在目标位置，不进行任何计算。")
                self._goal_reached_logged = True
            return current_pos.copy()  # 返回当前位置的副本，确保保持在原地
        
        # 初始化或在 reset 后重新初始化
        if not getattr(self, "is_initialized", False):
            self.current_target_idx = 0
            self.last_distance_to_target = float('inf')
            self.stuck_counter = 0
            self.max_stuck_steps = 10       # 判定阻挡的连续步数
            self.goal_threshold = self.config.get('min_distance', 1.0)
            self.goal_reached = False  # 初始化目标到达标志
            self.is_initialized = True
            print("规划器状态已初始化。")
            print("规划器状态已初始化。")

        # 若传入全局路径，则根据索引更新目标点（至少需要 2 个点才视为全局路径）
        multi_point_path = global_path is not None and len(global_path) > 1
        if multi_point_path:
            if self.current_target_idx >= len(global_path):
                print("所有目标点已到达或被跳过，规划结束。")
                return current_pos
            target_pos = np.array(global_path[self.current_target_idx])

            # 仅在目标点被障碍物覆盖或穿透时才跳过，避免误报
            if self._is_target_blocked(target_pos, static_obstacles, dynamic_obstacles):
                print(f"当前目标点 {self.current_target_idx} 被障碍物覆盖，立即跳过。")
                self.current_target_idx += 1
                if self.current_target_idx < len(global_path):
                    return self.plan(
                        current_pos,
                        np.array(global_path[self.current_target_idx]),
                        static_obstacles,
                        dynamic_obstacles,
                        velocity,
                        global_path
                    )
                print("所有目标点均无法通过直线路径到达，规划终止。")
                self.goal_reached = True
                return target_pos.copy()

        # ---------- 到目标的距离 ----------
        distance_to_target = np.linalg.norm(target_pos - current_pos)

        # ---------- 到达检测 ----------
        if distance_to_target < self.goal_threshold:
            print(f"到达目标点 {self.current_target_idx}，切换下一个。")
            if multi_point_path:
                self.current_target_idx += 1
                if self.current_target_idx < len(global_path):
                    # 切换下一个目标点继续规划
                    return self.plan(current_pos, global_path[self.current_target_idx],
                                    static_obstacles, dynamic_obstacles, velocity, global_path)
                else:
                    print("最终目标点已到达，路径规划完成，立即设置到达标志。")
                    self.goal_reached = True  # 设置目标到达标志
                    return target_pos.copy()  # 返回目标点的副本，确保精确到达
            else:
                # 无全局路径输入时直接停止
                print("目标点已到达，路径规划完成，立即设置到达标志。")
                self.goal_reached = True  # 设置目标到达标志
                return target_pos.copy()  # 返回目标点的副本，确保精确到达

        # ---------- 更新历史状态 ----------
        self._update_history(current_pos)

        # ---------- 振荡检测与临时目标点 ----------
        if not self.use_temp_goal and self._detect_oscillation():
            self.temp_goal = self._select_temp_goal(current_pos, target_pos, static_obstacles, dynamic_obstacles)
            if self.temp_goal is not None:
                self.use_temp_goal = True
                self.temp_goal_count = 0

        if self.use_temp_goal:
            self.temp_goal_count += 1
            if np.linalg.norm(current_pos - self.temp_goal) < self.config['min_distance'] or \
            self.temp_goal_count > self.max_temp_goal_steps:
                self.use_temp_goal = False
                self.temp_goal = None
                self.temp_goal_count = 0

        current_goal = self.temp_goal if self.use_temp_goal else target_pos

        # ---------- 计算合力 ----------
        attractive_force = self._calculate_attractive_force(current_pos, current_goal)
        
        # 使用新的分离式斥力计算
        static_repulsive = self._calculate_static_repulsion(current_pos, static_obstacles)
        dynamic_repulsive = self._calculate_dynamic_repulsion(current_pos, dynamic_obstacles)
        repulsive_force = static_repulsive + dynamic_repulsive
        
        total_force = attractive_force + repulsive_force
        
        # 调试输出 - 显示力的信息
        if dynamic_obstacles:
            static_magnitude = np.linalg.norm(static_repulsive)
            dynamic_magnitude = np.linalg.norm(dynamic_repulsive)
            attractive_magnitude = np.linalg.norm(attractive_force)
            total_magnitude = np.linalg.norm(total_force)
            
            # 计算力的方向
            if dynamic_magnitude > 0:
                dynamic_dir = dynamic_repulsive / dynamic_magnitude
            else:
                dynamic_dir = np.array([0.0, 0.0])
                
            if attractive_magnitude > 0:
                attractive_dir = attractive_force / attractive_magnitude
            else:
                attractive_dir = np.array([0.0, 0.0])
                
            if total_magnitude > 0:
                total_dir = total_force / total_magnitude
            else:
                total_dir = np.array([0.0, 0.0])
            
            print(f"调试 - 引力: {attractive_magnitude:.2f}({attractive_dir[0]:.2f},{attractive_dir[1]:.2f}), "
                  f"静态斥力: {static_magnitude:.2f}, 动态斥力: {dynamic_magnitude:.2f}({dynamic_dir[0]:.2f},{dynamic_dir[1]:.2f}), "
                  f"合力: {total_magnitude:.2f}({total_dir[0]:.2f},{total_dir[1]:.2f})")

        # ---------- 计算下一步位置 ----------
        if np.linalg.norm(total_force) > 0:
            direction = total_force / np.linalg.norm(total_force)
            next_pos = current_pos + direction * min(self.config['max_step'], distance_to_target)
            # 检测是否会碰撞
            if self._check_collision(current_pos, next_pos, static_obstacles, dynamic_obstacles):
                next_pos = self._find_collision_free_direction(current_pos, direction, static_obstacles, dynamic_obstacles)
        else:
            direction = (current_goal - current_pos) / np.linalg.norm(current_goal - current_pos)
            next_pos = current_pos + direction * min(self.config['max_step'], distance_to_target)

        # ---------- 可达性检测与跳过机制 ----------
        # 若连续多次未靠近目标，则认为目标点被障碍物阻挡
        if distance_to_target >= self.last_distance_to_target - 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_distance_to_target = distance_to_target

        # 根据与目标的距离自适应调整阻塞检测阈值，让“绕行”判断更敏感
        adaptive_limit = self.max_stuck_steps
        if distance_to_target < 80:
            adaptive_limit = min(adaptive_limit, 4)
        elif distance_to_target < 150:
            adaptive_limit = min(adaptive_limit, 6)
        else:
            adaptive_limit = min(adaptive_limit, 8)

        if self.stuck_counter > adaptive_limit:
            print(f"当前目标点 {self.current_target_idx} 被阻挡，自动跳过。")
            self.stuck_counter = 0
            self.last_distance_to_target = float('inf')

            if global_path is not None:
                self.current_target_idx += 1
                if self.current_target_idx < len(global_path):
                    # 跳过当前目标点，继续规划下一个
                    return self.plan(current_pos, global_path[self.current_target_idx],
                                    static_obstacles, dynamic_obstacles, velocity, global_path)
                else:
                    print("所有目标点均无法到达，规划终止。")
                    return current_pos

        return next_pos

    def _is_target_blocked(self, target_pos, static_obstacles, dynamic_obstacles):
        """
        判断目标点是否被障碍物覆盖/穿透
        """
        target_point = Point(target_pos[0], target_pos[1])

        # 静态障碍物：检查目标点是否被多边形包含
        for obstacle in static_obstacles:
            try:
                if len(obstacle) >= 3:
                    if obstacle[0] != obstacle[-1]:
                        poly = Polygon(obstacle + [obstacle[0]])
                    else:
                        poly = Polygon(obstacle)
                    if poly.contains(target_point):
                        return True
            except Exception:
                continue

        # 动态障碍物：判断目标点是否落入动态障碍物的覆盖半径
        for obstacle in dynamic_obstacles:
            if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                obs_pos = obstacle.position
                distance = np.linalg.norm(np.array(obs_pos) - np.array(target_pos))
                if distance <= float(obstacle.size):
                    return True

        return False

    
    def _update_history(self, position):
        """
        更新历史位置记录
        
        Args:
            position: 当前位置
        """
        self.history_positions.append(np.array(position))
        # 只保留最近的位置记录
        if len(self.history_positions) > self.config['oscillation_threshold'] + 1:
            self.history_positions.pop(0)
    
    def _detect_oscillation(self):
        """
        检测是否发生振荡
        
        Returns:
            bool: 是否检测到振荡
        """
        # 如果历史位置不足，无法检测
        if len(self.history_positions) < self.config['oscillation_threshold'] + 1:
            return False
        
        # 检查是否在小范围内反复移动
        recent_positions = self.history_positions[-self.config['oscillation_threshold']:]
        first_pos = recent_positions[0]
        
        # 检查所有最近的位置是否都在一个小范围内
        all_close = all(np.linalg.norm(pos - first_pos) < self.config['oscillation_distance'] 
                       for pos in recent_positions)
        
        if not all_close:
            return False
        
        # 检查移动方向是否反复变化
        directions = []
        for i in range(1, len(recent_positions)):
            dir_vec = recent_positions[i] - recent_positions[i-1]
            if np.linalg.norm(dir_vec) > 0:
                directions.append(dir_vec / np.linalg.norm(dir_vec))
        
        if len(directions) < 2:
            return False
        
        # 检查相邻方向之间的夹角是否较大（振荡特征）
        oscillation_detected = False
        for i in range(1, len(directions)):
            dot_product = np.dot(directions[i], directions[i-1])
            angle = np.degrees(np.arccos(max(-1.0, min(1.0, dot_product))))
            if angle > self.config['oscillation_angle']:
                oscillation_detected = True
                break
        
        return oscillation_detected
    
    def _select_temp_goal(self, current_pos, target_pos, static_obstacles, dynamic_obstacles):
        """
        选择无碰撞的临时目标点
        
        Args:
            current_pos: 当前位置
            target_pos: 全局目标位置
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            
        Returns:
            numpy.ndarray: 临时目标点坐标，如果没有找到则返回None
        """
        # 计算朝向目标的主方向
        main_direction = target_pos - current_pos
        if np.linalg.norm(main_direction) > 0:
            main_direction = main_direction / np.linalg.norm(main_direction)
        else:
            main_direction = np.array([1.0, 0.0])
        
        # 生成临时目标点候选
        temp_goals = []
        for angle_deg in self.config['temp_goal_angles']:
            # 转换角度为弧度
            angle_rad = np.radians(angle_deg)
            
            # 旋转主方向向量
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            temp_direction = np.dot(rotation_matrix, main_direction)
            
            # 计算临时目标点
            temp_goal = current_pos + temp_direction * self.config['temp_goal_distance']
            
            # 检查从当前位置到临时目标点的路径是否无碰撞
            if not self._check_collision(current_pos, temp_goal, static_obstacles, dynamic_obstacles):
                # 计算临时目标点到全局目标的距离（优先选择更接近全局目标的方向）
                distance_to_global = np.linalg.norm(temp_goal - target_pos)
                temp_goals.append((temp_goal, distance_to_global))
        
        # 如果找到无碰撞的临时目标点，选择最接近全局目标的
        if temp_goals:
            temp_goals.sort(key=lambda x: x[1])  # 按到全局目标的距离排序
            return temp_goals[0][0]
        
        # 如果没有找到无碰撞的临时目标点，尝试使用较小的距离
        reduced_distance = self.config['temp_goal_distance'] * 0.5
        for angle_deg in self.config['temp_goal_angles']:
            angle_rad = np.radians(angle_deg)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            temp_direction = np.dot(rotation_matrix, main_direction)
            temp_goal = current_pos + temp_direction * reduced_distance
            
            if not self._check_collision(current_pos, temp_goal, static_obstacles, dynamic_obstacles):
                return temp_goal
        
        return None
    
    def _check_collision(self, start_pos, end_pos, static_obstacles, dynamic_obstacles):
        """
        检查从起点到终点的线段是否与障碍物碰撞
        
        Args:
            start_pos: 起点位置
            end_pos: 终点位置
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            
        Returns:
            bool: 是否发生碰撞
        """
        # 创建线段
        line = LineString([tuple(start_pos), tuple(end_pos)])
        
        # 检查静态障碍物碰撞
        for obstacle in static_obstacles:
            try:
                if len(obstacle) >= 3:
                    if obstacle[0] != obstacle[-1]:
                        obstacle_poly = Polygon(obstacle + [obstacle[0]])
                    else:
                        obstacle_poly = Polygon(obstacle)
                    
                    # 检查线段是否与多边形相交或距离太近
                    if line.distance(obstacle_poly) < 1e-6 or line.intersects(obstacle_poly):
                        return True
            except Exception:
                continue
        
        # 检查动态障碍物碰撞
        for obstacle in dynamic_obstacles:
            try:
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    # 计算动态障碍物的预测位置
                    if hasattr(obstacle, 'direction') and hasattr(obstacle, 'speed'):
                        # direction 需单位化，否则预测会炸裂
                        direction = np.array(obstacle.direction)
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        
                        predicted_position = np.array(obstacle.position) + direction * float(obstacle.speed) * self.config['prediction_time']
                    else:
                        predicted_position = np.array(obstacle.position)
                    
                    # 使用半径进行碰撞检测（修复：使用 radius + safety_distance）
                    radius = float(obstacle.size)
                    circle = Point(predicted_position[0], predicted_position[1]).buffer(radius + self.config['dynamic_safety_distance'])
                    
                    # 检查线段是否与障碍物圆圈相交
                    if line.intersects(circle):
                        return True
            except Exception:
                continue
        
        return False
    
    def _find_collision_free_direction(self, current_pos, original_direction, static_obstacles, dynamic_obstacles):
        """
        寻找无碰撞的方向
        
        Args:
            current_pos: 当前位置
            original_direction: 原始方向
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            
        Returns:
            numpy.ndarray: 无碰撞的下一个位置
        """
        # 尝试不同角度的方向
        angles = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90]
        
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            
            # 旋转原始方向向量
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])
            new_direction = np.dot(rotation_matrix, original_direction)
            new_direction = new_direction / np.linalg.norm(new_direction)
            
            # 计算新的目标位置
            new_pos = current_pos + new_direction * self.config['max_step']
            
            # 检查是否无碰撞
            if not self._check_collision(current_pos, new_pos, static_obstacles, dynamic_obstacles):
                return new_pos
        
        # 如果所有方向都有碰撞，尝试减小步长
        reduced_step = self.config['max_step'] * 0.5
        return current_pos + original_direction * reduced_step
    
    def _calculate_dynamic_repulsion(self, current_pos, dynamic_obstacles):
        """
        计算动态障碍物的斥力（修复版本）
        
        Args:
            current_pos: 当前位置
            dynamic_obstacles: 动态障碍物列表
            
        Returns:
            动态障碍物斥力向量
        """
        repulsive_force = np.zeros(2)

        for obs in dynamic_obstacles:
            try:
                # 1. 障碍物当前和预测位置
                pos = np.array(obs.position)
                radius = float(obs.size)

                # direction 需单位化，否则预测会炸裂
                direction = np.array(obs.direction)
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)

                pred_pos = pos + direction * obs.speed * self.config['prediction_time']

                # 2. 计算机器人与预测位置的距离
                diff = current_pos - pred_pos
                dist = np.linalg.norm(diff)

                influence = self.config['dynamic_influence_distance']
                safety = self.config['dynamic_safety_distance'] + radius

                # 🎯 修复远距离斥力为0的问题
                # 不再跳过远距离障碍物，而是限制最大距离
                if dist > influence:
                    dist = influence  # 限制最大距离，确保斥力不为0

                # 调试障碍物尺寸
                if dist < influence * 0.8:  # 只在接近时输出调试信息
                    print(f"障碍物调试 - 位置: ({pos[0]:.1f},{pos[1]:.1f}), 半径: {radius:.1f}, 速度: {obs.speed:.2f}")

                if dist < 1e-5:
                    continue

                # 3. 威胁指数（距离、速度、方向）
                # 障碍物是否正朝机器人移动
                relative_dir = direction  # 障碍物前进方向
                to_robot = (current_pos - pos)
                if np.linalg.norm(to_robot) > 0:
                    to_robot = to_robot / np.linalg.norm(to_robot)

                direction_factor = np.dot(relative_dir, to_robot)  # 朝着你冲过来 → 正数

                # 动态障碍物危险度模型（速度 + 接近方向 + 距离）
                # 修复接近速度计算 - 使用障碍物速度向量
                obstacle_velocity = direction * obs.speed
                relative_velocity = obstacle_velocity  # 障碍物相对于USV的速度
                
                # 计算障碍物朝向USV的速度分量
                to_obstacle = pred_pos - current_pos
                if np.linalg.norm(to_obstacle) > 0:
                    to_obstacle_normalized = to_obstacle / np.linalg.norm(to_obstacle)
                    closing_speed = np.dot(relative_velocity, to_obstacle_normalized)
                else:
                    closing_speed = 0.0
                
                closing_speed = max(0, closing_speed)  # 只有接近才算威胁，远离不算

                speed_factor = 1 + closing_speed * 2   # 降低速度增强倍数，避免过度反应

                # 调试威胁计算
                if dist < influence * 0.8:  # 只在接近时输出调试信息
                    print(f"威胁调试 - 距离: {dist:.1f}, 接近速度: {closing_speed:.2f}, "
                          f"速度增强: {speed_factor:.2f}, 方向因子: {direction_factor:.2f}")

                # 改进斥力：使用更直接的距离衰减公式
                if dist < safety:  # 在安全距离内，强力避障
                    rep_magnitude = (
                        self.config['repulsive_gain'] * speed_factor * (1.0 - dist / safety)
                    )
                else:  # 在影响距离内，渐进避障
                    rep_magnitude = (
                        self.config['repulsive_gain'] * speed_factor * 
                        (1.0 / dist - 1.0 / influence)
                    )

                rep = rep_magnitude * (diff / dist)

                # 🎯 切向力实现自然绕行（用户方案）
                if np.linalg.norm(rep) > 0:
                    rep_dir = rep / np.linalg.norm(rep)
                    
                    # 计算法向（垂直）方向：用于切向避障
                    tangent_dir = np.array([-rep_dir[1], rep_dir[0]])  # 逆时针旋转90°
                    
                    # 判断应该往左绕还是往右绕：根据相对运动方向
                    obs_vel = direction * obs.speed
                    cross = obs_vel[0] * rep_dir[1] - obs_vel[1] * rep_dir[0]  # 2D cross product
                    
                    if cross < 0:  
                        tangent_dir = -tangent_dir  # 改成顺时针方向
                    
                    # 组合力（非常关键）
                    alpha = 0.9    # 切向力权重，0~1，越大越偏向绕行（提高到0.9）
                    rep = rep_magnitude * ((1-alpha) * rep_dir + alpha * tangent_dir)
                    
                    # 调试切向力
                    if dist < influence * 0.8:
                        print(f"切向绕行 - 法向: ({rep_dir[0]:.2f},{rep_dir[1]:.2f}), "
                              f"切向: ({tangent_dir[0]:.2f},{tangent_dir[1]:.2f}), "
                              f"叉积: {cross:.2f}, alpha: {alpha}")

                # 调试斥力输出
                if dist < influence * 0.8:
                    rep_magnitude_final = np.linalg.norm(rep)
                    print(f"改进斥力 - 距离: {dist:.1f}, 接近速度: {closing_speed:.2f}, "
                          f"速度增强: {speed_factor:.2f}, 斥力大小: {rep_magnitude_final:.2f}")

                # 限制斥力大小，防止爆炸
                repulsive_force += np.clip(rep, -10000, 10000)  # 进一步增大力上限
                
            except Exception as e:
                continue

        return repulsive_force

    def _calculate_static_repulsion(self, current_pos, static_obstacles):
        """
        计算静态障碍物的斥力
        
        Args:
            current_pos: 当前位置
            static_obstacles: 静态障碍物列表
            
        Returns:
            静态障碍物斥力向量
        """
        total_repulsive = np.zeros(2)
        current_point = Point(current_pos[0], current_pos[1])
        
        # 1. 处理静态障碍物（多边形）
        for obstacle in static_obstacles:
            try:
                # 假设静态障碍物是多边形坐标点列表
                if len(obstacle) >= 3:
                    # 确保多边形闭合
                    if obstacle[0] != obstacle[-1]:
                        obstacle_poly = Polygon(obstacle + [obstacle[0]])
                    else:
                        obstacle_poly = Polygon(obstacle)
                    
                    # 计算到多边形的最短距离
                    if obstacle_poly.is_valid:
                        distance = current_point.distance(obstacle_poly)
                        
                        if distance < self.config['static_influence_distance']:
                            # 找到最近点
                            closest_point = obstacle_poly.exterior.interpolate(
                                obstacle_poly.exterior.project(current_point))
                            closest_coords = np.array([closest_point.x, closest_point.y])
                            
                            # 计算斥力方向和大小
                            direction = current_pos - closest_coords
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                
                                # 使用改进的斥力公式
                                distance = max(distance, 0.1)  # 避免除零
                                force_magnitude = self.config['repulsive_gain'] * (
                                    1.0 / distance - 1.0 / self.config['static_safety_distance']
                                ) * (1.0 / (distance ** 2))
                                
                                total_repulsive += direction * force_magnitude
            except Exception as e:
                continue
        
        return total_repulsive
    
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
    
    def _calculate_repulsive_force(self, current_pos: np.ndarray, static_obstacles: List, dynamic_obstacles: List, target_pos: np.ndarray = None) -> np.ndarray:
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
        current_point = Point(current_pos[0], current_pos[1])
        
        # 1. 处理静态障碍物（多边形）
        for obstacle in static_obstacles:
            try:
                # 假设静态障碍物是多边形坐标点列表
                if len(obstacle) >= 3:
                    # 确保多边形闭合
                    if obstacle[0] != obstacle[-1]:
                        obstacle_poly = Polygon(obstacle + [obstacle[0]])
                    else:
                        obstacle_poly = Polygon(obstacle)
                    
                    # 计算点到多边形的距离和最近点
                    distance = current_point.distance(obstacle_poly)
                    
                    # 获取最近点（用于确定斥力方向）
                    if current_point.distance(obstacle_poly) < 1e-6:
                        # 如果点在多边形内部，使用多边形边界的法线方向
                        closest_point = np.array([current_pos[0] + 1.0, current_pos[1]])
                    else:
                        # 计算最近点
                        closest_point = np.array(obstacle_poly.exterior.interpolate(obstacle_poly.exterior.project(current_point)).coords[0])
                    
                    # 计算斥力
                    direction = current_pos - closest_point
                    static_rep_force = self._calculate_static_obstacle_repulsion(
                        direction, distance, self.config['static_safety_distance'], 
                        self.config['static_influence_distance'])
                    total_repulsive += static_rep_force
            except Exception as e:
                continue
        
        # 2. 处理动态障碍物 - 所有动态障碍物都产生斥力
        high_threat_obstacles = []
        for obstacle in dynamic_obstacles:
            try:
                # 检查是否为DynamicObstacle对象
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size') and \
                   hasattr(obstacle, 'direction') and hasattr(obstacle, 'speed'):
                    
                    # 获取动态障碍物属性
                    obs_position = np.array(obstacle.position)
                    obs_size = float(obstacle.size)
                    obs_direction = np.array(obstacle.direction)
                    obs_speed = float(obstacle.speed)
                    
                    # 计算动态障碍物的预测位置
                    velocity = obs_direction * obs_speed
                    predicted_position = obs_position + velocity * self.config['prediction_time']
                    
                    # 计算到预测位置的距离和方向
                    direction_to_predicted = current_pos - predicted_position
                    distance = np.linalg.norm(direction_to_predicted)
                    
                    # 智能斥力方向计算
                    if distance > 0:
                        # 基础方向：远离预测位置
                        base_direction = direction_to_predicted / distance
                        
                        # 获取障碍物运动方向
                        obs_velocity_dir = obs_direction / np.linalg.norm(obs_direction) if np.linalg.norm(obs_direction) > 0 else np.array([1.0, 0.0])
                        
                        # 计算障碍物朝向当前点的方向
                        to_current_dir = current_pos - obs_position
                        if np.linalg.norm(to_current_dir) > 0:
                            to_current_dir = to_current_dir / np.linalg.norm(to_current_dir)
                        
                        # 判断是否为对向行驶（障碍物朝向当前位置移动）
                        approach_factor = np.dot(obs_velocity_dir, to_current_dir)
                        
                        # 调试对向检测
                        print(f"调试对向检测 - approach_factor: {approach_factor:.3f}, "
                              f"obs_velocity_dir: ({obs_velocity_dir[0]:.2f},{obs_velocity_dir[1]:.2f}), "
                              f"to_current_dir: ({to_current_dir[0]:.2f},{to_current_dir[1]:.2f})")
                        
                        if approach_factor < -0.3:  # 对向行驶，需要侧向避障
                            # 对于对向行驶的障碍物，选择真正的左右方向避障
                            # 而不是垂直于障碍物运动方向
                            
                            # 计算从障碍物到当前位置的方向
                            to_current = current_pos - obs_position
                            if np.linalg.norm(to_current) > 0:
                                to_current = to_current / np.linalg.norm(to_current)
                            
                            # 计算垂直于连接线的侧向方向（真正的左右避障）
                            lateral_dir1 = np.array([-to_current[1], to_current[0]])  # 左转90度
                            lateral_dir2 = np.array([to_current[1], -to_current[0]])   # 右转90度
                            
                            # 选择更接近目标的侧向方向
                            target_dir = target_pos - current_pos
                            if np.linalg.norm(target_dir) > 0:
                                target_dir = target_dir / np.linalg.norm(target_dir)
                            
                            lateral_score1 = np.dot(lateral_dir1, target_dir)
                            lateral_score2 = np.dot(lateral_dir2, target_dir)
                            
                            # 选择得分更高的侧向方向
                            if lateral_score1 > lateral_score2:
                                preferred_lateral = lateral_dir1
                            else:
                                preferred_lateral = lateral_dir2
                            
                            print(f"调试侧向避障 - to_current: ({to_current[0]:.2f},{to_current[1]:.2f}), "
                                  f"lateral_dir1: ({lateral_dir1[0]:.2f},{lateral_dir1[1]:.2f}), "
                                  f"lateral_dir2: ({lateral_dir2[0]:.2f},{lateral_dir2[1]:.2f}), "
                                  f"preferred_lateral: ({preferred_lateral[0]:.2f},{preferred_lateral[1]:.2f})")
                            
                            # 混合基础方向和侧向方向
                            direction = 0.2 * base_direction + 0.8 * preferred_lateral  # 增强侧向权重
                            direction = direction / np.linalg.norm(direction)
                            
                            print(f"调试混合方向 - base_direction: ({base_direction[0]:.2f},{base_direction[1]:.2f}), "
                                  f"final_direction: ({direction[0]:.2f},{direction[1]:.2f})")
                        else:
                            # 非对向行驶，使用基础方向
                            direction = base_direction
                            print(f"调试基础避障 - direction: ({direction[0]:.2f},{direction[1]:.2f})")
                    else:
                        direction = np.array([1.0, 0.0])
                    
                    # 计算威胁等级 - 使用预测位置进行更准确的威胁评估
                    threat_level = self._calculate_threat_level(
                        current_pos, predicted_position, obs_size, obs_speed, obs_direction)
                    
                    # 计算动态障碍物斥力 - 所有动态障碍物都计算斥力
                    dynamic_rep_force = self._calculate_dynamic_obstacle_repulsion(
                        direction, distance, obs_size, threat_level, 
                        self.config['dynamic_safety_distance'],
                        self.config['dynamic_influence_distance'])
                    total_repulsive += dynamic_rep_force
                    
                    # 记录高威胁障碍物 - 降低阈值，更容易触发智能避障
                    if threat_level > 0.3:  # 降低高威胁阈值从0.7到0.3
                        high_threat_obstacles.append({
                            'obstacle': obstacle,
                            'position': obs_position,
                            'predicted_position': predicted_position,
                            'direction': direction,
                            'distance': distance,
                            'threat_level': threat_level,
                            'velocity': velocity
                        })
            except Exception as e:
                continue
        
        # 3. 高威胁障碍物的智能侧向避障
        if high_threat_obstacles:
            # 选择威胁最高的障碍物
            most_threatening = max(high_threat_obstacles, key=lambda x: x['threat_level'])
            
            # 计算智能侧向避障力
            lateral_avoidance_force = self._calculate_lateral_avoidance_force(
                current_pos, most_threatening, target_pos)
            
            # 根据威胁等级调整侧向避障力的权重
            lateral_weight = min(1.0, most_threatening['threat_level'])
            total_repulsive += lateral_avoidance_force * lateral_weight
        
        return total_repulsive
    
    def _calculate_static_obstacle_repulsion(self, direction: np.ndarray, distance: float, 
                                           safety_distance: float, influence_distance: float) -> np.ndarray:
        """
        计算静态多边形障碍物的斥力
        
        Args:
            direction: 从障碍物最近点指向当前位置的方向向量
            distance: 当前位置到障碍物的距离
            safety_distance: 安全距离
            influence_distance: 影响距离
            
        Returns:
            斥力向量
        """
        # 如果距离过小，设置一个小值避免除零
        if distance < 1e-6:
            return np.array([1.0, 0.0]) * self.config['repulsive_gain'] * 10.0
        
        # 标准化方向向量
        normalized_dir = direction / np.linalg.norm(direction)
        
        # 最大影响距离
        max_influence_distance = safety_distance + influence_distance
        
        if distance < safety_distance:
            # 在安全距离内，斥力随距离减小而增大
            force_magnitude = self.config['repulsive_gain'] * (1.0 / distance - 1.0 / safety_distance) / (distance ** 2)
            return normalized_dir * force_magnitude
        elif distance < max_influence_distance:
            # 在影响范围内，斥力随距离增大而衰减
            decay_factor = (max_influence_distance - distance) / (max_influence_distance - safety_distance)
            force_magnitude = self.config['repulsive_gain'] * decay_factor / (distance ** 2)
            return normalized_dir * force_magnitude
        else:
            # 超出影响范围，无斥力
            return np.zeros(2)
    
    def _calculate_dynamic_obstacle_repulsion(self, direction: np.ndarray, distance: float, 
                                            obs_size: float, threat_level: float, 
                                            safety_distance: float, influence_distance: float) -> np.ndarray:
        """
        计算动态障碍物的斥力（优化版）
        
        Args:
            direction: 从障碍物预测位置指向当前位置的方向向量
            distance: 当前位置到障碍物预测位置的距离
            obs_size: 障碍物大小
            threat_level: 障碍物威胁等级
            safety_distance: 安全距离
            influence_distance: 影响距离
            
        Returns:
            斥力向量
        """
        # 如果距离过小，设置紧急避障斥力
        if distance < 1e-6:
            return np.array([1.0, 0.0]) * self.config['repulsive_gain'] * 30.0 * threat_level
        
        # 标准化方向向量
        normalized_dir = direction / np.linalg.norm(direction)
        
        # 安全距离 = 障碍物大小 + 配置安全距离
        safe_distance = obs_size + safety_distance
        
        # 最大影响距离
        max_influence_distance = safe_distance + influence_distance
        
        # 基础斥力系数乘以威胁等级
        effective_gain = self.config['repulsive_gain'] * threat_level
        
        # 优化斥力计算函数，使动态障碍物也能提前感知
        if distance < max_influence_distance:
            # 1. 基础斥力
            base_repulsion = effective_gain / (distance ** 2)
            
            # 2. 距离衰减因子 - 使用更平滑的三次函数衰减
            distance_ratio = distance / max_influence_distance
            distance_factor = (1.0 - distance_ratio ** 3) ** 3
            
            # 3. 安全距离内的强化因子
            if distance < safe_distance:
                # 在安全距离内，根据距离程度分级强化斥力
                if distance < safe_distance * 0.5:
                    safety_factor = 4.0  # 极近距离，强力避障
                else:
                    safety_factor = 3.0  # 近距离，强化避障
            else:
                safety_factor = 1.0
            
            # 4. 动态障碍物速度预测因子 - 根据威胁等级动态调整
            # 高威胁障碍物需要更早的避障反应
            prediction_factor = 1.0 + threat_level * 1.5  # 1.0-2.5倍增强（降低）
            
            # 5. 侧向避障增强因子 - 优先选择侧向避障而非直接后退
            # 这有助于保持路径的连贯性
            lateral_factor = 1.2  # 适度的侧向避障（降低）
            
            # 6. 动态障碍物额外增强因子 - 确保动态障碍物斥力足够强
            dynamic_boost_factor = 1.5  # 适度增强（降低）
            
            # 综合计算斥力大小
            force_magnitude = base_repulsion * distance_factor * safety_factor * prediction_factor * lateral_factor * dynamic_boost_factor
            
            # 6. 添加侧向避障分量 - 当障碍物直接迎面而来时，增强侧向避障
            # 这里可以进一步优化，添加垂直于运动方向的斥力分量
            final_force = normalized_dir * force_magnitude
            
            return final_force
        else:
            # 超出影响范围，无斥力
            return np.zeros(2)
    
    def _calculate_threat_level(self, current_pos: np.ndarray, obs_pos: np.ndarray, 
                              obs_size: float, obs_speed: float, obs_direction: np.ndarray) -> float:
        """
        计算动态障碍物的威胁等级（优化版）
        
        Args:
            current_pos: 当前位置
            obs_pos: 障碍物位置
            obs_size: 障碍物大小
            obs_speed: 障碍物速度
            obs_direction: 障碍物方向向量
            
        Returns:
            威胁等级（0-1之间）
        """
        # 计算距离
        dx = obs_pos[0] - current_pos[0]
        dy = obs_pos[1] - current_pos[1]
        distance = math.sqrt(dx **2 + dy** 2)
        
        # 距离过远，威胁等级为0
        if distance > self.config['dynamic_influence_distance'] + obs_size:
            return 0.0
        
        # 计算障碍物与当前位置的方向向量
        if distance > 0:
            current_direction = np.array([dx, dy]) / distance
        else:
            current_direction = np.array([1.0, 0.0])
        
        # 计算方向夹角余弦（越接近1表示方向越一致，威胁越大）
        direction_cos = max(-1.0, min(1.0, np.dot(obs_direction, current_direction)))
        
        # 优化威胁等级组件
        size_factor = min(1.0, obs_size / 30.0)  # 降低大小因子的分母，更容易达到高值
        speed_factor = min(1.0, obs_speed / 5.0)   # 降低速度因子的分母，更容易达到高值
        distance_factor = 1.0 - min(1.0, distance / (self.config['dynamic_influence_distance'] + obs_size))
        
        # 方向因子优化：对向移动威胁更大
        if direction_cos > 0.7:  # 同向移动，威胁较小
            direction_factor = 0.2
        elif direction_cos < -0.7:  # 对向移动，威胁很大
            direction_factor = 1.0
        else:  # 侧向移动，中等威胁
            direction_factor = 0.7
        
        # 新增：相对速度威胁因子
        # 如果障碍物朝向当前位置移动，威胁更大
        relative_speed_factor = speed_factor * max(0.0, -direction_cos)  # 对向时为正，同向时为0
        
        # 新增：碰撞概率因子
        # 基于距离和速度计算碰撞概率，降低时间阈值
        collision_time = distance / (obs_speed + 1e-6)  # 避免除零
        if collision_time < 3.0:  # 3秒内可能碰撞（从2秒增加到3秒）
            collision_probability_factor = 1.0 - min(1.0, collision_time / 3.0)
        else:
            collision_probability_factor = 0.0
        
        # 综合威胁等级（重新调整权重，让威胁更容易达到高值）
        threat_level = (
            self.config['threat_distance_weight'] * distance_factor * 0.25 +  # 距离权重
            self.config['threat_speed_weight'] * speed_factor * 0.25 +        # 速度权重
            self.config['threat_direction_weight'] * direction_factor * 0.25 + # 方向权重
            0.15 * relative_speed_factor +                                    # 相对速度权重
            0.10 * collision_probability_factor                                 # 碰撞概率权重
        ) * size_factor
        
        # 归一化到0-1范围，使用更宽松的最大值
        max_threat = 0.25 + 0.25 + 0.25 + 0.15 + 0.10  # 最大可能权重和
        threat_level = min(1.0, threat_level / max_threat)
        
        # 进一步增强威胁等级：对近距离和高速度进行额外加成
        if distance < 100.0:  # 近距离威胁加成
            threat_level = min(1.0, threat_level * 1.5)
        if obs_speed > 4.0:   # 高速威胁加成
            threat_level = min(1.0, threat_level * 1.3)
        
        return threat_level
    
    def _calculate_lateral_avoidance_force(self, current_pos: np.ndarray, threat_info: dict, target_pos: np.ndarray) -> np.ndarray:
        """
        计算智能侧向避障力
        
        Args:
            current_pos: 当前位置
            threat_info: 威胁障碍物信息
            target_pos: 目标位置
            
        Returns:
            侧向避障力向量
        """
        obs_velocity = threat_info['velocity']
        obs_position = threat_info['position']
        
        # 计算障碍物运动方向的垂直方向（侧向避障方向）
        obs_speed = np.linalg.norm(obs_velocity)
        if obs_speed > 1e-6:
            obs_direction_normalized = obs_velocity / obs_speed
            # 计算两个垂直方向
            lateral_direction1 = np.array([-obs_direction_normalized[1], obs_direction_normalized[0]])
            lateral_direction2 = np.array([obs_direction_normalized[1], -obs_direction_normalized[0]])
        else:
            # 如果障碍物静止，使用从障碍物指向当前位置的方向的垂直方向
            to_current = current_pos - obs_position
            if np.linalg.norm(to_current) > 1e-6:
                to_current_normalized = to_current / np.linalg.norm(to_current)
                lateral_direction1 = np.array([-to_current_normalized[1], to_current_normalized[0]])
                lateral_direction2 = np.array([to_current_normalized[1], -to_current_normalized[0]])
            else:
                lateral_direction1 = np.array([1.0, 0.0])
                lateral_direction2 = np.array([0.0, 1.0])
        
        # 评估两个侧向方向，选择更接近目标的方向
        candidate_pos1 = current_pos + lateral_direction1 * 50.0
        candidate_pos2 = current_pos + lateral_direction2 * 50.0
        
        # 计算到目标的距离
        dist_to_target1 = np.linalg.norm(candidate_pos1 - target_pos)
        dist_to_target2 = np.linalg.norm(candidate_pos2 - target_pos)
        
        # 选择更接近目标的侧向方向
        if dist_to_target1 < dist_to_target2:
            preferred_lateral_dir = lateral_direction1
        else:
            preferred_lateral_dir = lateral_direction2
        
        # 计算侧向避障力的大小
        # 威胁越高，侧向避障力越大
        threat_level = threat_info['threat_level']
        distance = threat_info['distance']
        
        # 侧向避障力随威胁等级和距离变化
        if distance < 80.0:  # 很近的情况
            lateral_force_magnitude = self.config['repulsive_gain'] * threat_level * 1.2  # 增强到1.2倍
        elif distance < 150.0:  # 中等距离
            lateral_force_magnitude = self.config['repulsive_gain'] * threat_level * 0.8  # 增强到0.8倍
        else:  # 较远距离
            lateral_force_magnitude = self.config['repulsive_gain'] * threat_level * 0.5  # 增强到0.5倍
        
        # 返回侧向避障力
        return preferred_lateral_dir * lateral_force_magnitude
    
    def discretize_path(self, path, distance_threshold=50.0):
        """
        对路径进行稀疏化处理
        
        Args:
            path: 原始路径点列表
            distance_threshold: 距离阈值，小于该阈值的点将被合并
            
        Returns:
            稀疏化后的路径点列表
        """
        if len(path) <= 2:
            return path
            
        sparse_path = [path[0]]  # 保留起点
        last_point = np.array(path[0])
        
        for point in path[1:-1]:  # 跳过起点和终点，单独处理
            current_point = np.array(point)
            distance = np.linalg.norm(current_point - last_point)
            
            if distance > distance_threshold:
                sparse_path.append(point)
                last_point = current_point
                
        sparse_path.append(path[-1])  # 保留终点
        return sparse_path
