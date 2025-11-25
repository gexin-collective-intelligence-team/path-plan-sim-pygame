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
            'attractive_gain': 6.0,        # 引力增益系数（略增，保持目标导向性）
            'min_distance': 5.0,           # 到达目标的最小距离阈值
            
            # 斥力参数 - 扩大避障范围
            'repulsive_gain': 1200.0,      # 斥力增益系数（增强避障力度）
            'static_safety_distance': 40.0, # 静态障碍物安全距离（从30增加到40）
            'dynamic_safety_distance': 80.0, # 动态障碍物安全距离（从60增加到80）
            'static_influence_distance': 90.0, # 静态障碍物影响距离（从70增加到90）
            'dynamic_influence_distance': 250.0, # 动态障碍物影响距离（从200增加到250）
            'prediction_time': 2.5,        # 动态障碍物基础预测时间（从2.0增加到2.5）
            'threat_distance_weight': 2.0, # 距离权重
            'threat_speed_weight': 1.5,    # 速度权重
            'threat_direction_weight': 0.5, # 方向权重
            'threat_size_weight': 0.6,      # 尺寸权重
            'threat_threshold': 0.15,       # 威胁阈值（从0.12增加到0.15，更早触发避障）
            'threat_gain_scale': 3.5,       # 高威胁时增强斥力的比例（从3.0增加到3.5）
            'threat_range_scale': 1.8,      # 高威胁时扩大作用范围的比例（从1.5增加到1.8）
            'braking_distance_base': 25.0,  # 基础制动距离（从20增加到25）
            'braking_distance_coeff': 10.0, # 与速度相关的制动距离系数（从8增加到10）
            
            # 移动参数 - 提高速度和步长
            'max_step': 15.0,              # 最大步长（从10增加到15，提高移动速度）
            'max_linear_speed': 35.0,      # 物理最大速度上限（从20提高到35）
            'max_linear_acc': 12.0,        # 物理最大加速度（从8增加到12）
            'max_angular_velocity': np.radians(25.0),  # 每步最大角速度（从18°增加到25°）
            'max_angular_acceleration': np.radians(45.0),  # 每步最大角加速度（从30°增加到45°）
            
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
        self.prev_velocity = np.zeros(2)
        self.prev_heading = np.array([1.0, 0.0])
        self.prev_angle_change = 0.0

        # 调试/分析日志：记录每一步的威胁指数和最近障碍物距离
        # 每个元素是一个 dict，包含 step, pos, min_dist, static_threat, dynamic_threat, total_threat
        self.debug_logs = []
    def reset_planner(self):
        """手动重置规划器状态，确保到达目标点后立即重置所有相关参数"""
        # 直接设置属性而非删除，确保一致的初始化状态
        self.current_target_idx = 0
        self.is_initialized = False
        self.last_target_pos = None
        self.stuck_counter = 0
        self.last_distance_to_target = float('inf')
        self.goal_reached = False  # 明确设置为False，确保重新开始规划
        self.goal_threshold = self.config.get('min_distance', 5.0)  # 初始化goal_threshold
        self.max_stuck_steps = 10  # 初始化max_stuck_steps
        self.history_positions = []  # 清空历史位置
        self.use_temp_goal = False
        self.temp_goal = None
        self.temp_goal_count = 0
        self.prev_velocity = np.zeros(2)
        self.prev_heading = np.array([1.0, 0.0])
        self.prev_angle_change = 0.0
        print("♻️ APF 规划器状态已完全重置，可重新开始路径规划。")
    def plan(self, current_pos, target_pos, static_obstacles, dynamic_obstacles, velocity=None, global_path=None):
        """
        改进版APF路径规划（增强版）
        ✅ 保留振荡检测与临时目标点机制
        ✅ 新增目标点可达性检测与自动跳过机制
        ✅ 添加goal_reached标志，到达目标点后立即停止计算并保持在目标位置
        """

        # ---------- 基本初始化 ----------
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)
        
        # 检查是否已到达目标点，如果是则直接返回目标点，不再进行任何计算
        if hasattr(self, "goal_reached") and self.goal_reached:
            # 如果传入非空全局路径，返回最后一个目标点；否则返回当前目标点
            if global_path is not None and len(global_path) > 0:
                final_target = np.array(global_path[-1])
                return final_target.copy()
            # 当 global_path 为 None 或空列表时，直接以传入的 target_pos 作为终点
            return target_pos.copy()  # 返回目标点的副本，避免引用问题
        
        # ✅ 仅首次运行时初始化，或确保关键属性存在
        if not hasattr(self, "is_initialized") or not self.is_initialized:
            self.current_target_idx = 0
            self.last_distance_to_target = float('inf')
            self.stuck_counter = 0
            self.max_stuck_steps = 10       # 判定阻挡的连续步数
            self.goal_threshold = self.config.get('min_distance', 5.0)
            self.goal_reached = False  # 初始化目标到达标志
            self.is_initialized = True
            print("🧭 规划器状态已初始化。")
        
        # 安全检查：确保goal_threshold存在（防止某些边缘情况）
        if not hasattr(self, 'goal_threshold'):
            self.goal_threshold = self.config.get('min_distance', 5.0)

        # 若传入非空全局路径，则根据索引更新目标点，并跳过被障碍物覆盖或明显不可达的点
        if global_path is not None and len(global_path) > 0:
            # 全局路径模式下使用更大的到达阈值
            global_min_distance = 12.0  # 全局路径下的最小距离阈值
            local_min_distance = self.config.get('min_distance', 5.0)  # 单目标模式的最小距离阈值
            
            # 动态调整goal_threshold
            self.goal_threshold = global_min_distance if len(global_path) > 2 else local_min_distance
            
            while True:
                if self.current_target_idx >= len(global_path):
                    print("✅ 所有目标点已到达或被跳过，规划结束。")
                    self.goal_reached = True
                    return current_pos
                candidate_target = np.array(global_path[self.current_target_idx])
                if not self._is_target_blocked(current_pos, candidate_target, static_obstacles, dynamic_obstacles):
                    target_pos = candidate_target
                    break
                print(f"⚠️ 全局路径点 {self.current_target_idx} 被障碍物覆盖或不可达，自动跳过。")
                self.current_target_idx += 1

        # ---------- 到目标的距离 ----------
        distance_to_target = np.linalg.norm(target_pos - current_pos)

        # ---------- 特殊策略：优先右侧绕行大静态障碍物（作为临时目标注入） ----------
        # 当当前点到目标点的直线会穿过某个静态障碍物，且距离该障碍物中心较近时，
        # 计算一个“右侧切向点”，但不直接作为下一步位置，而是作为 temp_goal，
        # 交给原有的临时目标机制和运动学约束去平滑逼近。
        tangent_pos = self._try_tangent_around_static_obstacle(current_pos, target_pos, static_obstacles)
        if tangent_pos is not None and not self.use_temp_goal:
            self.use_temp_goal = True
            self.temp_goal = np.array(tangent_pos, dtype=float)
            self.temp_goal_count = 0

        # ---------- 到达检测 ----------
        if distance_to_target < self.goal_threshold:
            print(f"🎯 到达目标点 {self.current_target_idx}，切换下一个。")
            if global_path is not None and len(global_path) > 0:
                self.current_target_idx += 1
                if self.current_target_idx < len(global_path):
                    # 切换下一个目标点继续规划
                    return self.plan(current_pos, global_path[self.current_target_idx],
                                    static_obstacles, dynamic_obstacles, velocity, global_path)
                else:
                    print("✅ 最终目标点已到达，路径规划完成，立即设置到达标志。")
                    self.goal_reached = True  # 设置目标到达标志
                    return target_pos.copy()  # 返回目标点的副本，确保精确到达
            else:
                # 无全局路径输入时直接停止
                print("✅ 目标点已到达，路径规划完成，立即设置到达标志。")
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
        current_velocity = np.array(velocity) if velocity is not None else np.zeros(2)
        repulsive_force, max_static_threat, max_dynamic_threat, min_obstacle_distance = self._calculate_repulsive_force(
            current_pos,
            static_obstacles,
            dynamic_obstacles,
            current_velocity
        )
        
        # 全局路径模式下增强引力，减少过度避障导致的减速
        if global_path is not None and len(global_path) > 2:
            # 在有全局路径时，适当增强引力，保持前进动力
            attractive_force *= 1.3  # 增强30%的引力
            # 如果威胁不大，适当减弱斥力，避免过度减速
            if max(max_static_threat, max_dynamic_threat) < 0.3:
                repulsive_force *= 0.7  # 减弱30%的斥力
        
        total_force = attractive_force + repulsive_force

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

        if self.stuck_counter > self.max_stuck_steps:
            print(f"⚠️ 当前目标点 {self.current_target_idx} 被阻挡，自动跳过/启用绕行策略。")
            self.stuck_counter = 0
            self.last_distance_to_target = float('inf')

            if global_path is not None and len(global_path) > 0:
                self.current_target_idx += 1
                if self.current_target_idx < len(global_path):
                    # 跳过当前目标点，继续规划下一个
                    return self.plan(current_pos, global_path[self.current_target_idx],
                                    static_obstacles, dynamic_obstacles, velocity, global_path)
                else:
                    print("⚠️ 所有目标点均无法到达，规划终止。")
                    return current_pos
            else:
                # 无全局路径场景：当前终点长期不可达，改为启用临时目标点进行绕行
                print("⚠️ 无全局路径场景下终点长期不可达，启用临时目标点绕行。")
                temp = self._select_temp_goal(current_pos, target_pos, static_obstacles, dynamic_obstacles)
                if temp is not None:
                    self.use_temp_goal = True
                    self.temp_goal = temp
                    self.temp_goal_count = 0
                # 本次不再前进，由下一次调用根据临时目标点重新规划
                return current_pos

        next_pos = self._enforce_motion_constraints(current_pos, next_pos, current_velocity)

        # ---------- 记录当前步的威胁与最近距离，用于后续分析 ----------
        try:
            total_threat = max(max_static_threat, max_dynamic_threat)
            log_entry = {
                'step': len(self.debug_logs),
                'x': float(current_pos[0]),
                'y': float(current_pos[1]),
                'min_obstacle_distance': float(min_obstacle_distance) if min_obstacle_distance is not None else None,
                'static_threat': float(max_static_threat),
                'dynamic_threat': float(max_dynamic_threat),
                'total_threat': float(total_threat),
            }
            self.debug_logs.append(log_entry)
        except Exception:
            pass

        return next_pos

    def _try_tangent_around_static_obstacle(self, current_pos: np.ndarray, target_pos: np.ndarray, static_obstacles: List) -> Optional[np.ndarray]:
        """在特殊几何情况下优先采用“右侧绕行”的切向步长。

        条件：
        - 当前点到目标点的连线与某个静态障碍物（多边形）相交；
        - 当前点距离该障碍物中心小于（等效半径 + 安全裕度）。

        满足条件时：
        - 计算当前点相对于障碍物中心的方向向量 r；
        - 在平面上以 r 为半径向量，取右手法则方向的切向向量 t = (r_y, -r_x)；
        - 沿 t 方向迈出一步（步长不超过 max_step），返回该位置。
        """
        if not static_obstacles:
            return None

        try:
            line_to_goal = LineString([tuple(current_pos), tuple(target_pos)])
        except Exception:
            return None

        best_step = None
        best_obstacle_dist = float('inf')

        for obstacle in static_obstacles:
            try:
                if len(obstacle) < 3:
                    continue
                if obstacle[0] != obstacle[-1]:
                    obstacle_poly = Polygon(obstacle + [obstacle[0]])
                else:
                    obstacle_poly = Polygon(obstacle)

                # 如果当前到目标的直线根本不碰这个障碍物，就不需要特别处理
                try:
                    if not line_to_goal.intersects(obstacle_poly):
                        continue
                except Exception:
                    continue

                # 估计障碍物中心和等效半径
                center = np.array(obstacle_poly.centroid.coords[0], dtype=float)
                area = max(obstacle_poly.area, 0.0)
                approx_radius = math.sqrt(area / (math.pi + 1e-6))

                # 当前点到中心的距离
                vec_to_center = current_pos - center
                dist_to_center = np.linalg.norm(vec_to_center)

                # 只有在足够靠近障碍物时才启动切向策略
                safety_margin = max(20.0, 0.5 * approx_radius)
                trigger_dist = approx_radius + safety_margin
                if approx_radius < 1e-3 or dist_to_center > trigger_dist:
                    continue

                # 选择“右侧”绕行：以从中心指向当前点的向量为 r，
                # 右侧切向 t = (r_y, -r_x)
                if dist_to_center < 1e-6:
                    # 极端情况：恰好在中心附近，随便给一个方向
                    radial_dir = np.array([1.0, 0.0])
                else:
                    radial_dir = vec_to_center / dist_to_center

                tangent_dir = np.array([radial_dir[1], -radial_dir[0]])
                if np.linalg.norm(tangent_dir) < 1e-6:
                    continue
                tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)

                step_len = min(self.config['max_step'], max(10.0, 0.5 * trigger_dist))
                candidate_pos = current_pos + tangent_dir * step_len

                # 确保候选点不在障碍物内部，如在内部则沿径向方向稍微推出去
                try:
                    cand_point = Point(float(candidate_pos[0]), float(candidate_pos[1]))
                    if obstacle_poly.contains(cand_point):
                        # 往外推到障碍边界稍远处
                        push_len = max(5.0, 0.2 * approx_radius)
                        candidate_pos = candidate_pos + radial_dir * push_len
                except Exception:
                    pass

                # 记录距离最近的那个障碍物对应的切向步长
                if dist_to_center < best_obstacle_dist:
                    best_obstacle_dist = dist_to_center
                    best_step = candidate_pos
            except Exception:
                continue

        return best_step

    
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
                        velocity = np.array(obstacle.direction) * float(obstacle.speed)
                        predicted_position = np.array(obstacle.position) + velocity * self.config['prediction_time']
                    else:
                        predicted_position = np.array(obstacle.position)
                    
                    # 创建表示障碍物的点（简化为点碰撞检测）
                    obs_point = Point(predicted_position[0], predicted_position[1])
                    
                    # 检查线段与障碍物的距离是否小于安全距离
                    if line.distance(obs_point) < float(obstacle.size) + self.config['dynamic_safety_distance']:
                        return True
            except Exception:
                continue
        
        return False
    
    def _is_target_blocked(self, current_pos, target_pos, static_obstacles, dynamic_obstacles):
        """判断全局路径上的某个目标点是否被障碍物直接覆盖

        注意：这里只关心“点本身是否落在障碍物里/太近”，
        不再用 current_pos->target_pos 直线是否穿过障碍 来决定是否跳点，
        避免把障碍物后面的所有路径点都判为不可达。
        """
        # 1) 静态多边形：目标点是否在多边形内部或边界上
        try:
            target_point = Point(float(target_pos[0]), float(target_pos[1]))
        except Exception:
            return False

        for obstacle in static_obstacles:
            try:
                if len(obstacle) >= 3:
                    if obstacle[0] != obstacle[-1]:
                        obstacle_poly = Polygon(obstacle + [obstacle[0]])
                    else:
                        obstacle_poly = Polygon(obstacle)
                    if obstacle_poly.contains(target_point) or obstacle_poly.touches(target_point):
                        return True
            except Exception:
                continue

        # 2) 动态障碍：目标点是否落在动态障碍的安全半径附近
        for obstacle in dynamic_obstacles:
            try:
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    obs_pos = np.array(obstacle.position, dtype=float)
                    distance = np.linalg.norm(obs_pos - np.array(target_pos, dtype=float))
                    # 这里用较保守的判定：障碍本体半径 + 一小段安全距离
                    safe_dist = float(obstacle.size) + self.config['dynamic_safety_distance'] * 0.5
                    if distance <= safe_dist:
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
    
    def _calculate_repulsive_force(self, current_pos: np.ndarray, static_obstacles: List,
                                   dynamic_obstacles: List, velocity: np.ndarray):
        """
        计算所有障碍物的斥力
        
        Args:
            current_pos: 当前位置
            static_obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
            velocity: 当前速度向量
            
        Returns:
            总斥力向量
        """
        total_repulsive = np.zeros(2)
        max_static_threat = 0.0
        max_dynamic_threat = 0.0
        min_obstacle_distance = None
        current_point = Point(current_pos[0], current_pos[1])
        braking_distance = self._compute_braking_distance(velocity)
        threat_threshold = self.config['threat_threshold']
        
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

                    # 记录最近障碍物距离
                    if min_obstacle_distance is None or distance < min_obstacle_distance:
                        min_obstacle_distance = float(distance)

                    # 根据多边形面积估计一个“等效半径”，用于放大小障碍物的安全距离/影响范围
                    # area ≈ π R^2  =>  R ≈ sqrt(area / π)
                    try:
                        approx_radius = math.sqrt(max(obstacle_poly.area, 0.0) / (math.pi + 1e-6))
                    except Exception:
                        approx_radius = 0.0
                    # 对于非常小的障碍物，近似为 0；对于大障碍物，此值显著增大
                    # 系数适当加大，以拉开大小障碍物的“提前躲避”距离差异
                    size_safety_extra = 1.0 * approx_radius
                    size_influence_extra = 2.0 * approx_radius
                    
                    # 获取最近点（用于确定斥力方向）
                    if current_point.distance(obstacle_poly) < 1e-6:
                        # 如果点在多边形内部，使用多边形边界的法线方向
                        closest_point = np.array([current_pos[0] + 1.0, current_pos[1]])
                    else:
                        # 计算最近点
                        closest_point = np.array(obstacle_poly.exterior.interpolate(obstacle_poly.exterior.project(current_point)).coords[0])
                    
                    # 计算斥力
                    direction = current_pos - closest_point
                    threat_index = self._compute_static_threat_index(
                        current_pos,
                        obstacle_poly,
                        distance,
                        braking_distance
                    )
                    if threat_index > max_static_threat:
                        max_static_threat = float(threat_index)
                    if threat_index < threat_threshold:
                        # 威胁低于阈值：认为对当前路径基本无影响，不产生斥力
                        continue

                    # 对刚超过阈值的威胁做“软启动”：
                    # threat_index 在 [threshold, 1] 映射到 [0, 1]
                    norm_threat = (threat_index - threat_threshold) / (1.0 - threat_threshold + 1e-6)
                    # 低威胁（norm_threat 接近 0）斥力非常小，高威胁才被放大
                    soft_factor = norm_threat ** 2
                    gain_scale = 1.0 + self.config['threat_gain_scale'] * soft_factor
                    range_scale = 1.0 + self.config['threat_range_scale'] * soft_factor

                    # 静态安全距离/影响距离随障碍物尺寸增加：
                    #   小障碍物几乎不变，大障碍物的安全圈明显增大，从而更早开始躲避
                    static_safe = self.config['static_safety_distance'] + size_safety_extra
                    static_influence = (self.config['static_influence_distance'] + size_influence_extra) * range_scale

                    static_rep_force = self._calculate_static_obstacle_repulsion(
                        direction, distance, static_safe,
                        static_influence,
                        self.config['repulsive_gain'] * gain_scale)
                    total_repulsive += static_rep_force
            except Exception as e:
                continue
        
        # 2. 处理动态障碍物
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
                    
                    # 计算动态障碍物速度
                    obs_velocity = obs_direction * obs_speed

                    # --- 基于相对速度的自适应预测时间（TTC） ---
                    # 相对位置与相对速度
                    rel_pos = obs_position - current_pos
                    rel_vel = obs_velocity - velocity

                    # 默认预测时间：配置中的 prediction_time
                    t_predict = self.config['prediction_time']

                    # 若存在明显的相对速度，基于 TTC 估算可能的碰撞时间
                    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
                    if rel_speed_sq > 1e-6:
                        # 线性模型下的最小距离时间 t* = - (r·v) / |v|^2
                        t_star = - float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
                        # 只关心未来一段时间内的潜在碰撞（t>0）
                        if t_star > 0.0:
                            # 将预测时间限制在 [0.3, 1.5 * prediction_time]
                            t_min = 0.3
                            t_max = 1.5 * self.config['prediction_time']
                            t_predict = max(t_min, min(t_star, t_max))

                    # 使用自适应预测时间得到障碍物未来位置
                    predicted_position = obs_position + obs_velocity * t_predict
                    
                    # 计算到预测位置的距离和方向（径向斥力方向）
                    direction = current_pos - predicted_position
                    distance = np.linalg.norm(direction)
                    
                    threat_level = self._compute_dynamic_threat_index(
                        current_pos,
                        obs_position,
                        predicted_position,
                        obs_size,
                        obs_velocity,
                        distance,
                        braking_distance,
                        velocity
                    )
                    if threat_level < threat_threshold:
                        # 低威胁动态障碍只作为背景存在，不施加斥力
                        continue

                    # 同样对动态障碍的威胁做软启动
                    dyn_norm_threat = (threat_level - threat_threshold) / (1.0 - threat_threshold + 1e-6)
                    dyn_soft_factor = dyn_norm_threat ** 2
                    # 高威胁才显著放大斥力和作用范围
                    gain_scale = 1.0 + self.config['threat_gain_scale'] * dyn_soft_factor
                    range_scale = 1.0 + self.config['threat_range_scale'] * dyn_soft_factor
                    dynamic_rep_force = self._calculate_dynamic_obstacle_repulsion(
                        direction, distance, obs_size, threat_level, 
                        self.config['dynamic_safety_distance'],
                        self.config['dynamic_influence_distance'] * range_scale,
                        self.config['repulsive_gain'] * gain_scale)

                    # 额外的切向（侧向）避障力：用于打破正面相向时引力/斥力共线的问题
                    tangent_force = np.zeros(2)
                    try:
                        # 只有在障碍基本位于前方且威胁较大时才启用
                        if distance > 1e-3 and threat_level >= threat_threshold * 1.5:
                            # 若当前速度近似为零，则用目标方向代替
                            ref_dir = velocity.copy()
                            if np.linalg.norm(ref_dir) < 1e-3 and global_path is not None and len(global_path) > 0:
                                ref_dir = np.array(global_path[-1]) - current_pos
                            if np.linalg.norm(ref_dir) < 1e-3:
                                ref_dir = np.array([1.0, 0.0])

                            ref_dir = ref_dir / np.linalg.norm(ref_dir)

                            # 前方判定：障碍在 ref_dir 前方且距离不远
                            to_obs = predicted_position - current_pos
                            proj_len = float(np.dot(to_obs, ref_dir))
                            if proj_len > 0:
                                # 使用二维“叉积”符号决定向上绕行还是向下绕行
                                cross_z = ref_dir[0] * to_obs[1] - ref_dir[1] * to_obs[0]
                                # 选择一侧切向单位向量
                                if abs(cross_z) < 1e-6:
                                    # 完全正对，任意给一个垂直方向
                                    tangent_dir = np.array([ref_dir[1], -ref_dir[0]])
                                else:
                                    # 根据相对位置决定左右绕行
                                    sign = np.sign(cross_z)
                                    tangent_dir = sign * np.array([ref_dir[1], -ref_dir[0]])

                                tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)

                                # 切向力大小与威胁等级和距离成反比
                                tangent_strength = self.config['repulsive_gain'] * dyn_soft_factor / max(distance, 5.0)
                                # 稍微缩放，防止侧向力过猛
                                tangent_strength *= 0.3
                                tangent_force = tangent_dir * tangent_strength
                    except Exception:
                        tangent_force = np.zeros(2)

                    total_repulsive += dynamic_rep_force + tangent_force

                    # 记录最近动态障碍距离（用圆心距离减半径的简化近似）
                    try:
                        center_dist = max(0.0, distance - obs_size)
                        if min_obstacle_distance is None or center_dist < min_obstacle_distance:
                            min_obstacle_distance = float(center_dist)
                    except Exception:
                        pass

                    if threat_level > max_dynamic_threat:
                        max_dynamic_threat = float(threat_level)
            except Exception as e:
                continue
        
        return total_repulsive, max_static_threat, max_dynamic_threat, min_obstacle_distance
    
    def _calculate_static_obstacle_repulsion(self, direction: np.ndarray, distance: float, 
                                           safety_distance: float, influence_distance: float,
                                           repulsive_gain: float) -> np.ndarray:
        """
        计算静态多边形障碍物的斥力
        
        Args:
            direction: 从障碍物最近点指向当前位置的方向向量
            distance: 当前位置到障碍物的距离
            safety_distance: 安全距离
            influence_distance: 影响距离
            repulsive_gain: 调整后的斥力增益
            
        Returns:
            斥力向量
        """
        # 如果距离过小，设置一个小值避免除零
        if distance < 1e-6:
            return np.array([1.0, 0.0]) * repulsive_gain * 10.0
        
        # 标准化方向向量
        normalized_dir = direction / np.linalg.norm(direction)
        
        # 最大影响距离
        max_influence_distance = safety_distance + influence_distance
        
        if distance < safety_distance:
            # 在安全距离内，斥力随距离减小而增大
            force_magnitude = repulsive_gain * (1.0 / distance - 1.0 / safety_distance) / (distance ** 2)
            return normalized_dir * force_magnitude
        elif distance < max_influence_distance:
            # 在影响范围内，斥力随距离增大而衰减
            decay_factor = (max_influence_distance - distance) / (max_influence_distance - safety_distance)
            force_magnitude = repulsive_gain * decay_factor / (distance ** 2)
            return normalized_dir * force_magnitude
        else:
            # 超出影响范围，无斥力
            return np.zeros(2)
    
    def _calculate_dynamic_obstacle_repulsion(self, direction: np.ndarray, distance: float, 
                                            obs_size: float, threat_level: float, 
                                            safety_distance: float, influence_distance: float,
                                            repulsive_gain: float) -> np.ndarray:
        """
        计算动态障碍物的斥力
        
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
        # 如果距离过小，设置一个小值避免除零
        if distance < 1e-6:
            return np.array([1.0, 0.0]) * repulsive_gain * 20.0 * max(threat_level, 0.1)
        
        # 标准化方向向量
        normalized_dir = direction / np.linalg.norm(direction)
        
        # 安全距离 = 障碍物大小 + 配置安全距离
        safe_distance = obs_size + safety_distance
        
        # 最大影响距离
        max_influence_distance = safe_distance + influence_distance
        
        # 优化斥力计算函数，使动态障碍物也能提前感知
        if distance < max_influence_distance:
            # 1. 基础斥力
            base_repulsion = repulsive_gain * max(threat_level, 0.1) / (distance ** 2)
            
            # 2. 距离衰减因子 - 使用平滑的二次函数衰减
            distance_factor = (1.0 - (distance / max_influence_distance) ** 2) ** 2
            
            # 3. 安全距离内的强化因子
            if distance < safe_distance:
                # 在安全距离内，进一步增强斥力
                safety_factor = 2.5  # 动态障碍物需要更强的安全因子
            else:
                safety_factor = 1.0
            
            # 4. 动态障碍物速度预测因子
            # 考虑到动态障碍物的运动性，增加预测因子使其更早被感知
            prediction_factor = 1.5
            
            # 综合计算斥力大小
            force_magnitude = base_repulsion * distance_factor * safety_factor * prediction_factor
            return normalized_dir * force_magnitude
        else:
            # 超出影响范围，无斥力
            return np.zeros(2)
    
    def _compute_braking_distance(self, velocity: np.ndarray) -> float:
        speed = np.linalg.norm(velocity)
        return self.config['braking_distance_base'] + self.config['braking_distance_coeff'] * speed
    
    def _compute_static_threat_index(self, current_pos: np.ndarray, obstacle_poly: Polygon,
                                     distance: float, braking_distance: float) -> float:
        # 使用略大的参照长度，避免中等大小障碍物的 size_factor 过早饱和到 1
        ref_length = self.config['static_influence_distance'] * 1.8
        size_factor = min(1.0, obstacle_poly.area / (ref_length ** 2 + 1e-6))
        distance_factor = math.exp(-distance / max(braking_distance, 1.0))
        w_size = self.config['threat_size_weight']
        w_dist = self.config['threat_distance_weight']
        if (w_size + w_dist) == 0:
            return 0.0
        threat = (w_size * size_factor + w_dist * distance_factor) / (w_size + w_dist)
        return float(np.clip(threat, 0.0, 1.0))
    
    def _compute_dynamic_threat_index(self, current_pos: np.ndarray, obs_pos: np.ndarray,
                                      predicted_pos: np.ndarray, obs_size: float,
                                      obs_velocity: np.ndarray, distance: float,
                                      braking_distance: float, velocity: np.ndarray) -> float:
        ref_length = self.config['dynamic_influence_distance']
        size_factor = min(1.0, math.pi * (obs_size ** 2) / (math.pi * (ref_length ** 2) + 1e-6))
        rel_velocity = obs_velocity - velocity
        rel_speed = np.linalg.norm(rel_velocity)
        
        # 增强速度敏感度：降低参考速度，使快速障碍物更容易达到高威胁
        speed_norm = min(1.0, rel_speed / 8.0)  # 从10.0降到8.0
        
        direction_factor = 0.0
        if rel_speed > 1e-6 and distance > 1e-6:
            rel_dir = rel_velocity / rel_speed
            to_robot = (current_pos - predicted_pos) / distance
            direction_factor = max(0.0, np.dot(rel_dir, to_robot))
        
        # 增强距离敏感度：使用更陡峭的衰减函数
        effective_braking = max(braking_distance * 1.5, 1.0)  # 增加有效制动距离
        distance_factor = math.exp(-distance / effective_braking)
        
        # 增加权重，更重视速度和方向因素
        w_size = self.config['threat_size_weight']
        w_speed = self.config['threat_speed_weight'] * 1.5  # 增加速度权重
        w_dir = self.config['threat_direction_weight'] * 1.2  # 增加方向权重
        w_dist = self.config['threat_distance_weight']
        total_weight = w_size + w_speed + w_dir + w_dist
        if total_weight == 0:
            return 0.0
        threat = (
            w_size * size_factor +
            w_speed * speed_norm +
            w_dir * direction_factor +
            w_dist * distance_factor
        ) / total_weight
        return float(np.clip(threat, 0.0, 1.0))

    def _enforce_motion_constraints(self, current_pos: np.ndarray, proposed_pos: np.ndarray,
                                    current_velocity: np.ndarray) -> np.ndarray:
        displacement = proposed_pos - current_pos
        disp_norm = np.linalg.norm(displacement)
        if disp_norm < 1e-6:
            return proposed_pos

        max_speed = self.config['max_linear_speed']
        if disp_norm > max_speed:
            displacement = displacement / disp_norm * max_speed

        target_velocity = displacement
        if current_velocity is None or len(current_velocity) == 0:
            current_velocity = np.zeros(2)

        # 线加速度限制
        acc_vec = target_velocity - current_velocity
        acc_norm = np.linalg.norm(acc_vec)
        max_acc = self.config['max_linear_acc']
        if acc_norm > max_acc:
            acc_vec = acc_vec / acc_norm * max_acc
            target_velocity = current_velocity + acc_vec

        speed = np.linalg.norm(target_velocity)
        if speed > max_speed:
            target_velocity = target_velocity / speed * max_speed
            speed = max_speed

        # 角速度/角加速度限制
        heading = target_velocity / (speed + 1e-9)
        heading = self._limit_heading_change(heading)
        target_velocity = heading * speed

        self.prev_velocity = target_velocity
        self.prev_heading = heading

        return current_pos + target_velocity

    def _limit_heading_change(self, desired_heading: np.ndarray) -> np.ndarray:
        prev_heading = self.prev_heading
        dot = float(np.clip(np.dot(prev_heading, desired_heading), -1.0, 1.0))
        angle_diff = math.acos(dot)
        cross = prev_heading[0] * desired_heading[1] - prev_heading[1] * desired_heading[0]
        direction = np.sign(cross) if abs(cross) > 1e-6 else 1.0

        max_ang_vel = self.config['max_angular_velocity']
        max_ang_acc = self.config['max_angular_acceleration']

        limited_angle = min(angle_diff, max_ang_vel)
        delta_angle = limited_angle - self.prev_angle_change
        if abs(delta_angle) > max_ang_acc:
            limited_angle = self.prev_angle_change + np.sign(delta_angle) * max_ang_acc
            limited_angle = np.clip(limited_angle, 0.0, max_ang_vel)

        self.prev_angle_change = limited_angle

        if limited_angle < 1e-6:
            return prev_heading

        rotation_matrix = np.array([
            [math.cos(limited_angle), -direction * math.sin(limited_angle)],
            [direction * math.sin(limited_angle),  math.cos(limited_angle)]
        ])
        new_heading = rotation_matrix.dot(prev_heading)
        norm = np.linalg.norm(new_heading)
        if norm < 1e-6:
            return prev_heading
        return new_heading / norm
    
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
