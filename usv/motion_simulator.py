import numpy as np
import pygame
from PyQt5.QtCore import QTimer
from shapely.geometry import Polygon, Point
from local_planner.local_planner_manager import LocalPlannerManager
from local_planner.algorithms.apf_local_planner import APFLocalPlanner
from local_planner.algorithms.apf_local_planner_simple import APFLocalPlannerSimple
from local_planner.algorithms.dwa_local_planner import DWALocalPlanner

class USVMotionSimulator:
    """无人船实时运动模拟器"""
    def __init__(self, pygame_widget):
        self.widget = pygame_widget
        self.current_pos = None
        self.target_pos = None
        self.velocity = np.zeros(2)
        self.path = []
        self.path_index = 0
        self.is_moving = False

        # 船舶参数
        self.max_speed = 5.0  # 最大速度（提高到5.0以超过障碍物速度）
        self.turn_rate = 0.1  # 转向率
        self.ship_radius = 5  # 船舶半径 - 从8减小到5以降低碰撞检测敏感度
        self.arrival_threshold = 2.0  # 到达目标点的阈值，设置为较小的值使船舶更接近目标点
        
        # 加速度约束参数
        self.max_linear_acceleration = 1.0  # 最大线性加速度（提高到1.0以更快达到高速度）
        self.max_angular_acceleration = 0.1  # 最大转向加速度（弧度/帧）
        self.old_velocity = np.zeros(2)  # 上一帧的速度，用于计算加速度

        # 碰撞状态标记
        self.has_collided = False
        self.collision_position = None
        
        # 历史轨迹记录
        self.history_path = []  # 记录完整的航行轨迹
        self.journey_completed = False  # 航行完成标记

        # 局部路径规划管理器
        self.local_planner_manager = LocalPlannerManager()
        
        # 注册可用的局部算法
        self.local_planner_manager.register_algorithm("TAPF", APFLocalPlanner)  # 威胁指数改进版
        self.local_planner_manager.register_algorithm("DWA算法", DWALocalPlanner)
        self.local_planner_manager.register_algorithm("原APF", APFLocalPlannerSimple)  # 原始简化版
        self.local_planner_manager.register_algorithm("动态窗口法", DWALocalPlanner)
        
        # 设置默认算法
        self.local_algorithm = "无局部算法"
        self.local_planner_manager.set_current_algorithm("无局部算法")

        # 实时局部路径
        self.local_path = []  # 存储实时计算的局部路径
        self.show_local_path = True  # 是否显示局部路径

        # 动画定时器
        self.motion_timer = QTimer()
        self.motion_timer.timeout.connect(self.update_position)
        # 延迟启动定时器，避免初始化冲突
        QTimer.singleShot(100, lambda: self.motion_timer.start(20))  # 50FPS，更流畅

    def start_journey(self, start_pos, target_pos, planned_path):
        """开始航行"""
        # 重置状态
        self.has_collided = False
        self.collision_position = None
        self.journey_completed = False
        self.history_path = []  # 清空历史轨迹
        
        self.current_pos = np.array(start_pos, dtype=float)
        self.target_pos = target_pos
        self.path = planned_path
        self.path_index = 0
        self.is_moving = True
        print(f"开始航行！起点: {start_pos}, 终点: {target_pos}, 路径点数: {len(planned_path)}")
        print(f"使用局部算法: {self.local_algorithm}")
        print(f"定时器状态: {self.motion_timer.isActive()}")
        if not self.motion_timer.isActive():
            self.motion_timer.start(50)
            print("定时器已启动")

    def set_local_algorithm(self, algorithm_name):
        """设置局部路径规划算法"""
        self.local_algorithm = algorithm_name
        self.local_planner_manager.set_current_algorithm(algorithm_name)
        print(f"已设置局部路径规划算法为: {algorithm_name}")

    def apply_local_algorithm(self, current_pos, target_pos, obstacles):
        """应用局部路径规划算法并生成完整路径"""
        # 直接使用原始障碍物对象，无需格式转换
        static_obstacles = []  # 静态障碍物目前为空
        dynamic_obstacles = obstacles  # 动态障碍物直接使用原始对象
        
        # 使用局部路径规划管理器获取单个目标点
        local_target = self.local_planner_manager.plan_local_path(
            np.array(current_pos),
            np.array(target_pos),
            static_obstacles,
            dynamic_obstacles,
            self.velocity
        )
        
        # 生成完整的局部路径（使用多个中间点）
        if self.show_local_path and local_target is not None:
            # 生成从当前位置到局部目标的平滑路径
            self.local_path = self._generate_smooth_local_path(
                current_pos, local_target, dynamic_obstacles
            )
        else:
            self.local_path = []
            
        return local_target
    
    def _generate_smooth_local_path(self, start_pos, end_pos, dynamic_obstacles, num_points=25):
        """修复后的智能多障碍物协同避障路径规划"""
        start = np.array(start_pos)
        end = np.array(end_pos)
        
        # 所有参数完全在函数内部定义
        # 基础参数设置 - 降低敏感度
        SAFE_DISTANCE_STATIC = 15  # 从35减小到15
        SAFE_DISTANCE_DYNAMIC = 12  # 从30减小到12
        SAFE_DISTANCE_BOUNDARY = 10  # 从25减小到10
        
        # 计算距离，处理可能的除零错误
        total_distance = max(np.linalg.norm(end - start), 1.0)  # 防止除零
        
        # 根据距离调整参数（简化版本，避免复杂条件）
        if total_distance > 200:
            K_att = 0.6
            K_rep_static = 2.8
            K_rep_dynamic = 2.2
            K_rep_boundary = 1.8
            SAFE_DISTANCE_STATIC = 20  # 从40减小到20
            SAFE_DISTANCE_DYNAMIC = 18  # 从35减小到18
            SAFE_DISTANCE_BOUNDARY = 15  # 从30减小到15
        elif total_distance < 100:
            K_att = 1.0
            K_rep_static = 2.2
            K_rep_dynamic = 1.8
            K_rep_boundary = 1.3
            SAFE_DISTANCE_STATIC = 12  # 从30减小到12
            SAFE_DISTANCE_DYNAMIC = 10  # 从25减小到10
            SAFE_DISTANCE_BOUNDARY = 8  # 从20减小到8
        else:
            K_att = 0.8
            K_rep_static = 2.5
            K_rep_dynamic = 2.0
            K_rep_boundary = 1.5
            SAFE_DISTANCE_STATIC = 15  # 从35减小到15
            SAFE_DISTANCE_DYNAMIC = 12  # 从30减小到12
            SAFE_DISTANCE_BOUNDARY = 10  # 从25减小到10
        
        # 最大影响范围
        MAX_INFLUENCE_STATIC = 85
        MAX_INFLUENCE_DYNAMIC = 75
        MAX_INFLUENCE_BOUNDARY = 55
        
        # 生成路径
        path = [start.tolist()]
        
        # 用于跟踪附近障碍物数量
        nearby_obstacles_count = 0
        
        for i in range(1, num_points):
            t = i / num_points
            
            # 当前路径点
            current_point = start + t * (end - start)
            
            # 计算总势场力
            total_force = np.zeros(2)
            
            # 1. 目标吸引力（简化计算，避免除零）
            distance_to_goal = max(np.linalg.norm(end - current_point), 1.0)
            attraction_strength = K_att * (distance_to_goal / total_distance)
            attraction_force = K_att * (end - current_point) * min(attraction_strength, 2.0)
            total_force += attraction_force
            
            # 2. 静态障碍物斥力
            if hasattr(self.widget, 'obstacles'):
                nearby_obstacles_count = 0
                for static_obstacle in self.widget.obstacles:
                    try:
                        static_polygon = Polygon(static_obstacle)
                        point = Point(current_point[0], current_point[1])
                        distance = max(static_polygon.distance(point), 0.1)  # 防止除零
                        
                        if distance < MAX_INFLUENCE_STATIC:
                            nearby_obstacles_count += 1
                            
                            # 找到最近点
                            nearest_point = static_polygon.exterior.interpolate(
                                static_polygon.exterior.project(point)
                            )
                            nearest_coords = np.array([nearest_point.x, nearest_point.y])
                            
                            # 计算避障方向
                            to_obstacle = current_point - nearest_coords
                            repulsion_dir = to_obstacle / np.linalg.norm(to_obstacle)
                            
                            # 计算斥力强度（避免除零）
                            if distance < SAFE_DISTANCE_STATIC:
                                repulsion_strength = K_rep_static * (1.0/max(distance, 0.1) - 1.0/SAFE_DISTANCE_STATIC)
                            else:
                                repulsion_strength = K_rep_static * (1.0 - distance/MAX_INFLUENCE_STATIC)
                            
                            total_force += repulsion_dir * repulsion_strength * 20
                    except Exception:
                        # 处理几何计算异常
                        continue
            
            # 3. 动态障碍物斥力（简化预测）
            for obstacle in dynamic_obstacles:
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'size'):
                    try:
                        obs_pos = np.array(obstacle.position)
                        obs_velocity = np.array(obstacle.direction) * obstacle.speed
                        
                        # 简化的预测：只考虑当前位置和预测0.5秒后的位置
                        predicted_pos = obs_pos + obs_velocity * 0.5
                        distance = max(np.linalg.norm(current_point - predicted_pos), 0.1)
                        
                        if distance < MAX_INFLUENCE_DYNAMIC:
                            to_obstacle = current_point - predicted_pos
                            repulsion_dir = to_obstacle / np.linalg.norm(to_obstacle)
                            
                            if distance < SAFE_DISTANCE_DYNAMIC:
                                repulsion_strength = K_rep_dynamic * (1.0/max(distance, 0.1) - 1.0/SAFE_DISTANCE_DYNAMIC)
                            else:
                                repulsion_strength = K_rep_dynamic * (1.0 - distance/MAX_INFLUENCE_DYNAMIC)
                            
                            total_force += repulsion_dir * repulsion_strength * 15
                    except Exception:
                        continue
            
            # 4. 边界斥力（简化实现）
            boundaries = [
                (0, current_point[1], 0),           # 左边界
                (self.widget.width, current_point[1], 0),  # 右边界
                (current_point[0], 0, 1),           # 上边界
                (current_point[0], self.widget.height, 1)   # 下边界
            ]
            
            for boundary_val, current_val, axis in boundaries:
                if axis == 0:  # x轴边界
                    distance = max(abs(current_point[0] - boundary_val), 0.1)
                    repulsion_dir = np.array([1, 0]) if boundary_val == 0 else np.array([-1, 0])
                else:  # y轴边界
                    distance = max(abs(current_point[1] - boundary_val), 0.1)
                    repulsion_dir = np.array([0, 1]) if boundary_val == 0 else np.array([0, -1])
                
                if distance < MAX_INFLUENCE_BOUNDARY:
                    if distance < SAFE_DISTANCE_BOUNDARY:
                        repulsion_strength = K_rep_boundary * (1.0/max(distance, 0.1) - 1.0/SAFE_DISTANCE_BOUNDARY)
                    else:
                        repulsion_strength = K_rep_boundary * (1.0 - distance/MAX_INFLUENCE_BOUNDARY)
                    
                    total_force += repulsion_dir * repulsion_strength * 10
            
            # 5. 限制合力大小（防止过度震荡）
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > 50:
                total_force = (total_force / force_magnitude) * 50
            elif force_magnitude > 0:
                # 添加阻尼
                total_force *= 0.8
            
            # 6. 计算新位置
            step_factor = 0.7 if nearby_obstacles_count > 1 else 0.9
            new_point = current_point + total_force * step_factor
            
            # 7. 边界保护（确保在合理范围内）
            margin = 15
            new_point[0] = max(margin, min(self.widget.width - margin, new_point[0]))
            new_point[1] = max(margin, min(self.widget.height - margin, new_point[1]))
            
            path.append(new_point.tolist())
        
        # 简化路径平滑
        if len(path) > 2:
            # 简单的两点平滑
            smoothed_path = [path[0]]
            for i in range(1, len(path)):
                if i % 2 == 0 or i == len(path) - 1:  # 保留关键点
                    smoothed_path.append(path[i])
            if len(smoothed_path) < 3:
                smoothed_path = path
            path = smoothed_path
        
        path.append(end.tolist())
        return path


    def check_collision_with_dynamic_obstacles(self, ship_pos):
        """检测船舶与动态障碍物的碰撞"""
        ship_radius = self.ship_radius
        
        if hasattr(self.widget, 'dynamic_obstacles'):
            for obstacle in self.widget.dynamic_obstacles:
                # 计算船舶中心到障碍物中心的距离
                obs_x, obs_y = obstacle.position
                distance = np.sqrt((ship_pos[0] - obs_x)**2 + (ship_pos[1] - obs_y)**2)
                
                # 碰撞条件：距离小于船舶半径 + 障碍物半径
                collision_distance = ship_radius + obstacle.size
                
                if distance < collision_distance:
                    # 发生碰撞
                    collision_info = {
                        'ship_position': tuple(ship_pos),
                        'obstacle_position': obstacle.position,
                        'obstacle_shape': obstacle.shape,
                        'obstacle_size': obstacle.size,
                        'obstacle_direction': obstacle.direction,
                        'obstacle_speed': obstacle.speed,
                        'collision_distance': distance,
                        'required_clearance': collision_distance,
                        'obstacle_type': 'dynamic'
                    }
                    return True, collision_info
        
        return False, None

    def check_collision_with_static_obstacles(self, ship_pos):
        """检测船舶与静态障碍物的碰撞"""
        ship_radius = self.ship_radius
        
        if hasattr(self.widget, 'obstacles'):
            for static_obstacle in self.widget.obstacles:
                # 将静态障碍物转换为多边形
                static_polygon = Polygon(static_obstacle)
                
                # 创建船舶的圆形区域
                ship_circle = Point(ship_pos[0], ship_pos[1]).buffer(ship_radius)
                
                # 检查船舶与静态障碍物是否相交
                if ship_circle.intersects(static_polygon):
                    # 计算最近距离
                    distance = static_polygon.distance(Point(ship_pos[0], ship_pos[1]))
                    
                    collision_info = {
                        'ship_position': tuple(ship_pos),
                        'obstacle_type': 'static',
                        'obstacle_polygon': static_obstacle,
                        'collision_distance': distance,
                        'required_clearance': ship_radius
                    }
                    return True, collision_info
        
        return False, None

    def check_all_obstacles_collision(self, ship_pos):
        """检测船舶与所有类型障碍物的碰撞"""
        # 先检查动态障碍物
        collision, info = self.check_collision_with_dynamic_obstacles(ship_pos)
        if collision:
            return True, info
            
        # 再检查静态障碍物
        collision, info = self.check_collision_with_static_obstacles(ship_pos)
        if collision:
            return True, info
            
        return False, None

    def update_position(self):
        """实时更新船舶位置 - 完全依赖APFLocalPlanner"""
        if not self.is_moving or self.path_index >= len(self.path):
            # 检查是否航行完成
            if self.path_index >= len(self.path) and len(self.path) > 0 and not self.journey_completed:
                # 立即重置状态
                self.journey_completed = True
                self.is_moving = False
                # 到达目标点后，获取当前使用的局部规划器并立即重置参数
                if hasattr(self, 'local_planner_manager') and self.local_planner_manager.get_current_algorithm() != "无局部算法":
                    algorithm_name = self.local_planner_manager.get_current_algorithm()
                    planner = self.local_planner_manager.get_algorithm(algorithm_name)
                    if planner and hasattr(planner, 'reset_planner'):
                        planner.reset_planner()
                        print(f"🚩 到达目标点，{algorithm_name}规划器参数已立即重置")
            return

        # 记录当前位置到历史轨迹
        if len(self.history_path) == 0 or np.linalg.norm(np.array(self.current_pos) - np.array(self.history_path[-1])) > 1:
            self.history_path.append(self.current_pos.copy())

        target_point = self.path[self.path_index]
        
        # 获取障碍物信息
        static_obstacles = []
        dynamic_obstacles = []
        
        if hasattr(self.widget, 'dynamic_obstacles'):
            dynamic_obstacles = self.widget.dynamic_obstacles
        
        if hasattr(self.widget, 'obstacles'):
            # 将静态障碍物转换为多边形格式
            for obs in self.widget.obstacles:
                static_obstacles.append(obs)
        
        # 碰撞检测
        collision_detected, collision_info = self.check_all_obstacles_collision(self.current_pos)
        if collision_detected:
            print("🚨 碰撞检测！")
            self.is_moving = False
            self.has_collided = True
            self.collision_position = self.current_pos.copy()
            if hasattr(self, 'motion_timer'):
                self.motion_timer.stop()
            self.widget.update()
            return
        
        # 传递全局路径给局部规划器，以便实现前视距离动态选择局部目标点
        # 只传递当前目标点之后的路径段，减少计算量
        remaining_path = self.path[self.path_index:]
        
        # 使用APFLocalPlanner计算下一步位置（实时避障）
        apf_next_pos = self.local_planner_manager.plan_local_path(
            np.array(self.current_pos),
            np.array(target_point),
            static_obstacles,
            dynamic_obstacles,
            self.velocity,
            remaining_path  # 传递剩余的全局路径
        )
        
        # 添加平滑过渡：使用当前位置和APF计算的位置之间的加权平均
        # 平滑因子控制平滑程度（值越大越平滑，但响应越慢）
        smooth_factor = 0.2  # 平滑因子，0-1之间，增大值可以获得更平滑的运动
        
        # 计算平滑后的位置
        smoothed_pos = np.array(self.current_pos) * (1 - smooth_factor) + np.array(apf_next_pos) * smooth_factor
        
        # 更新位置
        self.current_pos = smoothed_pos
        
        # 计算到当前路径点的距离
        distance_to_target = np.linalg.norm(np.array(target_point) - self.current_pos)
        
        # 检查是否到达当前路径点
        if distance_to_target < self.arrival_threshold:
            self.path_index += 1
            if self.path_index >= len(self.path):
                # 到达最终目标点
                self.current_pos = np.array(self.path[-1])
                self.history_path.append(self.current_pos.copy())
                self.is_moving = False
                self.journey_completed = True
                self.velocity = np.zeros(2)
                self.local_path = []
                if hasattr(self.motion_timer, 'stop') and self.motion_timer.isActive():
                    self.motion_timer.stop()
                if hasattr(self.widget, 'handle_journey_completed_for_snapshot'):
                    self.widget.handle_journey_completed_for_snapshot()
                return
        
        # 更新速度向量（用于下次计算），添加转向加速度约束
        if len(self.path) > self.path_index:
            next_target = self.path[self.path_index]
            direction = np.array(next_target) - self.current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # 计算目标方向单位向量
                target_direction = direction / distance
                
                # 计算目标速度（不根据距离减速）
                target_speed = self.max_speed  # 始终使用最大速度，不减速
                target_velocity = target_direction * target_speed
                
                # 获取当前帧时间间隔（假设每帧约20ms）
                dt = 0.02  # 20ms = 0.02秒
                
                # 1. 应用动量平滑
                momentum = 0.7  # 动量因子，值越大，运动越平滑但响应越慢
                self.velocity = self.velocity * momentum + target_velocity * (1 - momentum)
                
                # 2. 计算角速度和转向加速度约束
                if np.linalg.norm(self.velocity) > 0 and np.linalg.norm(target_velocity) > 0:
                    # 计算当前方向和目标方向的角度
                    current_angle = np.arctan2(self.velocity[1], self.velocity[0])
                    target_angle = np.arctan2(target_velocity[1], target_velocity[0])
                    
                    # 计算角度差（确保在[-π, π]范围内）
                    angle_diff = np.arctan2(np.sin(target_angle - current_angle), 
                                          np.cos(target_angle - current_angle))
                    
                    # 减小最大转向角度变化，使转向更平滑
                    max_angle_change = 0.05 * dt  # 减小转向加速度
                    
                    # 限制角度变化
                    if abs(angle_diff) > max_angle_change:
                        # 只允许最大角度变化
                        limited_angle = current_angle + np.sign(angle_diff) * max_angle_change
                        # 重新计算速度向量
                        velocity_magnitude = np.linalg.norm(self.velocity)
                        self.velocity = np.array([
                            np.cos(limited_angle) * velocity_magnitude,
                            np.sin(limited_angle) * velocity_magnitude
                        ])
                
                # 3. 确保速度不超过最大速度
                velocity_magnitude = np.linalg.norm(self.velocity)
                if velocity_magnitude > self.max_speed:
                    self.velocity = self.velocity * (self.max_speed / velocity_magnitude)
            else:
                self.velocity = np.zeros(2)
            
            # 保存当前速度用于下次计算
            self.old_velocity = self.velocity.copy()
        
        # 触发重绘
        self.widget.update()

    def draw_ship(self, surface):
        """绘制船舶"""
        if self.current_pos is None:
            return

        # 绘制全局规划的完整路径（无论航行是否完成都显示）
        if len(self.path) >= 2:
            # 使用浅蓝色点线绘制全局路径
            pygame.draw.lines(surface, (173, 216, 230), False, self.path, 2)
            
            # 绘制全局路径的起点（蓝色圆圈）
            start_x, start_y = self.path[0]
            pygame.draw.circle(surface, (0, 0, 255), (int(start_x), int(start_y)), 6)
            
            # 绘制全局路径的终点（红色圆圈）
            end_x, end_y = self.path[-1]
            pygame.draw.circle(surface, (255, 0, 0), (int(end_x), int(end_y)), 6)
            
            # 绘制中间路径点
            for i, point in enumerate(self.path[1:-1]):
                pygame.draw.circle(surface, (173, 216, 230), (int(point[0]), int(point[1])), 3)

        # 绘制航行完成后的完整历史轨迹
        if self.journey_completed and len(self.history_path) >= 2:
            # 绘制完整的历史轨迹（绿色粗线）
            pygame.draw.lines(surface, (0, 255, 0), False, self.history_path, 4)
            
            # 添加轨迹信息文字
            font = pygame.font.Font(None, 20)
            info_text = f"航行完成! 轨迹点数: {len(self.history_path)}"
            text_surface = font.render(info_text, True, (0, 255, 0))
            surface.blit(text_surface, (10, self.widget.height - 30))

        # 检查是否发生碰撞
        if self.has_collided and self.collision_position is not None:
            # 碰撞状态：绘制红色船舶并添加标记
            angle = np.arctan2(self.velocity[1], self.velocity[0])
            points = self.get_ship_shape(self.collision_position, angle)
            pygame.draw.polygon(surface, (255, 0, 0), points)  # 红色表示碰撞
            
            # 绘制碰撞标记
            collision_x, collision_y = self.collision_position
            pygame.draw.circle(surface, (255, 255, 0), (int(collision_x), int(collision_y)), 15, 3)  # 黄色警告圈
            pygame.draw.circle(surface, (255, 0, 0), (int(collision_x), int(collision_y)), 5)  # 红色中心点
            
            # 绘制碰撞信息
            font = pygame.font.Font(None, 24)
            text = font.render("COLLISION!", True, (255, 0, 0))
            surface.blit(text, (int(collision_x) + 20, int(collision_y) - 30))
            
        elif not self.journey_completed:
            # 正常航行状态：绘制青色船舶
            angle = np.arctan2(self.velocity[1], self.velocity[0])
            points = self.get_ship_shape(self.current_pos, angle)
            pygame.draw.polygon(surface, (0, 255, 255), points)

        # 绘制实时航迹（浅蓝色）
        if not self.journey_completed and len(self.history_path) >= 2:
            pygame.draw.lines(surface, (135, 206, 250), False, self.history_path, 2)

        # 绘制实时局部路径
        if not self.journey_completed and self.show_local_path and len(self.local_path) >= 2:
            # 绘制局部路径规划器生成的实时路径（红色虚线）
            if len(self.local_path) >= 3:
                # 使用更粗的线条和更亮的颜色
                pygame.draw.lines(surface, (255, 50, 50), False, self.local_path, 3)
                
                # 绘制路径点
                for i, point in enumerate(self.local_path):
                    if i == 0:
                        # 起点（当前船舶位置）- 黄色
                        pygame.draw.circle(surface, (255, 255, 0), (int(point[0]), int(point[1])), 4)
                    elif i == len(self.local_path) - 1:
                        # 终点（局部目标点）- 红色
                        pygame.draw.circle(surface, (255, 0, 0), (int(point[0]), int(point[1])), 5)
                    else:
                        # 中间点 - 橙色
                        pygame.draw.circle(surface, (255, 165, 0), (int(point[0]), int(point[1])), 2)
            else:
                # 简单直线路径
                pygame.draw.lines(surface, (255, 100, 100), False, self.local_path, 2)

    def get_ship_shape(self, pos, angle):
        """获取船舶形状点"""
        x, y = pos
        size = self.ship_radius
        return [
            (x + size * np.cos(angle), y + size * np.sin(angle)),
            (x + size * np.cos(angle + 2.5), y + size * np.sin(angle + 2.5)),
            (x + size * np.cos(angle - 2.5), y + size * np.sin(angle - 2.5))
        ]