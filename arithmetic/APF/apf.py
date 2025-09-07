import math
import time
import random

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from shapely import affinity
from shapely.geometry import Point, Polygon
import pygame
from shapely.ops import nearest_points
from scipy.interpolate import splprep, splev
from arithmetic.APF.Node import point


class apf:
    def __init__(self, mapdata):
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])  # 储存此次搜索的开始点
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])  # 储存此次搜索的目的点
        # self.Map = mapdata.map  # 一个二维数组，为此次搜索的地图引用
        self.result = []  # 当计算完成后，将最终得到的路径写入到此属性中
        self.count = 0  # 记录此次搜索所搜索过的节点数
        self.width = mapdata.width
        self.height = mapdata.height
        self.obstacles = mapdata.obstacles
        self.dynamic_obstacles = mapdata.dynamic_obstacles  # 动态障碍物
        # 参数
        self.attraction_coeff = 5.0  # 吸引力系数
        self.repulsion_coeff = 1000.0  # 斥力系数
        self.repulsion_threshold = 100  # 斥力作用距离阈值
        self.obstacle = mapdata.obs_surface  # 多边形障碍物顶点
        self.position_history = []  # 用于存储历史位置的列表
        self.history_size = 100  # 检测震荡时的点是否大于这个值
        self.oscillation_detection_threshold = 3  # 震荡检测阈值
        self.stall_speed_eps = 0.1
        self.stall_window = 25

        # 分段吸引阈值（C¹ 连续）
        self.attr_switch_dist = 80.0

        # 斥力最大幅度（饱和防爆力）
        self.repulsion_max = 150.0

        # 切向引导强度（靠近障碍物时启用）
        self.tangential_gain = 0.6

        # 动量（0~1，大则更平滑）
        self.momentum = 0.85
        self.velocity = np.zeros(2, dtype=float)

        # 自适应步长 & 上下限
        self.base_step = 4.0
        self.max_step = 10.0
        self.min_step = 0.5

        # 边界斥力设置
        self.boundary_thresh = 80.0
        self.boundary_gain = 600.0

    # ---------- 工具 ----------
    @staticmethod
    def _norm(v):
        n = np.linalg.norm(v)
        return n if n > 1e-9 else 0.0

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        if n < 1e-9:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _rot90(v):
        # 逆时针旋转 90°
        return np.array([-v[1], v[0]], dtype=float)

    def distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def colregs_adjustment(self, current_point, velocity, safety_distance=5.0):
        """
        基于 COLREGs + CPA 的避碰调整
        """
        adjustment = np.zeros(2, dtype=float)
        usv_pos = np.array([current_point.x, current_point.y])
        usv_vel = velocity if np.linalg.norm(velocity) > 1e-6 else np.array([1.0, 0.0])

        for obs in self.dynamic_obstacles:
            obs_vel = np.array(obs.direction) * obs.speed
            obs_pos = np.array(obs.position)

            rel_pos = obs_pos - usv_pos      # 相对位置
            rel_vel = obs_vel - usv_vel      # 相对速度

            if np.linalg.norm(rel_vel) < 1e-6:  # 相对静止，无需避让
                continue

            # 计算 CPA (最近点距离)
            t_cpa = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_vel) ** 2)
            if t_cpa < 0:
                continue  # 最近点发生在过去，无需处理

            cpa_pos = rel_pos + t_cpa * rel_vel
            dist_cpa = np.linalg.norm(cpa_pos)

            if dist_cpa < safety_distance:  # 触发避让
                bearing = math.degrees(math.atan2(rel_pos[1], rel_pos[0]))
                heading = math.degrees(math.atan2(usv_vel[1], usv_vel[0]))
                angle_diff = (bearing - heading + 360) % 360

                # 会遇
                if 165 <= angle_diff <= 195:
                    adjustment += np.array([usv_vel[1], -usv_vel[0]])  # 向右转
                # 交叉
                elif 0 < angle_diff < 180:  # 右舷
                    adjustment += np.array([usv_vel[1], -usv_vel[0]]) * 1.5
                else:  # 左舷
                    adjustment += np.zeros(2)
                # 追越
                if 112.5 <= angle_diff <= 247.5:
                    adjustment += np.array([usv_vel[1], -usv_vel[0]]) * 1.2

        return adjustment


    # ---------- 分段连续吸引力 ----------
    def calculate_attraction(self, current_point):
        dx = self.end.x - current_point.x
        dy = self.end.y - current_point.y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return 0.0, 0.0

        # C¹ 连续：d <= d* 为二次势；d > d* 为线性势
        if d <= self.attr_switch_dist:
            # grad U = k * (q - q_goal)
            fx = self.attraction_coeff * (dx)
            fy = self.attraction_coeff * (dy)
        else:
            # grad U = k * d* * (q - q_goal)/||q - q_goal||
            k = self.attraction_coeff * self.attr_switch_dist
            ux, uy = dx / d, dy / d
            fx = k * ux
            fy = k * uy
        return fx, fy

    def distance(self, point1, point2):
        # 计算两点直接的距离
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def calculate_attraction(self, current_point):
        # 计算吸引力向量
        dx = self.end.x - current_point.x
        dy = self.end.y - current_point.y
        distance_to_goal = self.distance(current_point, self.end)
        if distance_to_goal > 0:
            force_x = dx / distance_to_goal * self.attraction_coeff
            force_y = dy / distance_to_goal * self.attraction_coeff
        else:
            force_x = force_y = 0
        return force_x, force_y

    def calculate_repulsion(self, current_point):
        # 计算斥力
        pos = Point(current_point.x, current_point.y)
        Fx = Fy = 0.0
        for obstacle in self.obstacles:
            poly = Polygon(obstacle)
            # 最近点与距离
            nearest_pt = nearest_points(pos, poly)[1]
            d = pos.distance(poly)  # 到多边形（边界）的最小距离

            if d < 1e-6:
                d = 1e-6  # 在障碍物上，防除零

            if d < self.repulsion_threshold:
                # 法向（从当前指向最近边界点）
                nx = nearest_pt.x - current_point.x
                ny = nearest_pt.y - current_point.y
                n_vec = np.array([nx, ny], dtype=float)
                n_hat = self._unit(n_vec)

                # 经典斥力梯度（C¹）+ 饱和
                # |F| = η * (1/d - 1/Q) * (1/d^2)
                eta = self.repulsion_coeff
                Q = self.repulsion_threshold
                mag = eta * (1.0 / d - 1.0 / Q) / (d ** 2)
                # 饱和到 repulsion_max，防止爆力
                mag = self.repulsion_max * np.tanh(mag / self.repulsion_max)

                Fx -= mag * n_hat[0]
                Fy -= mag * n_hat[1]

                # ---- 切向引导（绕行）----
                # 取法向旋转 90° 的切向方向，选择更靠近目标的方向
                t_hat_pos = self._unit(self._rot90(n_hat))
                t_hat_neg = -t_hat_pos

                # 选择能让“朝向目标的投影增加”的切向方向
                goal_vec = np.array([self.end.x - current_point.x,
                                     self.end.y - current_point.y], dtype=float)
                cand1 = np.dot(t_hat_pos, self._unit(goal_vec))
                cand2 = np.dot(t_hat_neg, self._unit(goal_vec))
                t_hat = t_hat_pos if cand1 >= cand2 else t_hat_neg

                # 切向权重随距离衰减（越近越强）
                w_tan = self.tangential_gain * max(0.0, (self.repulsion_threshold - d) / self.repulsion_threshold)
                Fx += w_tan * mag * t_hat[0]
                Fy += w_tan * mag * t_hat[1]

        return Fx, Fy

        # ---------- 边界斥力（四围墙） ----------

    def calculate_boundary_repulsion(self, current_point):
        x, y = current_point.x, current_point.y
        Fx = Fy = 0.0
        # 到各墙的距离
        d_left = max(1e-6, x)
        d_right = max(1e-6, self.width - x)
        d_bottom = max(1e-6, y)
        d_top = max(1e-6, self.height - y)

        def wall_force(d, nx, ny):
            if d < self.boundary_thresh:
                mag = self.boundary_gain * (1.0 / d - 1.0 / self.boundary_thresh) / (d ** 2)
                mag = self.repulsion_max * np.tanh(mag / self.repulsion_max)
                return -mag * nx, -mag * ny
            return 0.0, 0.0

        # 法向外指向墙面（我们需要把机器人推离墙）
        fx, fy = wall_force(d_left, -1.0, 0.0)  # 左墙法向 (-1,0)，推向 +x
        Fx += fx;
        Fy += fy
        fx, fy = wall_force(d_right, 1.0, 0.0)  # 右墙法向 (1,0)，推向 -x
        Fx += fx;
        Fy += fy
        fx, fy = wall_force(d_bottom, 0.0, -1.0)  # 下墙法向 (0,-1)，推向 +y
        Fx += fx;
        Fy += fy
        fx, fy = wall_force(d_top, 0.0, 1.0)  # 上墙法向 (0,1)，推向 -y
        Fx += fx;
        Fy += fy

        return Fx, Fy

    def add_position_to_history(self, current_point):
        """记录当前位置到历史位置列表"""
        if len(self.position_history) >= self.history_size:
            self.position_history.pop(0)  # 移除最旧的记录
        self.position_history.append((current_point.x, current_point.y))

    def detect_oscillation(self):
        if len(self.position_history) < self.history_size:
            return False
        # 连续小幅往返：相邻位移都很小
        small_moves = 0
        for i in range(1, len(self.position_history)):
            x1, y1 = self.position_history[-i]
            x0, y0 = self.position_history[-i - 1]
            if math.hypot(x1 - x0, y1 - y0) < self.oscillation_detection_threshold:
                small_moves += 1
            else:
                break
        return small_moves > (0.8 * self.history_size)

    def escape_from_oscillation(self):
        """实施逃逸策略"""
        # 逃逸策略，可以是改变系数、随机移动等
        # 增加引力系数，减少斥力系数
        self.attraction_coeff *= 1.2
        self.repulsion_coeff *= 0.8

        # 随机选择一个方向移动一段距离
        random_angle = random.uniform(0, 2 * math.pi)
        escape_distance = 10  # 可以根据实际情况调整
        escape_x = math.cos(random_angle) * escape_distance
        escape_y = math.sin(random_angle) * escape_distance

        return escape_x, escape_y

    def smooth_path(self, path_points):
        """
        Smooth the given path points using B-spline.

        Parameters:
        path_points (list of tuples): A list of (x, y) points defining the path.

        Returns:
        list of tuples: The smoothed path points.
        """
        if len(path_points) < 3:
            return path_points

        x = [p[0] for p in path_points]
        y = [p[1] for p in path_points]

        tck, u = splprep([x, y], s=0.0, per=False)
        u_new = np.linspace(u.min(), u.max(), len(path_points))
        x_new, y_new = splev(u_new, tck, der=0)

        smoothed_path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
        return smoothed_path

    def get_dynamic_obstacle_distance(self, current_point, t_future=1.0):
        """
        计算当前点到最近动态障碍物（预测未来位置）的距离
        """
        pos = Point(current_point.x, current_point.y)
        min_dist = float('inf')

        for obs in self.dynamic_obstacles:
            future_center = obs.predict_future_position(t_future)
            poly = Polygon(obs.to_polygon())  # 基于当前位置的多边形
            poly_shifted = affinity.translate(poly,
                                              xoff=future_center.x - obs.position[0],
                                              yoff=future_center.y - obs.position[1])
            dist = pos.distance(poly_shifted)
            min_dist = min(min_dist, dist)

        return min_dist

    def remove_oscillations(self, path_points, angle_threshold=0.1):
        """
        Remove oscillations from the path by detecting and removing points with high angle changes.

        Parameters:
        path_points (list of tuples): A list of (x, y) points defining the path.
        angle_threshold (float): The threshold for angle change to detect oscillations.

        Returns:
        list of tuples: The path points with oscillations removed.
        """
        if len(path_points) < 3:
            return path_points

        def angle_between_points(p1, p2, p3):
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
            return np.abs(angle)

        filtered_path = [path_points[0]]
        for i in range(1, len(path_points) - 1):
            p1 = path_points[i - 1]
            p2 = path_points[i]
            p3 = path_points[i + 1]

            angle = angle_between_points(p1, p2, p3)
            if angle < angle_threshold:
                filtered_path.append(p2)

        filtered_path.append(path_points[-1])
        return filtered_path

    def detect_potential_collision(self, point, velocity, safe_dist=5.0):
        """
        判断当前位置在未来一段时间内是否可能发生碰撞
        """
        for obs in self.dynamic_obstacles:
            future_pos = obs.predict_future_position(5.0)
            dist = np.linalg.norm(np.array([future_pos.x - point.x,
                                            future_pos.y - point.y]))
            if dist < safe_dist:  # 距离过近，判定有风险
                return True
        return False

    def plan(self, plan_surface):
        # 寻找路径的方法（已改进：用 shapely 计算最近障碍距离，避免之前的类型错误）
        current_point = self.start
        self.result = []
        self.velocity = np.zeros(2, dtype=float)
        self.position_history = []
        self.count = 0

        start_time = time.time()
        while self.distance(current_point, self.end) > 1.0:
            # 吸引 / 斥力 / 边界
            ax, ay = self.calculate_attraction(current_point)
            rx, ry = self.calculate_repulsion(current_point)
            bx, by = self.calculate_boundary_repulsion(current_point)

            Fx = ax + rx + bx
            Fy = ay + ry + by
            F = np.array([Fx, Fy], dtype=float)
            if self.detect_potential_collision(current_point, self.velocity):
                colregs_F = self.colregs_adjustment(
                    current_point,
                    self.velocity,
                    safety_distance=getattr(self, "colregs_horizon", 5.0)
                )
                # --- 方式1：加权融合，而不是直接叠加 ---
                F = 0.8 * F + 0.2 * colregs_F
            #F += self.colregs_adjustment(current_point, self.velocity, t_future=getattr(self, "colregs_horizon", 5.0))
            F_norm = self._norm(F)

            # 目标方向
            goal_vec = np.array([self.end.x - current_point.x, self.end.y - current_point.y], dtype=float)
            goal_dist = np.linalg.norm(goal_vec)
            goal_dir = self._unit(goal_vec)

            # ------- 正确计算当前点到最近障碍的距离（使用 shapely） -------
            # ------- 计算到静态/动态障碍物的最近距离 -------
            if self.obstacles:
                pos = Point(current_point.x, current_point.y)
                nearest_static_dist = min(pos.distance(Polygon(obs)) for obs in self.obstacles)
            else:
                nearest_static_dist = float('inf')

                # 动态障碍物距离：未来位置
            lookahead_t = getattr(self, "dyn_predict_dt", 0.3)  # 预测 1 秒，可调 0.5~3.0
            nearest_dynamic_dist = self.get_dynamic_obstacle_distance(current_point, t_future=lookahead_t)
            nearest_obs_dist = min(nearest_static_dist, nearest_dynamic_dist)

            # SAFE_RADIUS：基于斥力影响半径（repulsion_threshold）
            SAFE_RADIUS = 1.5 * self.repulsion_threshold

            # 根据最近障碍距离选择模式：直推（无障碍）或避障（合力+动量）
            if nearest_obs_dist > SAFE_RADIUS:
                # === 无障碍模式：直推目标方向（清除横向分量） ===
                step = np.clip(
                    self.base_step * (0.7 + 0.3 * (goal_dist / max(1.0, self.attr_switch_dist))),
                    self.min_step, self.max_step
                )
                desired_velocity = goal_dir * step

                v_prev = self.velocity.copy()
                proj_along = np.dot(v_prev, goal_dir) * goal_dir
                blend_along = 0.3
                self.velocity = blend_along * proj_along + (1.0 - blend_along) * desired_velocity

            else:
                # === 避障模式：合力 + 动量过滤 ===
                if F_norm < 1e-9:
                    step = self.min_step
                    direction = np.zeros(2)
                else:
                    direction = F / F_norm
                    step = np.clip(
                        self.base_step * (0.5 + 0.5 * np.tanh(F_norm / 20.0)),
                        self.min_step, self.max_step
                    )

                self.velocity = self.momentum * self.velocity + (1.0 - self.momentum) * (direction * step)

                if np.linalg.norm(self.velocity) < self.stall_speed_eps:
                    jitter = 0.8 * self.max_step * self._unit(
                        np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
                    )
                    self.velocity += jitter

            # 接近目标：投影到目标方向并减速，避免冲过头
            if goal_dist < max(5.0, 0.2 * self.attr_switch_dist):
                self.velocity = np.dot(self.velocity, goal_dir) * goal_dir
                self.velocity *= 0.5

            # 再次卡死检测
            if np.linalg.norm(self.velocity) < self.stall_speed_eps:
                jitter = 0.8 * self.max_step * self._unit(
                    np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
                )
                self.velocity += jitter

            # 更新位置
            next_x = current_point.x + float(self.velocity[0])
            next_y = current_point.y + float(self.velocity[1])
            next_point = point(next_x, next_y)

            # 记录与振荡检测
            self.add_position_to_history(current_point)
            if self.detect_oscillation():
                self.tangential_gain = min(1.2, self.tangential_gain + 0.15)
                self.velocity += 0.3 * self.max_step * self._unit(
                    np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
                )

            # 到达判定
            if self.distance(next_point, self.end) < 10.0:
                self.result.append((self.end.x, self.end.y))
                break

            # 绘制与记录
            if plan_surface is not None:
                pygame.draw.circle(plan_surface, (0, 100, 255), (int(next_point.x), int(next_point.y)), 2)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()

            self.result.append((next_point.x, next_point.y))
            current_point = next_point
            self.count += 1

            if self.count > 10000:
                print("没有可行路径或陷入局部极小值点。")
                break

        end_time = time.time()
        filtered_path = self.remove_oscillations(self.result)
        smoothed_path = self.smooth_path(filtered_path)
        print("花费时间为", end_time - start_time)
        return smoothed_path, end_time - start_time
