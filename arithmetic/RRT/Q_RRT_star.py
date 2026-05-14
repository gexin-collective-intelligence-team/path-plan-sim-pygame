import math
import random
import time
import numpy as np
import pygame
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PyQt5.QtWidgets import QApplication

from .Node import point


# ====================== 路径规划类 ======================
class Q_RRT_star:

    def __init__(self, start, goal=None, obstacle_list=None, rand_area=None, params=None):
        self.mapdata = None
        self.surface_obstacle = None
        self.obstacle_distance_map = None
        self.surface_obstacle_density = None
        self.width = None
        self.height = None

        # 项目内调用方式：Q_RRT_star(mapdata).plan(plan_surface)
        if goal is None and hasattr(start, "start_point") and hasattr(start, "end_point"):
            self.mapdata = start
            if not self.mapdata.start_point or not self.mapdata.end_point:
                raise ValueError("Q_RRT* 需要先设置起点和终点")
            goal = self.mapdata.end_point
            start = self.mapdata.start_point
            self.surface_obstacle = getattr(self.mapdata, "obs_surface", None)
            self.width = getattr(self.mapdata, "width", None)
            self.height = getattr(self.mapdata, "height", None)
            self._init_surface_metrics()
            raw_obstacles = getattr(self.mapdata, "obstacles", [])
            obstacle_list = obstacle_list if obstacle_list is not None else self._obstacles_to_circles(
                raw_obstacles
            )
            rand_area = rand_area if rand_area is not None else (0, self.width, 0, self.height)
            project_defaults = {
                "robot_radius": 2.0,
                "safety_margin": 2.0,
                "formation_radius": 0.0,
                "maxIterAuto": 2200,
                "maxSampleAttempts": 80,
                "patience": 90,
            }
            params = {**project_defaults, **(params or {})}

        self.start = start
        self.goal = goal
        self.obstacle_list = obstacle_list if obstacle_list is not None else []
        self.params = params if params is not None else {}
        self.node_list = None
        self.start_to_goal_distance = None
        if rand_area is None:
            rand_area = self._infer_map_bounds(
                start=self.start,
                goal=self.goal,
                obstacle_list=self.obstacle_list
            )

        # ==============================
        # 自适应邻域半径相关参数（RRT*）# 自动尺度配置：让算法不依赖固定地图大小和固定步长
        # ==============================
        self.dim = 2

        # 1. 自动解析地图范围
        self._parse_map_bounds(rand_area)

        # 2. 计算地图尺度
        self.map_width = self.x_max - self.x_min
        self.map_height = self.y_max - self.y_min
        self.map_area = max(1e-6, self.map_width * self.map_height)
        self.map_diag = math.hypot(self.map_width, self.map_height)

        # 3. 起终点距离
        sx, sy = self.start
        gx, gy = self.goal
        self.start_goal_dist = max(1e-6, math.hypot(gx - sx, gy - sy))

        # 4. 障碍物尺度
        if len(self.obstacle_list) > 0:
            self.mean_obs_radius = sum([obs[2] for obs in self.obstacle_list]) / len(self.obstacle_list)
            self.max_obs_radius = max([obs[2] for obs in self.obstacle_list])
        else:
            self.mean_obs_radius = 0.0
            self.max_obs_radius = 0.0

        # 5. 障碍物面积和自由空间面积
        obstacle_area = 0.0
        for ox, oy, size in self.obstacle_list:
            obstacle_area += math.pi * (size ** 2)

        self.free_area = max(1e-6, self.map_area - obstacle_area)
        self.obstacle_density = min(0.95, obstacle_area / self.map_area)
        if self.surface_obstacle_density is not None:
            self.obstacle_density = min(0.95, self.surface_obstacle_density)
            self.free_area = max(1e-6, self.map_area * (1.0 - self.obstacle_density))

        # 6. 自动生成全局规划参数
        self._auto_config()

        # 7. RRT* 自适应邻域半径参数
        zeta_2 = math.pi
        gamma_star = 2.0 * (1.0 + 1.0 / self.dim) ** (1.0 / self.dim) \
                     * (self.free_area / zeta_2) ** (1.0 / self.dim)

        self.gamma_rrt = 1.2 * gamma_star

        # 邻域半径上下限不要再直接依赖固定 expandDis
        self.r_min = 2.0 * self.step_min
        self.r_max = min(self.gamma_rrt, 0.35 * self.map_diag)

    def rrt_star_planning(self):
        start_time = time.time()

        self.start = self._make_node(self.start[0], self.start[1])
        self.goal = self._make_node(self.goal[0], self.goal[1])
        self.node_list = [self.start]
        self.start_to_goal_distance = self.line_cost(self.start, self.goal)

        # 当前最优路径长度，供 Informed RRT* 椭圆采样使用
        self.current_best_path_len = float('inf')

        # 初始化 KDTree
        self.kd_tree = None
        self.kd_tree_size = 0
        self.rebuild_kdtree()

        # 检查起点和终点是否合法
        if self.check_sample_collision([self.start.x, self.start.y]):
            print("路径规划失败：起点位于障碍物或安全膨胀区域内")
            return None, time.time() - start_time

        if self.check_sample_collision([self.goal.x, self.goal.y]):
            print("路径规划失败：终点位于障碍物或安全膨胀区域内")
            return None, time.time() - start_time

        path = None
        lastPathLength = float('inf')
        sampling_count = 0

        no_improve_count = 0
        first_path_found_iter = None

        for i in range(self.max_iter):

            rnd = self.sample_with_collision_check()

            if rnd is None:
                continue

            sampling_count += 1

            # 找最近节点
            n_ind = self.get_nearest_node_index(rnd)
            nearestNode = self.node_list[n_ind]

            # 生成新节点
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            # 检查 nearestNode 到 newNode 是否碰撞
            noCollision = self.check_segment_collision(
                newNode.x, newNode.y,
                nearestNode.x, nearestNode.y
            )

            if noCollision:
                # 邻域节点
                nearInds = self.find_near_nodes(newNode)

                # 祖先节点
                parentInds = self.find_ancestry_nodes(nearInds)

                # 去重，避免重复候选父节点
                nearparentInds = list(set(nearInds + parentInds))

                # 选择父节点
                newNode = self.choose_parent(newNode, nearparentInds)

                # 添加节点
                self.node_list.append(newNode)

                # 添加节点后更新 KDTree
                self.update_kdtree_after_append()

                # 重连
                self.rewire(newNode, nearInds)

                # 判断是否接近目标点
                if self.is_near_goal(newNode):
                    if self.check_segment_collision(
                            newNode.x, newNode.y,
                            self.goal.x, self.goal.y
                    ):
                        lastIndex = len(self.node_list) - 1

                        tempPath, temp_path_length = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)

                        # 第一次找到路径
                        if path is None:
                            path = tempPath
                            lastPathLength = tempPathLen
                            self.current_best_path_len = lastPathLength

                            first_path_found_iter = i
                            no_improve_count = 0

                            print(
                                f"首次找到路径：第 {i + 1} 次迭代，"
                                f"路径长度={lastPathLength:.2f}"
                            )

                        # 找到明显更短的路径
                        elif tempPathLen < lastPathLength * (1.0 - self.min_improve_ratio):
                            old_len = lastPathLength

                            path = tempPath
                            lastPathLength = tempPathLen
                            self.current_best_path_len = lastPathLength

                            no_improve_count = 0

                            print(
                                f"路径更新：第 {i + 1} 次迭代，"
                                f"{old_len:.2f} -> {lastPathLength:.2f}"
                            )

                        # 找到路径，但改善不明显
                        else:
                            no_improve_count += 1

            # 提前终止判断：必须放在 for 循环内部，但不能放在 if rnd 的深层缩进里
            if self.early_stop and path is not None:
                if no_improve_count >= self.patience:
                    print(
                        f"提前终止：第 {i + 1} 次迭代停止，"
                        f"首次找到路径迭代={first_path_found_iter + 1}，"
                        f"连续 {no_improve_count} 次无明显改善"
                    )
                    break

        end_time = time.time()
        planning_time = end_time - start_time

        if path:
            # 路径后处理：剪枝 + 重采样
            if self.use_path_post_process:
                path = self.post_process_path(path)

            path_total_length = self.get_path_len(path)

            print(
                f"路径规划完成！规划时间：{planning_time:.4f} 秒，"
                f"采样次数：{sampling_count}，"
                f"节点数：{len(self.node_list)}，"
                f"路径点个数：{len(path)}，"
                f"路径总长度：{path_total_length:.2f}"
            )
        else:
            print(
                f"路径规划失败！规划时间：{planning_time:.4f} 秒，"
                f"采样次数：{sampling_count}，"
                f"节点数：{len(self.node_list)}"
            )

        return path, planning_time

    def plan(self, plan_surface):
        """
        适配项目统一接口：
        Q_RRT_star(mapdata).plan(plan_surface) -> ([point...], elapsed_seconds)
        """
        path, elapsed = self.rrt_star_planning()
        if not path:
            return [], elapsed

        project_path = [point(p[0], p[1]) for p in path]
        self._draw_path(plan_surface, project_path)
        return project_path, elapsed

    @staticmethod
    def _obstacles_to_circles(obstacles):
        """把项目里的多边形障碍物粗略转成圆形障碍物，仅供自适应参数估计使用。"""
        circles = []
        for obs in obstacles or []:
            if not obs:
                continue
            xs = [p[0] for p in obs]
            ys = [p[1] for p in obs]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            radius = max(math.hypot(x - cx, y - cy) for x, y in obs)
            circles.append((cx, cy, radius))
        return circles

    def _make_node(self, x, y, cost=0.0, parent=None):
        """使用项目已有的 point 节点，并补充 Q-RRT* 需要的运行时属性。"""
        node = point(float(x), float(y))
        node.cost = cost
        node.parent = parent
        node.father = None
        return node

    def _init_surface_metrics(self):
        """从项目真实障碍物图层提取密度和距离场，用于自适应参数。"""
        if self.surface_obstacle is None:
            return
        pixels = pygame.surfarray.array3d(self.surface_obstacle)
        black_mask = np.all(pixels[:, :, :3] == 0, axis=2)
        self.surface_obstacle_density = float(np.mean(black_mask))
        # distance_transform_edt 计算的是非障碍像素到最近障碍像素的距离。
        self.obstacle_distance_map = distance_transform_edt(~black_mask.T)

    def _is_black_obstacle_pixel(self, x, y):
        if self.surface_obstacle is None:
            return False
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.surface_obstacle.get_width() or iy >= self.surface_obstacle.get_height():
            return True
        color = self.surface_obstacle.get_at((ix, iy))
        return color[0] == 0 and color[1] == 0 and color[2] == 0

    def _draw_path(self, plan_surface, path):
        if plan_surface is None or not path:
            return
        for node in self.node_list or []:
            pygame.draw.circle(plan_surface, (0, 120, 220), (int(node.x), int(node.y)), 1)
            if node.parent is not None:
                parent = self.node_list[node.parent]
                pygame.draw.line(plan_surface, (120, 170, 220), (parent.x, parent.y), (node.x, node.y), 1)
        for i in range(len(path) - 1):
            pygame.draw.line(plan_surface, (128, 0, 128), (path[i].x, path[i].y), (path[i + 1].x, path[i + 1].y), 4)
            pygame.draw.circle(plan_surface, (0, 100, 255), (int(path[i].x), int(path[i].y)), 3)
        pygame.draw.circle(plan_surface, (0, 100, 255), (int(path[-1].x), int(path[-1].y)), 3)
        QApplication.processEvents()

    def _infer_map_bounds(self, start, goal, obstacle_list, margin_ratio=0.15):
        """
        当外部没有提供 rand_area 时，根据起点、终点和障碍物自动推断地图范围。

        返回格式：
        (xmin, xmax, ymin, ymax)
        """

        points = []

        # 起点和终点
        points.append((start[0], start[1]))
        points.append((goal[0], goal[1]))

        # 障碍物范围
        for ob in obstacle_list:
            ox, oy, r = ob[0], ob[1], ob[2]

            points.append((ox - r, oy - r))
            points.append((ox + r, oy + r))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        width = max(xmax - xmin, 1e-6)
        height = max(ymax - ymin, 1e-6)

        # 留出边界余量，避免采样空间贴着障碍物或终点
        margin = margin_ratio * max(width, height)

        return (
            xmin - margin,
            xmax + margin,
            ymin - margin,
            ymax + margin
        )

    def _parse_map_bounds(self, rand_area):
        """
        支持两种输入：
        rand_area = (min, max)
        rand_area = (xmin, xmax, ymin, ymax)
        """
        if len(rand_area) == 2:
            self.x_min = rand_area[0]
            self.x_max = rand_area[1]
            self.y_min = rand_area[0]
            self.y_max = rand_area[1]
        elif len(rand_area) == 4:
            self.x_min = rand_area[0]
            self.x_max = rand_area[1]
            self.y_min = rand_area[2]
            self.y_max = rand_area[3]
        else:
            raise ValueError("rand_area 必须是 (min, max) 或 (xmin, xmax, ymin, ymax)")

    def _auto_config(self):
        """
        根据地图尺度、起终点距离、障碍物密度自动生成规划参数。
        外部 params 里有对应参数时优先使用外部参数；
        没有时自动生成。
        """

        # 基础尺度：综合地图对角线和起终点距离
        base_scale = max(self.start_goal_dist, 0.5 * self.map_diag)

        # ==============================
        # 1. 安全距离参数
        # ==============================
        self.robot_radius = self.params.get('robot_radius', 0.01 * self.map_diag)
        self.safety_margin = self.params.get('safety_margin', 0.01 * self.map_diag)

        # 编队半径：用于全局路径规划的额外安全膨胀
        self.formation_radius = self.params.get('formation_radius', 0.0)

        # 编队安全权重：避免完整编队半径导致全局路径过度保守
        self.formation_safety_weight = self.params.get('formation_safety_weight', 0.5)

        # 全局规划安全距离 = 机器人半径 + 安全余量 + 部分编队半径
        self.safe_clearance = (
                self.robot_radius
                + self.safety_margin
                + self.formation_safety_weight * self.formation_radius
        )

        # ==============================
        # 2. 自适应步长参数
        # 必须先定义 step_min / step_max
        # 后面 path_sample_step 才能使用 self.step_min
        # ==============================
        self.step_min = self.params.get(
            'minExpandDis',
            max(0.02 * base_scale, 0.3 * max(self.mean_obs_radius, 1e-6))
        )

        self.step_max = self.params.get(
            'maxExpandDis',
            max(0.08 * base_scale, 1.5 * self.step_min)
        )

        self.base_step = self.params.get(
            'expandDis',
            0.5 * (self.step_min + self.step_max)
        )

        # ==============================
        # 3. 路径后处理参数
        # 这里必须放在 step_min 后面
        # ==============================
        self.use_path_post_process = self.params.get('usePathPostProcess', True)

        self.path_sample_step = self.params.get(
            'pathSampleStep',
            max(0.5 * self.step_min, 0.02 * self.map_diag)
        )

        # ==============================
        # 4. 终点连接距离
        # ==============================
        self.goal_connect_dist = self.params.get(
            'goalConnectDist',
            max(self.step_max, 0.04 * self.map_diag)
        )

        # ==============================
        # 5. 障碍物距离归一化范围
        # ==============================
        self.adaptive_step_range = self.params.get(
            'adaptiveStepRange',
            max(2.0 * self.max_obs_radius, 0.12 * self.map_diag)
        )

        # ==============================
        # 6. 最大迭代次数
        # ==============================
        auto_iter = int(300 + 1200 * self.obstacle_density + 25 * len(self.obstacle_list))
        self.max_iter = self.params.get('maxIterAuto', self.params.get('maxIter', auto_iter))
        self.max_iter = max(self.max_iter, auto_iter)

        # ==============================
        # KDTree 加速参数
        # ==============================
        self.use_kdtree = self.params.get('useKDTree', True)

        # 每增加多少个节点重建一次 KDTree
        # 数值太小：重建太频繁
        # 数值太大：新节点需要线性补查较多
        self.kdtree_rebuild_interval = self.params.get(
            'kdtreeRebuildInterval',
            max(10, int(0.02 * self.max_iter))
        )

        # ==============================
        # Informed RRT* 椭圆采样参数
        # ==============================
        self.use_informed_sampling = self.params.get('useInformedSampling', True)

        # 找到第一条路径后，有多少概率在椭圆区域采样
        # 建议 70~90，过高可能导致探索不足
        self.informed_sample_rate = self.params.get('informedSampleRate', 80)

        # ==============================
        # 7. 目标采样率
        # ==============================
        auto_goal_rate = int(25 - 15 * self.obstacle_density)
        auto_goal_rate = min(30, max(5, auto_goal_rate))
        self.goal_sample_rate = self.params.get('goalSampleRateAuto', auto_goal_rate)

        # ==============================
        # 8. 采样尝试次数
        # ==============================
        self.max_sample_attempts = self.params.get(
            'maxSampleAttempts',
            max(100, int(50 + 200 * self.obstacle_density))
        )

        # ==============================
        # 9. 祖先节点深度
        # ==============================
        self.d_ancestor = self.params.get(
            'd_ancestor',
            1 if self.obstacle_density < 0.25 else 2
        )

        # ==============================
        # 10. 提前终止参数
        # ==============================
        self.early_stop = True
        self.min_improve_ratio = 0.01
        self.patience = self.params.get(
            'patience',
            max(40, int(0.12 * self.max_iter))
        )

    def sample_with_collision_check(self):
        """
        生成采样点并检查其是否与障碍物碰撞。
        """
        for _ in range(self.max_sample_attempts):
            rnd = self.sample()

            if rnd is None:
                continue

            # Informed 椭圆采样可能落到地图外，地图外采样点直接丢弃
            if not self.is_inside_map(rnd):
                continue

            if not self.check_sample_collision(rnd):
                return rnd

        return None

    def check_sample_collision(self, rnd):
        """
        检查采样点是否与障碍物或安全膨胀区域发生碰撞。
        """
        if self._is_black_obstacle_pixel(rnd[0], rnd[1]):
            return True
        if self.surface_obstacle is not None:
            return False
        for (ox, oy, size) in self.obstacle_list:
            safe_r = size + self.safe_clearance
            if (rnd[0] - ox) ** 2 + (rnd[1] - oy) ** 2 <= safe_r ** 2:
                return True
        return False

    def check_segment_collision(self, x1, y1, x2, y2):
        if self.surface_obstacle is not None:
            return not self.collision((x1, y1), (x2, y2))

        distance = math.hypot(x2 - x1, y2 - y1)
        if distance < 1e-6:
            return not self.check_sample_collision([x1, y1])

        for (ox, oy, size) in self.obstacle_list:
            safe_r = size + self.safe_clearance
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]),
                np.array([x2, y2]),
                np.array([ox, oy])
            )
            if dd <= safe_r ** 2:
                return False
        return True

    def collision(self, src, dst):
        """
        使用项目 RRT 同款地图碰撞检测：沿线段逐像素检查 obs_surface 上的黑色障碍物。
        返回 True 表示发生碰撞。
        """
        vx, vy = self.normalize(dst[0] - src[0], dst[1] - src[1])
        curr = list(src)
        if math.hypot(curr[0] - dst[0], curr[1] - dst[1]) <= 1:
            return self._is_black_obstacle_pixel(curr[0], curr[1])
        while math.hypot(curr[0] - dst[0], curr[1] - dst[1]) > 1:
            if self._is_black_obstacle_pixel(curr[0], curr[1]):
                return True
            curr[0] += vx
            curr[1] += vy
        return False

    def normalize(self, vx, vy):
        norm = math.sqrt(vx * vx + vy * vy)
        if norm > 1e-6:
            return vx / norm, vy / norm
        return 0, 0

    def get_nearest_obstacle_distance(self, node):
        """
        计算节点到最近障碍物边界的距离。
        """
        if self.obstacle_distance_map is not None:
            ix = int(round(node.x))
            iy = int(round(node.y))
            if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
                return 0.0
            return float(self.obstacle_distance_map[iy, ix])

        if len(self.obstacle_list) == 0:
            return self.map_diag

        min_dist = float('inf')
        for ox, oy, size in self.obstacle_list:
            center_dist = math.hypot(node.x - ox, node.y - oy)
            boundary_dist = center_dist - size - self.safe_clearance
            min_dist = min(min_dist, boundary_dist)

        return max(0.0, min_dist)

    def get_adaptive_expand_dis(self, nearestNode):
        """
        距离障碍物越远，步长越大；
        距离障碍物越近，步长越小。
        """
        obs_dist = self.get_nearest_obstacle_distance(nearestNode)

        ratio = obs_dist / max(1e-6, self.adaptive_step_range)
        ratio = min(1.0, max(0.0, ratio))

        step = self.step_min + ratio * (self.step_max - self.step_min)

        return step

    def is_inside_map(self, rnd):
        """
        判断采样点是否在地图范围内。
        """
        return (
                self.x_min <= rnd[0] <= self.x_max
                and self.y_min <= rnd[1] <= self.y_max
        )

    def sample_unit_ball(self):
        """
        在二维单位圆内均匀采样。
        """
        r = math.sqrt(random.random())
        theta = random.uniform(0.0, 2.0 * math.pi)

        return np.array([
            r * math.cos(theta),
            r * math.sin(theta)
        ])

    def informed_sample(self):
        """
        Informed RRT* 椭圆采样。

        找到第一条路径后，将采样空间限制在以起点和终点为焦点、
        当前最优路径长度为长轴的椭圆区域内。
        """
        if not self.use_informed_sampling:
            return None

        if self.current_best_path_len == float('inf'):
            return None

        c_best = self.current_best_path_len
        c_min = self.start_to_goal_distance

        # 如果当前路径长度接近起终点直线距离，椭圆退化，直接不使用 informed 采样
        if c_best <= c_min + 1e-6:
            return None

        # 椭圆中心
        center = np.array([
            (self.start.x + self.goal.x) / 2.0,
            (self.start.y + self.goal.y) / 2.0
        ])

        # 椭圆长半轴和短半轴
        a = c_best / 2.0
        b = math.sqrt(max(c_best ** 2 - c_min ** 2, 0.0)) / 2.0

        # 在单位圆内采样
        x_ball = self.sample_unit_ball()

        # 缩放到椭圆
        x_ellipse = np.array([
            a * x_ball[0],
            b * x_ball[1]
        ])

        # 椭圆方向：从起点指向终点
        angle = math.atan2(
            self.goal.y - self.start.y,
            self.goal.x - self.start.x
        )

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        rotation = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        x_world = rotation @ x_ellipse + center

        return [float(x_world[0]), float(x_world[1])]

    def sample(self):
        """
        采样策略：
        1. 未找到路径前：全局随机采样 + 目标偏置采样；
        2. 找到第一条路径后：以较高概率使用 Informed RRT* 椭圆采样；
        3. 保留一定概率的全局采样，避免搜索空间过早收缩。
        """

        # 已经找到第一条路径后，优先进行 Informed 椭圆采样
        if (
                self.use_informed_sampling
                and self.current_best_path_len < float('inf')
                and random.randint(0, 100) < self.informed_sample_rate
        ):
            rnd = self.informed_sample()

            if rnd is not None and self.is_inside_map(rnd):
                return rnd

        # 否则执行原来的全局采样 + 目标偏置
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [
                random.uniform(self.x_min, self.x_max),
                random.uniform(self.y_min, self.y_max)
            ]
        else:
            rnd = [self.goal.x, self.goal.y]

        return rnd

    def rebuild_kdtree(self):
        """
        根据当前 node_list 重建 KDTree。
        KDTree 只存储节点坐标，用于加速最近邻和邻域搜索。
        """
        if not self.use_kdtree or self.node_list is None or len(self.node_list) == 0:
            self.kd_tree = None
            self.kd_tree_size = 0
            return

        points = np.array([[node.x, node.y] for node in self.node_list])
        self.kd_tree = cKDTree(points)
        self.kd_tree_size = len(self.node_list)

    def update_kdtree_after_append(self):
        """
        添加新节点后，按固定间隔重建 KDTree。
        没有进入 KDTree 的新增节点，会在查询时用线性补查保证正确性。
        """
        if not self.use_kdtree:
            return

        if self.kd_tree is None:
            self.rebuild_kdtree()
            return

        pending_count = len(self.node_list) - self.kd_tree_size

        if pending_count >= self.kdtree_rebuild_interval:
            self.rebuild_kdtree()

    def get_nearest_node_index(self, rnd):
        """
        使用 KDTree 查询最近节点；
        对尚未重建进 KDTree 的新增节点进行线性补查。
        """
        if not self.use_kdtree or self.kd_tree is None:
            return self.get_nearestList_index(self.node_list, rnd)

        query_point = np.array([rnd[0], rnd[1]])

        # 1. 在 KDTree 已覆盖的节点中找最近节点
        _, idx = self.kd_tree.query(query_point)
        best_index = int(idx)

        best_dist = (
                (self.node_list[best_index].x - rnd[0]) ** 2
                + (self.node_list[best_index].y - rnd[1]) ** 2
        )

        # 2. 线性检查 KDTree 尚未包含的新节点
        for i in range(self.kd_tree_size, len(self.node_list)):
            node = self.node_list[i]
            d = (node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2

            if d < best_dist:
                best_dist = d
                best_index = i

        return best_index

    # 找到与采样点最近距离的点
    @staticmethod
    def get_nearestList_index(nodes, rnd):
        # 利用两点之间距离公式计算出每一个点和采样点的距离
        dList = [(node.x - rnd[0]) ** 2
                + (node.y - rnd[1]) ** 2 for node in nodes]
        # 返回与采样点最近的点在集合中的位置索引
        minIndex = dList.index(min(dList))
        return minIndex

    def get_new_node(self, theta, n_ind, nearestNode):
        adaptive_expand_dis = self.get_adaptive_expand_dis(nearestNode)

        new_x = nearestNode.x + adaptive_expand_dis * math.cos(theta)
        new_y = nearestNode.y + adaptive_expand_dis * math.sin(theta)

        # 防止新节点跑出地图边界
        new_x = min(max(new_x, self.x_min), self.x_max)
        new_y = min(max(new_y, self.y_min), self.y_max)

        return self._make_node(new_x, new_y, nearestNode.cost + adaptive_expand_dis, n_ind)

    # 计算点到直线的距离方法
    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        #第一种情况,两端点在同于一个点上时
        if np.array_equal(v, w):
            return (p - v).dot(p - v)
        # 如果不再同一点就计算v-w的距离
        l2 = (w - v).dot(w - v)
        # 计算相似三角形比例用于求圆心到直线的距离
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        # 算出投影点
        project = v + t * (w - v)
        # 返回圆心到直线距离的平方值
        return (p - project).dot(p - project)

    def find_near_nodes(self, newNode):
        """
        使用自适应邻域半径 r(n) 的 RRT* 邻域搜索。
        若启用 KDTree，则使用 KDTree 加速半径查询；
        对尚未进入 KDTree 的新增节点进行线性补查。
        """
        n_node = len(self.node_list) + 1
        if n_node <= 1:
            return []

        # RRT* 理论邻域半径
        r = self.gamma_rrt * math.sqrt(math.log(n_node) / n_node)

        # 半径上下限约束
        r = max(self.r_min, min(r, self.r_max))

        query_point = np.array([newNode.x, newNode.y])

        # 不使用 KDTree 时，回退到原始线性搜索
        if not self.use_kdtree or self.kd_tree is None:
            dlist = [
                (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
                for node in self.node_list
            ]
            return [idx for idx, d in enumerate(dlist) if d <= r ** 2]

        # 1. KDTree 查询已包含节点
        nearInds = self.kd_tree.query_ball_point(query_point, r)
        nearInds = list(nearInds)

        # 2. 线性补查尚未进入 KDTree 的新增节点
        r2 = r ** 2
        for i in range(self.kd_tree_size, len(self.node_list)):
            node = self.node_list[i]
            d = (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2

            if d <= r2:
                nearInds.append(i)

        # 去重，防止极端情况下重复
        nearInds = list(set(nearInds))

        return nearInds

    def find_ancestry_nodes(self, nearInds):
        X_parent = []

        # 遍历邻域节点索引集Qn
        for q_index in nearInds:
            current_depth = 0
            current_index = q_index

            # 向上遍历树结构，直到达到指定深度
            while current_depth < self.d_ancestor:
                parent_index = self.node_list[current_index].parent

                # 如果找到了父节点，则更新深度
                if parent_index is not None:
                    current_index = parent_index
                    current_depth += 1
                else:
                    break

            # 如果查询深度达标，则将当前节点索引添加到Qparent
            if current_depth == self.d_ancestor:
                X_parent.append(current_index)

        return X_parent

    def choose_parent(self, newNode, nearInds):
        # 判断节点范围内的集合列表是否为空，为空证明无候选节点 直接返回原节点
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        # 判断新的父节点是否是无穷大，如果是无穷大就碰撞了
        if minCost == float('inf'):
            print("min cost is inf")
            return newNode

        # 更新节点代价以及父节点
        newNode.cost = minCost
        newNode.parent = minInd
        return newNode

    def check_collision(self, nearNode, theta, d):
        end_x = nearNode.x + math.cos(theta) * d
        end_y = nearNode.y + math.sin(theta) * d
        return self.check_segment_collision(nearNode.x, nearNode.y, end_x, end_y)

    def rewire(self, newNode, nearInds):
        """
        重连阶段：让附近节点尝试连接到 newNode 的父节点，而不是 newNode 本身
        """
        # 如果 newNode 没有父节点，无法进行重连
        if newNode.parent is None:
            return

        parent_index = newNode.parent
        parent_node = self.node_list[parent_index]

        for i in nearInds:
            nearNode = self.node_list[i]

            # 计算 父节点 → 邻域节点 的距离
            dx = nearNode.x - parent_node.x
            dy = nearNode.y - parent_node.y
            d = math.hypot(dx, dy)

            # 经过父节点到达邻域节点的候选代价
            s_cost = parent_node.cost + d

            # 如果候选代价更小，检查碰撞
            if nearNode.cost > s_cost:
                theta = math.atan2(dy, dx)  # 父节点指向邻域节点的方向

                # 检查 父节点 → 邻域节点 这条线段是否无碰撞
                if self.check_collision(parent_node, theta, d):
                    nearNode.parent = parent_index
                    nearNode.cost = s_cost

                    # 关键修改：递归更新 nearNode 的所有后代节点 cost
                    self.propagate_cost_to_leaves(i)

    def propagate_cost_to_leaves(self, parent_index):
        """
        当某个节点的父节点或 cost 改变后，
        递归更新它所有子节点、孙节点的累计代价。
        """
        parent_node = self.node_list[parent_index]

        for i, node in enumerate(self.node_list):
            if node.parent == parent_index:
                # 子节点 cost = 父节点 cost + 父子节点之间的距离
                node.cost = parent_node.cost + self.line_cost(parent_node, node)

                # 继续更新该子节点的后代
                self.propagate_cost_to_leaves(i)

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        return d < self.goal_connect_dist

    def post_process_path(self, path):
        """
        路径后处理：
        1. 路径剪枝；
        2. 等间距重采样；
        3. 安全性复检。
        """
        if path is None or len(path) <= 2:
            return path

        raw_len = self.get_path_len(path)
        raw_points = len(path)

        # 1. 剪枝
        pruned_path = self.prune_path(path)

        # 2. 重采样
        resampled_path = self.resample_path(pruned_path)

        # 3. 安全性检查
        if self.is_path_collision_free(resampled_path):
            new_len = self.get_path_len(resampled_path)
            print(
                f"路径后处理完成：点数 {raw_points} -> {len(resampled_path)}，"
                f"长度 {raw_len:.2f} -> {new_len:.2f}"
            )
            return resampled_path

        print("路径后处理结果存在碰撞，保留原始路径")
        return path

    def prune_path(self, path):
        """
        路径剪枝：
        如果 path[i] 可以直接连接到 path[j] 且无碰撞，
        就删除 i 和 j 之间的冗余路径点。
        """
        if path is None or len(path) <= 2:
            return path

        pruned_path = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1

            # 从终点往前找，尽量连接最远的可达点
            while j > i + 1:
                if self.check_segment_collision(
                        path[i][0], path[i][1],
                        path[j][0], path[j][1]
                ):
                    break
                j -= 1

            # 如果没有找到更远的可直连点，就至少前进一步
            if j == i:
                j = i + 1

            pruned_path.append(path[j])
            i = j

        return pruned_path

    def resample_path(self, path, step=None):
        """
        对路径进行等间距重采样。
        作用：
        1. 避免剪枝后路径点过少；
        2. 保证 leader 沿路径运动时航向变化更平稳；
        3. 保证 main.py 中 future_point 的计算更稳定。
        """
        if path is None or len(path) <= 1:
            return path

        if step is None:
            step = self.path_sample_step

        resampled_path = [path[0]]

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)

            if dist < 1e-6:
                continue

            # 当前线段至少分成 1 段
            n_segment = max(1, int(math.ceil(dist / step)))

            for k in range(1, n_segment + 1):
                t = k / n_segment
                new_x = x1 + t * dx
                new_y = y1 + t * dy

                # 避免重复点
                last_x, last_y = resampled_path[-1]
                if math.hypot(new_x - last_x, new_y - last_y) > 1e-6:
                    resampled_path.append([new_x, new_y])

        return resampled_path

    def is_path_collision_free(self, path):
        """
        检查整条路径是否无碰撞。
        """
        if path is None or len(path) <= 1:
            return False

        for i in range(len(path) - 1):
            if not self.check_segment_collision(
                    path[i][0], path[i][1],
                    path[i + 1][0], path[i + 1][1]
            ):
                return False

        return True

    def get_final_course(self, lastIndex):
        # 将目标点作为路径的初始节点
        path = [[self.goal.x, self.goal.y]]
        # 循环直到找到没有父节点的节点，就是初始点
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])

        # 反转路径，使其从起点到终点
        path.reverse()

        path_length = len(path)  # 计算路径点的个数
        return path, path_length

    @staticmethod
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y) ** 2)
        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
