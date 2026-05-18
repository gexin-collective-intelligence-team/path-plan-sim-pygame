import math
import random
import time
import numpy as np
import pygame
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import QApplication

from .Node import point


class _PlannerNode:
    """Q-RRT* 内部轻量节点，避免项目 point 的全局去重扫描拖慢采样树。"""

    __slots__ = ("x", "y", "cost", "parent", "father")

    def __init__(self, x, y, cost=0.0, parent=None):
        self.x = float(x)
        self.y = float(y)
        self.cost = cost
        self.parent = parent
        self.father = None


# ====================== 路径规划类 ======================
class Q_RRT_star:

    def __init__(self, start, goal=None, obstacle_list=None, rand_area=None, params=None):
        self.mapdata = None
        self.surface_obstacle = None
        self.obstacle_mask = None
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
            rand_area = rand_area if rand_area is not None else (0, self.width - 1, 0, self.height - 1)
            project_defaults = {
                "agent_radius": 0.0,
                "clearance": 0.0,
                "maxIterAuto": 2200,
                "maxSampleAttempts": 80,
                "patience": 90,
            }
            params = {**project_defaults, **(params or {})}

        self.start = start
        self.goal = goal
        self._start_xy = self._as_xy(start)
        self._goal_xy = self._as_xy(goal)
        self.obstacle_list = obstacle_list if obstacle_list is not None else []
        self.params = params if params is not None else {}
        self.node_list = None
        self.children = None
        self.start_to_goal_distance = None
        if rand_area is None:
            rand_area = self._infer_map_bounds(
                start=self._start_xy,
                goal=self._goal_xy,
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
        sx, sy = self._start_xy
        gx, gy = self._goal_xy
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

    def plan_path(self):
        """
        只执行 Q-RRT* 路径规划，不做任何绘图。

        Returns:
            path: [[x, y], ...] 或 None
            elapsed: 规划耗时
            stats: 规划统计信息
        """
        start_time = time.time()

        self.start = self._make_node(self._start_xy[0], self._start_xy[1])
        self.goal = self._make_node(self._goal_xy[0], self._goal_xy[1])
        self.node_list = [self.start]
        self.children = {0: set()}
        self.start_to_goal_distance = self.line_cost(self.start, self.goal)

        # 当前最优路径长度，供 Informed RRT* 椭圆采样使用
        self.current_best_path_len = float('inf')

        # 初始化 KDTree
        self.kd_tree = None
        self.kd_tree_size = 0
        self.rebuild_kdtree()

        # 检查起点和终点是否合法
        if self.check_sample_collision([self.start.x, self.start.y]):
            elapsed = time.time() - start_time
            stats = {
                'success': False,
                'reason': 'start_in_obstacle_or_clearance',
                'sampling_count': 0,
                'node_count': len(self.node_list),
                'path_point_count': 0,
                'path_length': None,
                'first_path_found_iter': None,
            }
            print("路径规划失败：起点位于障碍物或安全膨胀区域内")
            return None, elapsed, stats

        if self.check_sample_collision([self.goal.x, self.goal.y]):
            elapsed = time.time() - start_time
            stats = {
                'success': False,
                'reason': 'goal_in_obstacle_or_clearance',
                'sampling_count': 0,
                'node_count': len(self.node_list),
                'path_point_count': 0,
                'path_length': None,
                'first_path_found_iter': None,
            }
            print("路径规划失败：终点位于障碍物或安全膨胀区域内")
            return None, elapsed, stats

        path = None
        lastPathLength = float('inf')
        sampling_count = 0

        no_improve_count = 0
        iterations_after_first_path = 0
        iterations_since_best_path = 0
        first_path_found_iter = None
        timeout_reached = False

        for i in range(self.max_iter):
            if self.max_plan_time is not None and time.time() - start_time >= self.max_plan_time:
                timeout_reached = True
                print(f"达到单次规划时间上限 {self.max_plan_time:.1f} 秒，停止本轮继续优化")
                break

            if i % 50 == 0:
                QApplication.processEvents()

            improved_this_iter = False

            rnd = self.sample_with_collision_check()

            if rnd is None:
                if path is not None:
                    iterations_after_first_path += 1
                    iterations_since_best_path += 1
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
                nearparentInds = self._merge_parent_candidates(nearInds, parentInds)

                # 选择父节点
                newNode = self.choose_parent(newNode, nearparentInds)

                # 添加节点
                self.node_list.append(newNode)
                self._register_child(len(self.node_list) - 1, newNode.parent)

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

                        tempPath, _ = self.get_final_course(lastIndex)
                        if not tempPath:
                            continue
                        tempPathLen = self.get_path_len(tempPath)

                        # 第一次找到路径
                        if path is None:
                            path = tempPath
                            lastPathLength = tempPathLen
                            self.current_best_path_len = lastPathLength

                            first_path_found_iter = i
                            no_improve_count = 0
                            iterations_after_first_path = 0
                            iterations_since_best_path = 0
                            improved_this_iter = True

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
                            iterations_since_best_path = 0
                            improved_this_iter = True

                            print(
                                f"路径更新：第 {i + 1} 次迭代，"
                                f"{old_len:.2f} -> {lastPathLength:.2f}"
                            )

                        # 找到路径，但改善不明显
                        else:
                            no_improve_count += 1

            if path is not None and not improved_this_iter:
                iterations_after_first_path += 1
                iterations_since_best_path += 1

            # 提前终止判断：必须放在 for 循环内部，但不能放在 if rnd 的深层缩进里
            if self.early_stop and path is not None:
                enough_optimization = iterations_after_first_path >= self.min_optimize_iterations_after_first_path
                goal_updates_stalled = no_improve_count >= self.patience
                search_stalled = iterations_since_best_path >= self.max_iterations_without_improvement
                if enough_optimization and (goal_updates_stalled or search_stalled):
                    print(
                        f"提前终止：第 {i + 1} 次迭代停止，"
                        f"首次找到路径迭代={first_path_found_iter + 1}，"
                        f"首条路径后优化{iterations_after_first_path}轮，"
                        f"连续{iterations_since_best_path}轮无明显改善"
                    )
                    break

        end_time = time.time()
        planning_time = end_time - start_time

        if path:
            # 路径后处理：剪枝 + 安全性复检
            if self.use_path_post_process:
                path = self.post_process_path(path)

            path_total_length = self.get_path_len(path)
            stats = {
                'success': True,
                'reason': 'timeout_with_path' if timeout_reached else None,
                'sampling_count': sampling_count,
                'node_count': len(self.node_list),
                'path_point_count': len(path),
                'path_length': path_total_length,
                'first_path_found_iter': first_path_found_iter,
                'iterations': i + 1 if 'i' in locals() else 0,
                'elapsed': planning_time,
            }

            print(
                f"路径规划完成！规划时间：{planning_time:.4f} 秒，"
                f"采样次数：{sampling_count}，"
                f"节点数：{len(self.node_list)}，"
                f"路径点个数：{len(path)}，"
                f"路径总长度：{path_total_length:.2f}"
            )
        else:
            stats = {
                'success': False,
                'reason': 'timeout' if timeout_reached else 'no_path_found',
                'sampling_count': sampling_count,
                'node_count': len(self.node_list),
                'path_point_count': 0,
                'path_length': None,
                'first_path_found_iter': first_path_found_iter,
                'iterations': i + 1 if 'i' in locals() else 0,
                'elapsed': planning_time,
            }
            print(
                f"路径规划失败！规划时间：{planning_time:.4f} 秒，"
                f"采样次数：{sampling_count}，"
                f"节点数：{len(self.node_list)}"
            )

        return path, planning_time, stats

    def plan(self, plan_surface):
        """
        适配项目统一接口：
        Q_RRT_star(mapdata).plan(plan_surface) -> ([point...], elapsed_seconds)
        """
        path, elapsed, stats = self.plan_path()
        self.last_plan_stats = stats
        if not path:
            return [], elapsed

        project_path = [point(p[0], p[1]) for p in path]
        return project_path, elapsed

    def rrt_star_planning(self):
        """
        兼容旧调用：只返回 path 和 elapsed。
        """
        path, elapsed, _ = self.plan_path()
        return path, elapsed

    def _make_node(self, x, y, cost=0.0, parent=None):
        """创建 Q-RRT* 内部节点；最终输出时再转为项目 point。"""
        return _PlannerNode(x, y, cost, parent)

    @staticmethod
    def _as_xy(value):
        if hasattr(value, 'x') and hasattr(value, 'y'):
            return (value.x, value.y)
        return (value[0], value[1])

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

    def _init_surface_metrics(self):
        """从项目真实障碍物图层提取黑色障碍物掩码和距离场。"""
        if self.surface_obstacle is None:
            return
        pixels = pygame.surfarray.array3d(self.surface_obstacle)
        black_mask = np.all(pixels[:, :, :3] == 0, axis=2)
        self.obstacle_mask = black_mask.T
        self.surface_obstacle_density = float(np.mean(self.obstacle_mask))
        if np.any(self.obstacle_mask):
            # distance_transform_edt 在非障碍像素上给出到最近障碍像素的欧氏距离。
            self.obstacle_distance_map = distance_transform_edt(~self.obstacle_mask)

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
        self.agent_radius = self.params.get(
            'agent_radius',
            self.params.get('robot_radius', 0.01 * self.map_diag)
        )
        self.clearance = self.params.get(
            'clearance',
            self.params.get('safety_margin', 0.01 * self.map_diag)
        )

        self.safe_clearance = self.params.get(
            'safe_clearance',
            self.agent_radius + self.clearance
        )

        # 兼容旧代码/旧参数命名
        self.robot_radius = self.agent_radius
        self.safety_margin = self.clearance

        # ==============================
        # 2. 自适应步长参数
        # ==============================
        self.step_min = self.params.get(
            'minExpandDis',
            max(0.02 * base_scale, 0.3 * max(self.mean_obs_radius, 1e-6))
        )

        self.step_max = self.params.get(
            'maxExpandDis',
            max(0.08 * base_scale, 1.5 * self.step_min)
        )

        # ==============================
        # 3. 路径后处理参数
        # ==============================
        self.use_path_post_process = self.params.get('usePathPostProcess', True)

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
        self.min_optimize_iterations_after_first_path = self.params.get(
            'minOptimizeIterationsAfterFirstPath',
            max(150, int(0.15 * self.max_iter))
        )
        self.max_iterations_without_improvement = self.params.get(
            'maxIterationsWithoutImprovement',
            max(self.patience * 3, int(0.25 * self.max_iter))
        )
        self.max_plan_time = self.params.get('maxPlanTime', 8.0)

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

    def is_inside_map(self, rnd):
        """
        判断采样点是否在地图范围内。
        """
        return (
                self.x_min <= rnd[0] <= self.x_max
                and self.y_min <= rnd[1] <= self.y_max
        )

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

    def get_nearest_obstacle_distance(self, node):
        """
        计算节点到最近障碍物边界的距离。
        """
        if self.surface_obstacle is not None:
            if self.obstacle_distance_map is None:
                return self.map_diag
            ix = int(round(node.x))
            iy = int(round(node.y))
            if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
                return 0.0
            return max(0.0, float(self.obstacle_distance_map[iy, ix]))

        if len(self.obstacle_list) == 0:
            return self.map_diag

        min_dist = float('inf')
        for ox, oy, size in self.obstacle_list:
            center_dist = math.hypot(node.x - ox, node.y - oy)
            boundary_dist = center_dist - size - self.safe_clearance
            min_dist = min(min_dist, boundary_dist)

        return max(0.0, min_dist)

    def check_sample_collision(self, rnd):
        """
        检查采样点是否与障碍物或安全膨胀区域发生碰撞。
        """
        if self.surface_obstacle is not None:
            return self._is_obstacle_pixel_fast(rnd[0], rnd[1])
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
        使用项目 RRT 同款地图语义：沿线段检查 obs_surface 上的黑色障碍物。
        返回 True 表示发生碰撞。
        """
        if self.obstacle_mask is None:
            for (ox, oy, size) in self.obstacle_list:
                safe_r = size + self.safe_clearance
                dd = self.distance_squared_point_to_segment(
                    np.array([src[0], src[1]]),
                    np.array([dst[0], dst[1]]),
                    np.array([ox, oy])
                )
                if dd <= safe_r ** 2:
                    return True
            return False

        distance = math.hypot(dst[0] - src[0], dst[1] - src[1])
        steps = max(2, int(math.ceil(distance)) + 1)
        xs = np.linspace(src[0], dst[0], steps).astype(np.int64)
        ys = np.linspace(src[1], dst[1], steps).astype(np.int64)

        out_of_bounds = (
            (xs < 0) | (ys < 0) |
            (xs >= self.width) | (ys >= self.height)
        )
        if np.any(out_of_bounds):
            return True

        return bool(np.any(self.obstacle_mask[ys, xs]))

    def _is_obstacle_pixel_fast(self, x, y):
        if self.obstacle_mask is None:
            return False
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        return bool(self.obstacle_mask[iy, ix])

    def check_collision(self, nearNode, theta, d):
        end_x = nearNode.x + math.cos(theta) * d
        end_y = nearNode.y + math.sin(theta) * d
        return self.check_segment_collision(nearNode.x, nearNode.y, end_x, end_y)

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
        seen = set()

        # 遍历邻域节点索引集Qn
        for q_index in nearInds:
            if q_index is None or q_index < 0 or q_index >= len(self.node_list):
                continue
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
            if current_depth == self.d_ancestor and current_index not in seen:
                seen.add(current_index)
                X_parent.append(current_index)

        return X_parent

    def _merge_parent_candidates(self, nearInds, parentInds):
        candidates = []
        seen = set()
        for idx in nearInds + parentInds:
            if idx is None or idx < 0 or idx >= len(self.node_list):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            candidates.append(idx)
        return candidates

    def choose_parent(self, newNode, nearInds):
        # 判断节点范围内的集合列表是否为空，为空证明无候选节点 直接返回原节点
        if len(nearInds) == 0:
            return newNode

        candidates = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            candidates.append((self.node_list[i].cost + d, i, d, dx, dy))

        candidates.sort(key=lambda item: item[0])
        for minCost, minInd, d, dx, dy in candidates:
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[minInd], theta, d):
                # 更新节点代价以及父节点
                newNode.cost = minCost
                newNode.parent = minInd
                return newNode

        print("min cost is inf")
        return newNode

    def _register_child(self, child_index, parent_index):
        if self.children is None:
            return
        self.children.setdefault(child_index, set())
        if parent_index is None:
            return
        if parent_index < 0 or parent_index >= len(self.node_list):
            return
        self.children.setdefault(parent_index, set()).add(child_index)

    def _move_child(self, child_index, old_parent, new_parent):
        if self.children is None:
            return
        self.children.setdefault(child_index, set())
        if old_parent is not None and old_parent in self.children:
            self.children[old_parent].discard(child_index)
        if new_parent is not None:
            self.children.setdefault(new_parent, set()).add(child_index)

    def rewire(self, newNode, nearInds):
        """
        重连阶段：附近节点同时尝试连接 newNode 和 newNode 的父节点。
        """
        if not self.node_list or self.node_list[-1] is not newNode:
            return

        new_node_index = len(self.node_list) - 1
        candidate_parents = [(new_node_index, newNode)]
        if newNode.parent is not None:
            candidate_parents.append((newNode.parent, self.node_list[newNode.parent]))

        for i in nearInds:
            if i == new_node_index:
                continue
            nearNode = self.node_list[i]
            best_parent_index = None
            best_cost = nearNode.cost
            best_distance = None
            best_dx = None
            best_dy = None

            for parent_index, parent_node in candidate_parents:
                if i == parent_index:
                    continue
                if self._is_descendant(parent_index, i):
                    continue
                dx = nearNode.x - parent_node.x
                dy = nearNode.y - parent_node.y
                d = math.hypot(dx, dy)
                candidate_cost = parent_node.cost + d
                if candidate_cost < best_cost:
                    best_parent_index = parent_index
                    best_cost = candidate_cost
                    best_distance = d
                    best_dx = dx
                    best_dy = dy

            # 如果候选代价更小，检查碰撞
            if best_parent_index is not None:
                theta = math.atan2(best_dy, best_dx)
                parent_node = self.node_list[best_parent_index]

                # 检查候选父节点 → 邻域节点 这条线段是否无碰撞
                if self.check_collision(parent_node, theta, best_distance):
                    old_parent = nearNode.parent
                    old_cost = nearNode.cost
                    nearNode.parent = best_parent_index
                    nearNode.cost = best_cost
                    if self._has_parent_cycle_from(i):
                        nearNode.parent = old_parent
                        nearNode.cost = old_cost
                        continue

                    self._move_child(i, old_parent, best_parent_index)
                    self.propagate_cost_to_leaves(i)

    def _is_descendant(self, node_index, possible_ancestor_index):
        """
        判断 node_index 是否已经是 possible_ancestor_index 的后代。
        防止重连时把祖先接到后代上，形成树环。
        """
        current_index = node_index
        visited = set()
        while current_index is not None:
            if current_index == possible_ancestor_index:
                return True
            if current_index in visited:
                return True
            visited.add(current_index)
            if current_index < 0 or current_index >= len(self.node_list):
                return False
            current_index = self.node_list[current_index].parent
        return False

    def _has_parent_cycle_from(self, start_index):
        current_index = start_index
        visited = set()
        while current_index is not None:
            if current_index in visited:
                return True
            visited.add(current_index)
            if current_index < 0 or current_index >= len(self.node_list):
                return True
            current_index = self.node_list[current_index].parent
        return False

    def propagate_cost_to_leaves(self, parent_index):
        """
        当某个节点的父节点或 cost 改变后，
        更新它所有子节点、孙节点的累计代价。
        """
        if self.children is None:
            return
        stack = list(self.children.get(parent_index, ()))
        visited = set()
        while stack:
            child_index = stack.pop()
            if child_index in visited:
                continue
            visited.add(child_index)
            if child_index < 0 or child_index >= len(self.node_list):
                continue
            node = self.node_list[child_index]
            if node.parent is None:
                continue
            parent = self.node_list[node.parent]
            node.cost = parent.cost + self.line_cost(parent, node)
            stack.extend(self.children.get(child_index, ()))

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        return d < self.goal_connect_dist

    def get_final_course(self, lastIndex):
        # 将目标点作为路径的初始节点
        path = [[self.goal.x, self.goal.y]]
        # 循环直到找到没有父节点的节点，就是初始点
        visited = set()
        while self.node_list[lastIndex].parent is not None:
            if lastIndex in visited:
                print("路径回溯失败：检测到父节点环，跳过当前候选路径")
                return None, 0
            visited.add(lastIndex)
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
            if lastIndex < 0 or lastIndex >= len(self.node_list):
                print("路径回溯失败：父节点索引越界，跳过当前候选路径")
                return None, 0
        path.append([self.start.x, self.start.y])

        # 反转路径，使其从起点到终点
        path.reverse()

        path_length = len(path)  # 计算路径点的个数
        return path, path_length

    def post_process_path(self, path):
        """
        路径后处理：
        1. 路径剪枝；
        2. 安全性复检。
        """
        if path is None or len(path) <= 2:
            return path

        raw_len = self.get_path_len(path)
        raw_points = len(path)

        # 1. 剪枝
        pruned_path = self.prune_path(path)

        # 2. 安全性检查
        if self.is_path_collision_free(pruned_path):
            new_len = self.get_path_len(pruned_path)
            print(
                f"路径后处理完成：点数 {raw_points} -> {len(pruned_path)}，"
                f"长度 {raw_len:.2f} -> {new_len:.2f}"
            )
            return pruned_path

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
