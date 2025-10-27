import math
import time
from random import random
import pygame
from PyQt5.QtWidgets import QApplication
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from .Node import point


class Cost_Rrt:
    def __init__(self, mapdata):
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])
        self.result = []
        self.width = mapdata.width
        self.obstacle = mapdata.obs_surface
        self.height = mapdata.height
        self.tree = []
        self.step = 15  # 节点扩展步长
        self.max_iterations = 10000
        self.node_count = 0
        self.static_polygons = [Polygon(obs) for obs in mapdata.obstacles]  # 障碍物多边形列表
        self.fail_count = 0  # 扩展失败计数
        #self.p_goal = 0.3  # 目标偏向概率：30%概率直接采样目标点
        # ---------------------- 新增：改进方案参数配置 ----------------------
        self.alpha = 1.0  # 距离代价（C_goal）权重
        self.beta = 10.0  # 安全代价（C_obs）权重（增大以强化远离障碍物）
        self.gamma = 0.5  # 连接代价（C_tree）权重
        self.eps = 1e-6  # 避免分母为0的极小值
        self.rho0 = 30.0  # 障碍物影响半径（超过此距离则安全代价为0）
        self.d_start_goal = self.dist((self.start.x, self.start.y), (self.end.x, self.end.y))  # 起点到目标点直线距离
        self.d_th = 0.3 * self.d_start_goal  # 探索期/收敛期划分阈值（30%总距离）
        self.N_max = 10  # 最大采样个数（探索期用）
        self.N_min = 5  # 最小采样个数（收敛期用）
        self.p_goal = 0.3  # 直接采样目标点的概率
        self.safe_dist = 15.0  # 新节点与障碍物的最小安全距离（强制远离）

    def rand_point(self):
        """生成随机采样点（地图范围内）"""
        x = random() * self.width
        y = random() * self.height
        return point(x, y)

    def nearest_neighbor(self, tree, target_point):
        """在树中寻找距离目标点最近的节点"""
        nearest_node = None
        min_distance = float('inf')
        for node in tree:
            distance = self.dist((node.x, node.y), target_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node

    def dist(self, p1, p2):
        """计算两点间欧氏距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ---------------------- 重写：完善多维度成本函数（C_goal + C_obs + C_tree） ----------------------
    def compute_cost(self, q, tree):
        """
        计算采样点q的总成本：
        C(q) = α*C_goal + β*C_obs + γ*C_tree
        成本越低，采样点质量越高
        """
        # 1. 距离代价 C_goal：采样点到目标点的距离
        goal_cost = self.dist((q.x, q.y), (self.end.x, self.end.y))

        # 2. 安全代价 C_obs：采样点到障碍物的距离（远离障碍物代价低）
        obs_dist = self.dist_to_nearest_obstacle(q)
        # 超过障碍物影响半径ρ0 → 安全代价为0；否则按反比例计算（加eps避免分母为0）
        if obs_dist >= self.rho0:
            obstacle_cost = 0.0
        else:
            # 若距离小于最小安全距离 → 成本惩罚（强制远离）
            if obs_dist < self.safe_dist:
                obstacle_cost = 1e6  # 极大值过滤危险点
            else:
                obstacle_cost = 1.0 / (obs_dist + self.eps)

        # 3. 连接代价 C_tree：采样点到现有树节点的最小距离（靠近树代价低）
        if not tree:  # 树为空（仅初始节点时）
            tree_cost = 0.0
        else:
            min_tree_dist = min([self.dist((q.x, q.y), (node.x, node.y)) for node in tree])
            tree_cost = min_tree_dist

        # 总成本计算（乘权重）
        total_cost = self.alpha * goal_cost + self.beta * obstacle_cost + self.gamma * tree_cost
        return total_cost

    # ---------------------- 优化：障碍物距离计算（保留原逻辑，适配成本函数） ----------------------
    def dist_to_nearest_obstacle(self, q):
        """计算采样点q到最近障碍物的距离"""
        min_dist = float('inf')
        q_point = Point(q.x, q.y)
        for poly in self.static_polygons:
            nearest = nearest_points(q_point, poly)[1]  # 障碍物上最近点
            dist = q_point.distance(nearest)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def steer(self, nearest_node, target_point, max_distance):
        """从最近节点朝目标点扩展，生成新节点（不超过最大步长）"""
        distance = self.dist((nearest_node.x, nearest_node.y), target_point)
        if distance <= max_distance:
            new_node = point(target_point[0], target_point[1])
        else:
            direction = self.normalize(target_point[0] - nearest_node.x, target_point[1] - nearest_node.y)
            new_x = nearest_node.x + direction[0] * max_distance
            new_y = nearest_node.y + direction[1] * max_distance
            new_node = point(new_x, new_y)

        # ---------------------- 新增：新节点障碍物安全检查（强制远离） ----------------------
        new_node_obs_dist = self.dist_to_nearest_obstacle(new_node)
        if new_node_obs_dist < self.safe_dist:
            return None  # 新节点过近障碍物 → 直接返回无效
        return new_node

    def normalize(self, vx, vy):
        """向量标准化（单位向量）"""
        norm = math.sqrt(vx * vx + vy * vy)
        if norm > 1e-6:
            normalized_vx = vx / norm
            normalized_vy = vy / norm
        else:
            normalized_vx = 0
            normalized_vy = 0
        return normalized_vx, normalized_vy

    def is_goal_reached(self, node, goal_point, tolerance=45):
        """检查节点是否到达目标区域（容差范围内）"""
        distance_to_goal = self.dist((node.x, node.y), goal_point)
        return distance_to_goal <= tolerance

    # ---------------------- 重写：扩展逻辑（适配动态采样+成本引导） ----------------------
    def expand(self, tree, max_distance, plan_surface=None):
        self.fail_count += 1
        # 连续失败5次 → 强制随机采样（破局）
        if self.fail_count >= 5:
            q_rand = self.rand_point()
            candidates = [q_rand]  # 仅一个候选点
            self.fail_count = 0
        else:
            # 计算动态采样个数N（原有逻辑保留）
            d_min = min([self.dist((node.x, node.y), (self.end.x, self.end.y)) for node in tree])
            if d_min > self.d_th:
                N = self.N_max
                self.gamma = 0.1
            else:
                N = self.N_min + (self.N_max - self.N_min) * (d_min / self.d_th)
                N = int(max(N, self.N_min))
                self.gamma = 0.8

            # 生成候选点并按成本排序（核心改动：缓存排序后的候选列表）
            if random() < self.p_goal:
                # 目标偏向采样：将目标点加入候选列表
                candidates = [self.end] + [self.rand_point() for _ in range(N - 1)]
            else:
                candidates = [self.rand_point() for _ in range(N)]

            # 计算成本并按升序排序（最优在前）
            candidates_with_cost = [(q, self.compute_cost(q, tree)) for q in candidates]
            candidates_with_cost.sort(key=lambda x: x[1])  # 按成本从小到大排序
            candidates = [q for q, _ in candidates_with_cost]  # 提取排序后的候选点

            # 绘制候选点（调试用）
            if plan_surface is not None:
                for i, c in enumerate(candidates):
                    # 最优候选点标为绿色，其余为黄色
                    color = (0, 255, 0) if i == 0 else (200, 200, 0)
                    pygame.draw.circle(plan_surface, color, (int(c.x), int(c.y)), 2)
                QApplication.processEvents()

        # ---------------------- 核心改动：遍历候选点重试，直到成功或用尽 ----------------------
        for q_rand in candidates:  # 按成本从低到高尝试每个候选点
            nearest_node = self.nearest_neighbor(tree, (q_rand.x, q_rand.y))
            new_node = self.steer(nearest_node, (q_rand.x, q_rand.y), max_distance)
            if new_node is None:
                continue  # 新节点过近障碍物，试下一个候选点

            new_node.father = nearest_node
            # 检查碰撞
            if not self.collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
                tree.append(new_node)
                self.fail_count = 0  # 成功，重置失败计数
                return new_node  # 找到有效节点，直接返回

        # 所有候选点都失败 → 才返回空
        return None

    def collision(self, src, dst):
        """检查线段（src→dst）是否与障碍物碰撞（逐点采样检测）"""
        vx, vy = self.normalize(dst[0] - src[0], dst[1] - src[1])
        curr = list(src)
        # 沿线段逐点检测（步长1像素）
        while self.dist(curr, dst) > 1:
            int_curr = (int(round(curr[0])), int(round(curr[1])))  # 像素坐标（四舍五入避免越界）
            # 检查坐标是否在地图范围内（避免数组越界）
            if 0 <= int_curr[0] < self.width and 0 <= int_curr[1] < self.height:
                if self.obstacle.get_at(int_curr) == (0, 0, 0):  # 黑色为障碍物
                    return True
            # 移动到下一个点
            curr[0] += vx
            curr[1] += vy
        return False

    def plan(self, plan_surface):
        """主规划函数：执行改进的RRT算法，返回路径和耗时"""
        start_time = time.time()
        tree = [self.start]  # 初始化树（起点为根节点）
        max_distance = self.step

        for i in range(self.max_iterations):
            # 1. 扩展树（动态采样+成本引导）
            new_node = self.expand(tree, max_distance, plan_surface)
            if new_node is None:
                continue  # 扩展失败 → 跳过当前迭代
            self.node_count += 1  # 节点计数+1

            # 2. 检查是否到达目标
            if self.is_goal_reached(new_node, (self.end.x, self.end.y)):
                # 回溯路径（从目标节点到起点）
                path = [new_node]
                current_node = new_node
                # 绘制目标节点与终点的连线
                pygame.draw.circle(plan_surface, (0, 255, 0), (int(current_node.x), int(current_node.y)), 4)
                pygame.draw.line(plan_surface, (255, 0, 0), (current_node.x, current_node.y),
                                 (self.end.x, self.end.y), 3)
                QApplication.processEvents()

                # 回溯父节点，构建完整路径
                while current_node != self.start:
                    parent = current_node.father
                    # 绘制路径线段（紫红色）
                    pygame.draw.line(plan_surface, (128, 0, 128), (parent.x, parent.y),
                                     (current_node.x, current_node.y), 3)
                    QApplication.processEvents()
                    path.append(parent)
                    current_node = parent

                path.reverse()  # 反转路径（起点→目标）
                end_time = time.time()
                cost_time = end_time - start_time
                # 打印日志
                print(f"路径找到！耗时：{cost_time:.2f}s，总迭代次数：{i + 1}，路径节点数：{len(path)}")
                return path, cost_time

            # 3. 绘制扩展的节点和树（浅蓝色）
            if new_node:
                pygame.draw.circle(plan_surface, (0, 100, 255), (int(new_node.x), int(new_node.y)), 2)
                pygame.draw.line(plan_surface, (0, 100, 255), (new_node.father.x, new_node.father.y),
                                 (new_node.x, new_node.y), 2)
                QApplication.processEvents()

        # 最大迭代次数未找到路径
        print("未找到路径！（已达最大迭代次数）")
        return [], time.time() - start_time