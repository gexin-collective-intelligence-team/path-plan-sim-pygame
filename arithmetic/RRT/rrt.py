import math
import time
from random import random
import pygame
from PyQt5.QtWidgets import QApplication

from .Node import point


class Rrt:
    def __init__(self, mapdata):
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])
        self.result = []
        self.width = mapdata.width
        self.obstacle = mapdata.obs_surface
        self.height = mapdata.height
        self.tree = []
        self.step = 15
        self.max_iterations = 10000
        self.node_count = 0

    def rand_point(self):
        x = random() * self.width
        y = random() * self.height
        return point(x, y)

    def nearest_neighbor(self, tree, target_point):
        """
        在树中寻找最接近目标点的节点。

    Args:
    - tree: 表示树中节点的点列表。
    - target_point: 要查找最近邻节点的目标点 (x, y)。

    Returns:
    - nearest_node: 树中距离目标点最近的节点。
        """
        nearest_node = None
        min_distance = float('inf')
        for node in tree:
            distance = self.dist((node.x, node.y), target_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node

    def dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def path_length(self, path):
        length = 0
        for i in range(1, len(path)):
            length += self.dist((path[i - 1].x, path[i - 1].y), (path[i].x, path[i].y))
        return length

    def steer(self, nearest_node, target_point, max_distance):
        """
        从最近节点朝目标点直线扩展，直到最大距离。

        Args:
        - nearest_node: 最近节点到目标点。
        - target_point: 要朝向的目标点 (x, y)。
        - max_distance: 最大扩展距离。

        Returns:
        - new_node: 从最近节点到达的新节点。
        """
        distance = self.dist((nearest_node.x, nearest_node.y), target_point)
        if distance <= max_distance:
            return point(target_point[0], target_point[1])
        else:
            direction = self.normalize(target_point[0] - nearest_node.x, target_point[1] - nearest_node.y)
            new_x = nearest_node.x + direction[0] * max_distance
            new_y = nearest_node.y + direction[1] * max_distance
            return point(new_x, new_y)

    def normalize(self, vx, vy):
        """
        将输入向量标准化，并返回其坐标。

        Args:
        - vx: 向量的 x 坐标分量。
        - vy: 向量的 y 坐标分量。

        Returns:
        - normalized_vx: 标准化后的 x 坐标分量。
        - normalized_vy: 标准化后的 y 坐标分量。
        """
        norm = math.sqrt(vx * vx + vy * vy)
        if norm > 1e-6:
            normalized_vx = vx / norm
            normalized_vy = vy / norm
        else:
            normalized_vx = 0
            normalized_vy = 0
        return normalized_vx, normalized_vy

    def is_goal_reached(self, node, goal_point, tolerance):
        """
        检查当前节点是否已经接近目标点。

        Args:
        - node: 要检查的当前节点 (点)。
        - goal_point: 要检查的目标点 (x, y)。
        - tolerance: 用于判断是否达到目标的容差距离。

        Returns:
        - reached: 如果达到目标，则为True，否则为False。
        """
        distance_to_goal = self.dist((node.x, node.y), goal_point)
        return distance_to_goal <= tolerance

    def expand(self, tree, max_distance):
        """
        扩展 RRT 树，添加一个新节点。

        Args:
        - tree: 表示树中节点的点列表。
        - obstacles: 障碍物列表（如 pygame surfaces）。
        - max_distance: 从最近节点扩展的最大距离。

        Returns:
        - new_node: 添加到树中的新节点，如果扩展失败则返回None。
        """
        # 1. 随机抽样一个点
        random_point = self.rand_point()

        # 2. 找到树中距离随机点最近的节点
        nearest_node = self.nearest_neighbor(tree, (random_point.x, random_point.y))

        # 3. 从最近节点向随机点扩展
        new_node = self.steer(nearest_node, (random_point.x, random_point.y), max_distance)
        new_node.father = nearest_node
        # 4. 检查是否与障碍物发生碰撞
        if not self.collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
            # 如果没有碰撞，则将新节点添加到树中
            tree.append(new_node)
            return new_node
        else:
            # 如果发生碰撞，则返回 None 表示扩展失败
            return None

    def collision(self, src, dst):
        """
            检查从源点到目标点的线段是否与障碍物发生碰撞。

            Args:
            - src: 源点的坐标 (x1, y1)。
            - dst: 目标点的坐标 (x2, y2)。

            Returns:
            - collided: 如果发生碰撞，则为 True，否则为 False。
            """
        vx, vy = self.normalize(dst[0] - src[0], dst[1] - src[1])
        curr = list(src)
        while self.dist(curr, dst) > 1:
            int_curr = int(curr[0]), int(curr[1])
            if self.obstacle.get_at(int_curr) == (0, 0, 0):
                return True
            curr[0] += vx
            curr[1] += vy
        return False

    def plan(self, plan_surface):
        """
        执行 RRT 算法，寻找从起点到目标点的路径。

        Args:
        - obstacles: 障碍物列表（如 pygame surfaces）。
        - max_iterations: 运行 RRT 算法的最大迭代次数。
        - max_distance: 从最近节点扩展的最大距离。

        Returns:
        - path: 表示 RRT 算法找到的路径的点列表，如果找不到路径则返回空列表。
        """
        start = time.time()
        # 使用起始节点初始化树
        tree = [self.start]
        max_iterations = self.max_iterations
        max_distance = self.step
        # 执行 RRT 迭代
        for i in range(max_iterations):
            # 1. 扩展树，添加一个新节点
            new_node = self.expand(tree, max_distance)
            self.node_count+=1
            # 2. 检查是否达到目标点
            if new_node and self.is_goal_reached(new_node, (self.end.x, self.end.y), 45):
                # 如果达到目标，则构建并返回路径
                path = [new_node]
                current_node = new_node
                print(new_node)

                pygame.draw.circle(plan_surface, (0, 100, 255), (current_node.x, current_node.y), 2)
                pygame.draw.line(plan_surface, (100, 0, 100), (current_node.x, current_node.y),
                                 (self.end.x, self.end.y), 4)
                QApplication.processEvents()

                while current_node != self.start:
                    # 最终节点回溯整条路径
                    parent = current_node.father

                    pygame.draw.line(plan_surface, (100, 0, 100), (parent.x, parent.y),
                                     (current_node.x, current_node.y), 4)
                    QApplication.processEvents()

                    path.append(parent)
                    current_node = parent

                path.reverse()
                end = time.time()
                print("花费时间为")
                print(end - start)
                print("总迭代次数为")
                print(i)
                print("路径总长度为")
                print(self.path_length(path))
                return path, end - start

            # 绘制过程中的点和线
            if new_node:
                pygame.draw.circle(plan_surface, (0, 100, 255), (new_node.x, new_node.y), 2)
                pygame.draw.line(plan_surface, (0, 100, 255), (new_node.father.x, new_node.father.y),
                                 (new_node.x, new_node.y), 2)
                QApplication.processEvents()  # 强制Qt处理事件队列（重绘）
        # 如果最大迭代次数后仍未找到路径，则返回空路径
        print("未找到路径！！！")
        return []
