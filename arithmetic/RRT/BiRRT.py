import copy
import math
import time
from random import random
import pygame
from PyQt5.QtWidgets import QApplication

from .Node import point


class BiRrt:
    def __init__(self, mapdata):
        self.node_count = 0
        self.all_node = None
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])
        self.result = []
        self.width = mapdata.width
        self.obstacle = mapdata.obs_surface
        self.height = mapdata.height
        self.step = 10
        self.max_iterations = 10000
        self.collision_distance_threshold = 5


    def rand_point(self):
        x = random() * self.width
        y = random() * self.height
        return point(x, y)

    def nearest_neighbor(self, tree, target_point):
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

    def steer(self, nearest_node, target_point, max_distance):
        distance = self.dist((nearest_node.x, nearest_node.y), target_point)
        if distance <= max_distance:
            return point(target_point[0], target_point[1])
        else:
            direction = self.normalize(target_point[0] - nearest_node.x, target_point[1] - nearest_node.y)
            new_x = nearest_node.x + direction[0] * max_distance
            new_y = nearest_node.y + direction[1] * max_distance
            return point(new_x, new_y)

    def normalize(self, vx, vy):
        norm = math.sqrt(vx * vx + vy * vy)
        if norm > 1e-6:
            normalized_vx = vx / norm
            normalized_vy = vy / norm
        else:
            normalized_vx = 0
            normalized_vy = 0
        return normalized_vx, normalized_vy

    def is_goal_reached(self, node, goal_point, tolerance):
        distance_to_goal = self.dist((node.x, node.y), goal_point)
        return distance_to_goal <= tolerance

    def expand(self, tree, max_distance):
        random_point = self.rand_point()
        nearest_node = self.nearest_neighbor(tree, (random_point.x, random_point.y))
        new_node = self.steer(nearest_node, (random_point.x, random_point.y), max_distance)
        new_node.father = nearest_node
        if not self.fuzzy_collision(new_node):
            if not self.collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
                if new_node not in self.all_node:
                    tree.append(new_node)
                    return new_node
        return None

    def fuzzy_collision(self, node):
        """
        快速碰撞检测：检查节点是否在障碍物附近。
        """
        int_x, int_y = int(node.x), int(node.y)
        for dx in range(-self.collision_distance_threshold, self.collision_distance_threshold + 1):
            for dy in range(-self.collision_distance_threshold, self.collision_distance_threshold + 1):
                if 0 <= int_x + dx < self.width and 0 <= int_y + dy < self.height:
                    if self.obstacle.get_at((int_x + dx, int_y + dy)) == (0, 0, 0):
                        return True
        return False
    def collision(self, src, dst):
        vx, vy = self.normalize(dst[0] - src[0], dst[1] - src[1])
        curr = list(src)
        while self.dist(curr, dst) > 1:
            int_curr = int(curr[0]), int(curr[1])
            if self.obstacle.get_at(int_curr) == (0, 0, 0):
                return True
            curr[0] += vx * 0.5
            curr[1] += vy * 0.5
        return False

    def plan(self, plan_surface):
        start_time = time.time()
        tree_start = [self.start]
        tree_end = [self.end]
        max_iterations = self.max_iterations
        max_distance = self.step
        self.all_node = set(tree_start + tree_end)  # 记录所有生成的节点
        for i in range(max_iterations):
            self.node_count+=2
            if i % 2 == 0:
                new_node_start = self.expand(tree_start, max_distance)
                if new_node_start:
                    #扩展终点树
                    nearest_node_end = self.nearest_neighbor(tree_end, (new_node_start.x, new_node_start.y))
                    new_node_end = self.steer(nearest_node_end, (new_node_start.x, new_node_start.y), max_distance)
                    new_node_end.father = copy.deepcopy(nearest_node_end)
                    if not self.collision((nearest_node_end.x, nearest_node_end.y), (new_node_end.x, new_node_end.y)):
                        if new_node_end not in self.all_node:
                            tree_end.append(new_node_end)
                            self.all_node.add(new_node_end)
                        if self.is_goal_reached(new_node_end, (new_node_start.x, new_node_start.y), max_distance):
                            if new_node_start == new_node_end:
                                new_node_end = self.create_nearby_node(new_node_end)
                            path = self.connect_trees(tree_start, tree_end, new_node_start, new_node_end, plan_surface)
                            pygame.draw.line(plan_surface, (100, 0, 100),
                                             (new_node_start.x, new_node_start.y),
                                             (new_node_end.x, new_node_end.y), 2)
                            end_time = time.time()

                            print("花费时间为", end_time - start_time)
                            print("总迭代次数为")
                            print(i)
                            return path, end_time - start_time
            else:
                new_node_end = self.expand(tree_end, max_distance)
                if new_node_end:
                    #扩展起点树
                    nearest_node_start = self.nearest_neighbor(tree_start, (new_node_end.x, new_node_end.y))
                    new_node_start = self.steer(nearest_node_start, (new_node_end.x, new_node_end.y), max_distance)
                    new_node_start.father = copy.deepcopy(nearest_node_start)
                    if not self.collision((nearest_node_start.x, nearest_node_start.y),
                                          (new_node_start.x, new_node_start.y)):
                        if new_node_start not in self.all_node:
                            tree_start.append(new_node_start)
                            self.all_node.add(new_node_start)
                        if self.is_goal_reached(new_node_start, (new_node_end.x, new_node_end.y), max_distance):
                            if new_node_start == new_node_end:
                                new_node_start = self.create_nearby_node(new_node_start)

                            path = self.connect_trees(tree_start, tree_end, new_node_start, new_node_end, plan_surface)

                            end_time = time.time()
                            print("花费时间为", end_time - start_time)
                            print("总迭代次数为")
                            print(i)
                            #print(path)
                            return path, end_time - start_time

            if new_node_start:
                pygame.draw.circle(plan_surface, (0, 100, 255), (new_node_start.x, new_node_start.y), 2)
                pygame.draw.line(plan_surface, (0, 100, 255), (new_node_start.father.x, new_node_start.father.y),
                                 (new_node_start.x, new_node_start.y), 2)
            if new_node_end:
                pygame.draw.circle(plan_surface, (0, 100, 255), (new_node_end.x, new_node_end.y), 2)
                pygame.draw.line(plan_surface, (0, 100, 255), (new_node_end.father.x, new_node_end.father.y),
                                 (new_node_end.x, new_node_end.y), 2)
            QApplication.processEvents()

        print("未找到路径！！！")
        return []

    def connect_trees(self, tree_start, tree_end, new_node_start, new_node_end, plan_surface):
        path = []
        pygame.draw.line(plan_surface, (100, 255, 100),
                         (new_node_start.x, new_node_start.y),
                         (new_node_end.x, new_node_end.y), 2)
        # 从起点树回溯路径
        current_node = new_node_start
        while current_node:
            path.append(current_node)
            current_node = current_node.father
        path.reverse()

        # 从终点树回溯路径
        current_node = new_node_end
        while current_node:
            path.append(current_node)
            current_node = current_node.father

        return path

    def create_nearby_node(self, node):
        nearby_node = point(node.x + 1, node.y + 1)  # 生成一个与原节点稍微偏移的新节点
        nearby_node.father = node.father
        return nearby_node