import math
import time
from random import random
import pygame
from PyQt5.QtWidgets import QApplication

from .Node import point

class RrtStar:
    def __init__(self, mapdata):
        self.node_count = 0
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])
        self.result = []
        self.width = mapdata.width
        self.obstacle = mapdata.obs_surface
        self.height = mapdata.height
        self.tree = []
        self.step = 15
        self.max_iterations = 10000
        self.radius = 20  # 搜索邻域半径

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
        if not self.collision((nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
            #tree.append(new_node)
            return new_node
        else:
            return None

    def collision(self, src, dst):
        vx, vy = self.normalize(dst[0] - src[0], dst[1] - src[1])
        curr = list(src)
        while self.dist(curr, dst) > 1:
            int_curr = int(curr[0]), int(curr[1])
            if self.obstacle.get_at(int_curr) == (0, 0, 0):
                return True
            curr[0] += vx
            curr[1] += vy
        return False

    def find_nearby_nodes(self, tree, new_node, radius):
        nearby_nodes = []
        for node in tree:
            if self.dist((node.x, node.y), (new_node.x, new_node.y)) <= radius:
                nearby_nodes.append(node)
        return nearby_nodes

    def choose_best_parent(self, tree, new_node, nearby_nodes):
        best_node = None
        min_cost = float('inf')
        for node in nearby_nodes:
            if not self.collision((node.x, node.y), (new_node.x, new_node.y)):
                cost = self.compute_cost(node) + self.dist((node.x, node.y), (new_node.x, new_node.y))
                if cost < min_cost:
                    min_cost = cost
                    best_node = node

        return best_node, min_cost

    def compute_cost(self, node):
        cost = 0
        current_node = node
        while current_node.father:
            cost += self.dist((current_node.x, current_node.y), (current_node.father.x, current_node.father.y))
            current_node = current_node.father
        return cost

    def rewire(self, tree, new_node, nearby_nodes, min_cost,plan_surface):
        for node in nearby_nodes:
            if node != new_node.father:
                potential_cost = min_cost + self.dist((new_node.x, new_node.y), (node.x, node.y))
                if potential_cost < self.compute_cost(node):
                    if not self.collision((new_node.x, new_node.y), (node.x, node.y)):
                        node.father = new_node
                        # 确保连接后更新
                        pygame.draw.line(plan_surface, (100, 0, 100), (node.x, node.y),
                                         (new_node.x, new_node.y), 4)
                        QApplication.processEvents()

    def plan(self, plan_surface):
        start = time.time()
        tree = [self.start]
        max_iterations = self.max_iterations
        max_distance = self.step
        for i in range(max_iterations):
            new_node = self.expand(tree, max_distance)
            self.node_count+=1
            if new_node:
                nearby_nodes = self.find_nearby_nodes(tree, new_node, self.radius)
                best_node, min_cost = self.choose_best_parent(tree, new_node, nearby_nodes)
                if best_node:
                    new_node.father = best_node
                    tree.append(new_node)
                    self.rewire(tree, new_node, nearby_nodes, min_cost,plan_surface)
                if self.is_goal_reached(new_node, (self.end.x, self.end.y), 45):
                    path= [self.end, new_node]
                    current_node = new_node
                    pygame.draw.circle(plan_surface, (0, 100, 255), (current_node.x, current_node.y), 2)
                    pygame.draw.line(plan_surface, (100, 0, 100), (current_node.x, current_node.y),
                                     (self.end.x, self.end.y), 4)
                    QApplication.processEvents()
                    while current_node != self.start:
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
                    return path, end - start
                    # 绘图更新每隔100次迭代执行一次

                pygame.draw.circle(plan_surface, (0, 100, 255), (new_node.x, new_node.y), 2)
                pygame.draw.line(plan_surface, (0, 100, 255), (new_node.father.x, new_node.father.y),
                                     (new_node.x, new_node.y), 2)
                QApplication.processEvents()
        print("未找到路径！！！")
        return []