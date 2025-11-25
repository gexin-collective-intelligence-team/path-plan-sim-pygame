import math
import re
import sys
import time
#from noise import pnoise2
import numpy as np
import pygame
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog
from PyQt5.QtWidgets import QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import QTimer
import random
import json
import geojson
import shapely.geometry
# from collections import defaultdict
# import geopandas as gpd
from geojson import Point, Feature, FeatureCollection
from collections import defaultdict
import geopandas as gpd
from shapely import Polygon
from shapely.geometry import Point, MultiPoint, shape
import cv2
import numpy
from scipy.interpolate import splprep, splev

from DynamicObstacle import DynamicObstacle
from arithmetic.APF.apf import apf
from arithmetic.APFRRT.APFRRT_dyn import APFRRT_dyn
from arithmetic.APFRRT.dbvsAPFRRT import dbvsAPFRRT_dyn
from arithmetic.Astar.Map import Map
from arithmetic.Astar.astar import astar
from arithmetic.RRT.BiRRT import BiRrt
from arithmetic.RRT.RRTstar import RrtStar
from arithmetic.RRT.costRRT import Cost_Rrt
from arithmetic.RRT.rrt import Rrt
from local_planner.path_optimizer import PathOptimizer

from result import Result_Demo

from arithmetic.RRT.rrt import Rrt
from arithmetic.APFRRT.APFRRT import APFRRT

#from arithmetic.PRM.prm import prm
from usv.motion_simulator import USVMotionSimulator
# from arithmetic.PRM.prm import prm
from arithmetic.PRM.Aprm import prm

# 导入局部路径规划模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_planner.local_planner_manager import LocalPlannerManager
from local_planner.algorithms.apf_local_planner import APFLocalPlanner
from local_planner.algorithms.dwa_local_planner import DWALocalPlanner

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED=(255,0,0)


def batch_run(self, algo: str = "PRM", runs: int = 50, out_dir: str = "runs",
              seed: int = None, zh: bool = True, prefix: str = None) -> str:
    """
    批量循环运行算法，并保存每次运行的完整指标（含复杂度）+ 汇总CSV + 指标直方图。
    参数：
      - algo: "PRM" / "APF" / "APF-RRT" / "APF-RRT-dyn" / "DbvsAPF-RRT-dyn" / "RRT" / "RRT*" / "BiRRT" / "CostRRT"
      - runs: 运行次数
      - out_dir: 结果输出目录
      - seed: 随机种子起点（若不为 None，则每次运行使用 seed+i）
      - zh: 结果JSON是否用中文键名（CSV固定英文便于分析）
      - prefix: 输出文件前缀（默认用 algo）
    返回：输出目录路径
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix or algo

    # 选择对应的起算函数
    algo_map = {
        "PRM": self.startPRm,
        "APF": self.startApf,
        "APF-RRT": self.startApfRrt,
        "APF-RRT-dyn": self.startApfRrt_dyn,
        "DbvsAPF-RRT-dyn": self.startDbvsPRrt,
        "RRT": self.startRtt,
        "RRT*": self.startRRTStar,
        "BiRRT": self.startBiRRT,
        "CostRRT": self.startcostRrt,
    }
    if algo not in algo_map:
        raise ValueError(f"不支持的算法标识：{algo}")

    start_func = algo_map[algo]

    # 汇总（英文键，便于CSV分析）
    summary_rows = []

    for i in range(runs):
        if seed is not None:
            import random
            random.seed(seed + i)
            np.random.seed(seed + i)

        # 清理规划层
        self.plan_surface.fill(self.back_color)

        # 运行一次算法
        try:
            ret = start_func()
            # 不同算法返回格式略有差异：大多数是 (track, time)
            if isinstance(ret, tuple) and len(ret) == 2:
                track, elapsed = ret
            else:
                # 回退：若算法只画到 self.result，则统一抽取
                track = []
                if self.result:
                    for pt in self.result[:-1]:
                        if hasattr(pt, 'x') and hasattr(pt, 'y'):
                            track.append((pt.x, pt.y))
                        elif isinstance(pt, (tuple, list)) and len(pt) >= 2:
                            track.append((pt[0], pt[1]))
                elapsed = None
        except Exception as e:
            # 失败也要写日志
            track, elapsed = [], None
            if hasattr(self, 'main_window'):
                self.main_window.printf(f"[{algo}] 第{i}次运行异常：{e}", None, None)

        # 生成指标（含复杂度）
        r = Result_Demo(
            self.start_point, self.end_point, elapsed,
            self.obstacles, self.dynamic_obstacles, track,
            obs_surface=self.obs_surface
        )

        # 逐次 JSON
        per_run = r.to_metrics_dict(zh=zh)
        # 附加运行元信息
        per_run['run_id'] = i if not zh else i
        per_run['seed'] = (seed + i) if seed is not None else None
        per_run['algo'] = algo if not zh else algo
        json_path = os.path.join(out_dir, f"{prefix}_run_{i:03d}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(per_run, f, ensure_ascii=not zh, indent=2)

        # 汇总行为英文键
        en = r.to_metrics_dict(zh=False)
        en['run_id'] = i
        en['seed'] = (seed + i) if seed is not None else None
        en['algo'] = algo
        summary_rows.append(en)

    # 写 CSV
    try:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(out_dir, f"{prefix}_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 生成简单的指标图（直方图）
        import matplotlib.pyplot as plt
        for col in ['time', 'pathlen', 'smoothness', 'straightness']:
            if col in df.columns and df[col].notna().any():
                plt.figure()
                df[col].dropna().hist(bins=20)
                plt.title(f'{algo} {col} distribution (n={runs})')
                plt.xlabel(col);
                plt.ylabel('count')
                plt.tight_layout()
                fig_path = os.path.join(out_dir, f"{prefix}_{col}_hist.png")
                plt.savefig(fig_path, dpi=180)
                plt.close()
    except Exception as e:
        if hasattr(self, 'main_window'):
            self.main_window.printf(f"汇总/绘图失败：{e}", None, None)

    if hasattr(self, 'main_window'):
        self.main_window.printf(f"批量运行完成，输出目录：{out_dir}", None, None)
    return out_dir


def load_map_file(self, file_path: str) -> bool:
    """
    直接从文件路径加载地图（无需弹窗）。
    支持你保存的 GeoJSON（FeatureCollection）格式：起点/终点/Polygon障碍物/动态障碍物。
    """
    if not os.path.isfile(file_path):
        if hasattr(self, 'main_window'): self.main_window.printf(f"地图不存在：{file_path}", None, None)
        return False
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_str = f.read()
        geojson_obj = geojson.loads(geojson_str)

        obstacles = []
        dynamic_obstacles = []
        start_point = None
        end_point = None

        for feature in geojson_obj['features']:
            geometry = feature['geometry']
            properties = feature.get('properties', {})
            if geometry['type'] == 'Point' and properties.get('name') == "起始点":
                start_point = geometry['coordinates']
            elif geometry['type'] == 'Point' and properties.get('name') == "终点":
                end_point = geometry['coordinates']
            elif geometry['type'] == 'Polygon':
                obstacles.append(tuple(geometry['coordinates'][0]))
            elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
                from DynamicObstacle import DynamicObstacle
                shape = properties.get('shape', '正方形')
                position = tuple(geometry['coordinates'])
                direction = tuple(properties.get('direction', (1, 0)))
                speed = properties.get('speed', 1.0)
                size = properties.get('size', 20.0)
                bounce = properties.get('bounce', True)  # 默认反弹，兼容旧文件
                dynamic_obstacles.append(DynamicObstacle(shape, position, direction, speed, size, bounce))

        # 写入当前 widget
        self.obstacles = obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.start_point = start_point
        self.end_point = end_point

        # 把障碍物画到 obs_surface 上
        self.update_obs_surface()
        self.update()  # 刷新
        if hasattr(self, 'main_window'): self.main_window.printf("地图加载完成", None, None)
        return True
    except Exception as e:
        if hasattr(self, 'main_window'): self.main_window.printf(f"加载失败：{e}", None, None)
        return False


def surface_to_cv_bgr(surface: pygame.Surface) -> cv2.typing.MatLike:
    """
    Converts pygame surface pixel data into opencv BGR format Mat
    """

    surface_string = pygame.image.tostring(surface, 'RGB')

    # convert from (width, height, channel) to (height, width, channel)
    size = surface.get_size()
    array = numpy.frombuffer(surface_string, dtype=numpy.uint8).reshape((size[1], size[0], 3))

    img_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    return img_bgr


def cv_bgr_to_surface(img_bgr: cv2.typing.MatLike) -> pygame.Surface:
    """
    Converts pygame surface pixel data into opencv BGR format Mat
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    surface = pygame.image.frombuffer(img_rgb.tobytes(), (width, height), 'RGB')

    return surface


def draw_line(surface, color, start_pos, end_pos, radius):
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = max(abs(dx), abs(dy))

    for i in range(distance):
        x = int(start_pos[0] + float(i) / distance * dx)
        y = int(start_pos[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, color, (x, y), radius)


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
        self.max_speed = 25.0  # 最大速度（适当调高，提高航行速度）
        self.turn_rate = 0.1  # 转向率
        self.ship_radius = 8  # 船舶半径
        self.arrival_threshold = 2.0  # 到达目标点的阈值，设置为较小的值使船舶更接近目标点

        # 碰撞状态标记
        self.has_collided = False
        self.collision_position = None

        # 历史轨迹记录
        self.history_path = []  # 记录完整的航行轨迹
        self.journey_completed = False  # 航行完成标记

        # 局部路径规划管理器
        self.local_planner_manager = LocalPlannerManager()
        
        # 注册可用的局部算法
        self.local_planner_manager.register_algorithm("APF算法", APFLocalPlanner)
        self.local_planner_manager.register_algorithm("DWA算法", DWALocalPlanner)
        self.local_planner_manager.register_algorithm("人工势场法", APFLocalPlanner)
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
        # 基础参数设置
        SAFE_DISTANCE_STATIC = 35
        SAFE_DISTANCE_DYNAMIC = 30
        SAFE_DISTANCE_BOUNDARY = 25

        # 计算距离，处理可能的除零错误
        total_distance = max(np.linalg.norm(end - start), 1.0)  # 防止除零

        # 根据距离调整参数（简化版本，避免复杂条件）
        if total_distance > 200:
            K_att = 0.6
            K_rep_static = 2.8
            K_rep_dynamic = 2.2
            K_rep_boundary = 1.8
            SAFE_DISTANCE_STATIC = 40
            SAFE_DISTANCE_DYNAMIC = 35
            SAFE_DISTANCE_BOUNDARY = 30
        elif total_distance < 100:
            K_att = 1.0
            K_rep_static = 2.2
            K_rep_dynamic = 1.8
            K_rep_boundary = 1.3
            SAFE_DISTANCE_STATIC = 30
            SAFE_DISTANCE_DYNAMIC = 25
            SAFE_DISTANCE_BOUNDARY = 20
        else:
            K_att = 0.8
            K_rep_static = 2.5
            K_rep_dynamic = 2.0
            K_rep_boundary = 1.5
            SAFE_DISTANCE_STATIC = 35
            SAFE_DISTANCE_DYNAMIC = 30
            SAFE_DISTANCE_BOUNDARY = 25

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
            if self.path_index >= len(self.path) and len(self.path) > 0:
                self.journey_completed = True
                self.is_moving = False
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
        
        # 使用APFLocalPlanner计算下一步位置（实时避障）
        next_pos = self.local_planner_manager.plan_local_path(
            np.array(self.current_pos),
            np.array(target_point),
            static_obstacles,
            dynamic_obstacles,
            self.velocity
        )

        # 更新位置
        self.current_pos = np.array(next_pos)

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
                return

        # 更新速度向量（用于下次计算）
        if len(self.path) > self.path_index:
            next_target = self.path[self.path_index]
            direction = np.array(next_target) - self.current_pos
            if np.linalg.norm(direction) > 0:
                self.velocity = direction / np.linalg.norm(direction) * self.max_speed
            else:
                self.velocity = np.zeros(2)

        # 触发重绘
        self.widget.update()

    def draw_ship(self, surface):
        """绘制船舶"""
        if self.current_pos is None:
            return

        # 绘制优化后的全局路径（无论航行是否完成都显示）
        if len(self.path) >= 2:
            # 绘制优化后的全局路径（浅蓝色点线）
            pygame.draw.lines(surface, (100, 100, 255), False, self.path, 2)

            # 绘制路径点
            for i, point in enumerate(self.path):
                if i == 0:
                    # 路径起点（蓝色圆圈）
                    pygame.draw.circle(surface, (0, 0, 255), (int(point[0]), int(point[1])), 6)
                elif i == len(self.path) - 1:
                    # 路径终点（红色圆圈）
                    pygame.draw.circle(surface, (255, 0, 0), (int(point[0]), int(point[1])), 6)
                else:
                    # 路径中间点（浅蓝色点）
                    pygame.draw.circle(surface, (100, 100, 255), (int(point[0]), int(point[1])), 3)

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
                        pygame.draw.circle(surface, (255, 165, 0), (int(point[0]), int(point[1])), 3)

        # 实时显示无人船速度（标量）
        try:
            speed = float(np.linalg.norm(self.velocity))
            font = pygame.font.Font(None, 22)
            text = f"Speed: {speed:.2f}"
            text_surface = font.render(text, True, (0, 0, 0))
            surface.blit(text_surface, (10, 10))
        except Exception:
            pass

    def get_ship_shape(self, pos, angle):
        """获取船舶形状点"""
        x, y = pos
        size = self.ship_radius
        return [
            (x + size * np.cos(angle), y + size * np.sin(angle)),
            (x + size * np.cos(angle + 2.5), y + size * np.sin(angle + 2.5)),
            (x + size * np.cos(angle - 2.5), y + size * np.sin(angle - 2.5))
        ]
class PygameWidget(QWidget):
    BACK_COLOR = WHITE
    OBS_COLOR = BLACK
    DYN_ObsColor = RED
    OBS_RADIUS = 5
    POINT_RADIUS = 5
    WIDTH = 920
    HEIGHT = 450
    CELL_SIZE = 10

    def __init__(self, main_window, parent=None):
        super(PygameWidget, self).__init__(parent)

        # 初始化pygame

        pygame.init()

        # 设置窗口大小
        self.width, self.height = PygameWidget.WIDTH, PygameWidget.HEIGHT
        self.setMinimumSize(self.width, self.height)

        # 设置背景颜色
        self.back_color = PygameWidget.BACK_COLOR
        # 设置障碍物颜色
        self.obs_color = PygameWidget.OBS_COLOR

        self.grid_color = PygameWidget.OBS_COLOR

        # 设置障碍物画笔半径
        self.obs_radius = PygameWidget.OBS_RADIUS

        self.point_radius = PygameWidget.POINT_RADIUS

        # 设置主平面，障碍物平面，路径规划平面和起点终点平面
        self.surface = pygame.Surface((self.width, self.height))
        self.obs_surface = pygame.Surface((self.width, self.height))
        self.dynamic_surface = pygame.Surface((self.width, self.height))
        self.plan_surface = pygame.Surface((self.width, self.height))
        self.point_surface = pygame.Surface((self.width, self.height))

        self.grid_surface = pygame.Surface((self.width, self.height))

        # 设置障碍物平面和路径规划平面的透明色
        self.obs_surface.set_colorkey(self.back_color)
        self.dynamic_surface.set_colorkey(self.back_color)
        self.plan_surface.set_colorkey(self.back_color)
        self.point_surface.set_colorkey(self.back_color)

        self.grid_surface.set_colorkey(self.back_color)

        self.obs_surface.fill(self.back_color)
        self.dynamic_surface.fill(self.back_color)  # 用全透明颜色填充

        self.plan_surface.fill(self.back_color)
        self.point_surface.fill(self.back_color)

        self.grid_surface.fill(self.back_color)

        # 栅格大小
        self.cell_size = PygameWidget.CELL_SIZE

        # 初始化多边形轮廓存储的障碍物列表，每个障碍物是一个包含其所有顶点的列表
        self.obstacles = []

        #初始化动态障碍物列表
        self.dynamic_obstacles = []

        # 路径规划算法

        # self.map = Map(self.width, self.height, self.cell_size)
        # self.apf = apf(self.map)
        # self.astar = astar(self.map)
        # self.bi_rrt = BiRrt(self.map)
        # self.rrt_star = RrtStar(self.map)
        # self.rrt = Rrt(self.map)
        # self.apf_rrt = APFRRT(self.map)
        # self.prm = prm(self.map)

        # 路径规划结果
        self.result = None

        # 计数器
        self.count = 0

        # 地图列表
        self.cols = self.width // self.cell_size
        self.rows = self.height // self.cell_size

        # 初始化栅格化存储的地图
        # self.map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.grid_map = numpy.zeros((self.cols, self.rows), dtype=numpy.uint8)

        # 起始点和终点
        self.start_point = None
        self.end_point = None

        # 创建一个定时器，用于更新界面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // 60)  # 设置帧率为60

        # 初始化状态
        self.drawing = False
        self.last_pos = None
        self.grid = False

        # 图形障碍物临时配置（由主窗口设置，鼠标点击时使用）
        self.pending_graph_shape = None  # 1-矩形, 2-圆形, 3-三角形, 4-椭圆, 5-菱形, 6-五角星
        self.pending_graph_size = None

        self.main_window = main_window
        # 矩形初始位置和大小
        self.rect_size = 50
        self.rect_newsize = 50
        self.rect = pygame.Rect(100, 100, self.rect_size, self.rect_size)

        self.circle_center = (30, 30)
        self.circle_radius = 30
        self.circle_newradius = 30

        self.elli_center = (30, 90)
        self.elli_width = 50
        self.elli_newwidth = 50
        self.elli_newheight = 100
        self.elli_height = 100
        self.elli_points = []
        self.elli_newpoints = []

        self.dia_center = (30, 30)
        self.dia_size = 60
        self.dia_vertices = []
        self.dia_newsize = 60
        self.dia_newvertices = []

        self.star_center = (30, 30)
        self.star_points = []
        self.star_newpoints = []
        self.star_outer_radius = 40
        self.star_newouter_radius = 40

        self.tri_center = (60, 60)
        self.tri_size = 100
        self.tri_newsize = 100
        self.tri_vertices = []
        self.tri_newvertices = []

        self.angle = 2 * math.pi / 5  # 五角星的内角
        self.dragging = False
        self.offset_x = 0
        self.offset_x = 0
        self.result = None
        # 新增：三次样条插值路径，避免重复计算
        self.spline_path = None
        # 新增：实时运动模拟器
        self.motion_simulator = USVMotionSimulator(self)

        # 新增：船舶运动图层
        self.ship_surface = pygame.Surface((self.width, self.height))
        self.ship_surface.set_colorkey(self.back_color)

        # 实时快照相关：从主窗口继承配置
        self.snapshot_enabled = getattr(self.main_window, 'snapshot_enabled', False)
        self.snapshot_dir = getattr(self.main_window, 'snapshot_dir', "")
        self.snapshot_index = 0
        self.final_snapshot_taken = False  # 终点快照是否已保存
        self.snapshot_start_time = None    # 快照开始时间，用于时间上限控制

        # 定时器：每 500ms 触发一次截图尝试
        self.snapshot_timer = QTimer(self)
        self.snapshot_timer.timeout.connect(self._maybe_save_snapshot)
        self.snapshot_timer.start(500)

    def start_realtime_simulation(self):
        """开始实时模拟"""
        # 仅要求起点和终点存在
        if not self.start_point or not self.end_point:
            self.main_window.printf("请先设置起点和终点！")
            return

        path_for_motion = None  # 传给运动模拟器的路径

        # 情况一：存在规划结果 self.result，按原逻辑生成平滑/稀疏化路径
        if self.result and len(self.result) >= 2:
            # 优先使用已计算的三次样条插值路径
            if self.spline_path is not None:
                spline_result = self.spline_path
                print(f"🚢 使用已计算的三次样条插值路径进行航行，路径点数: {len(spline_result)} 点")
            else:
                # 统一路径格式为(x,y)元组列表
                path_coords = []
                for point in self.result:
                    if hasattr(point, 'x') and hasattr(point, 'y'):
                        # 如果点是对象（有x,y属性）
                        path_coords.append((point.x, point.y))
                    elif isinstance(point, (tuple, list)) and len(point) >= 2:
                        # 如果点是元组或列表
                        path_coords.append((point[0], point[1]))
                    else:
                        # 其他情况，跳过
                        continue

                # 使用三次样条插值对路径进行平滑处理，确保无人船沿着平滑路径行驶
                path_optimizer = PathOptimizer()
                spline_result = path_optimizer.cubic_spline_interpolation(path_coords, num_interpolated_points=100)
                print(f"🚢 使用三次样条插值后的平滑路径进行航行: {len(path_coords)} → {len(spline_result)} 点")
                # 保存样条插值后的路径副本，确保原始路径不会消失
                self.spline_path = spline_result.copy()

            if len(spline_result) < 2:
                self.main_window.printf("路径点数量不足，无法开始实时模拟！")
                return

            # 新增：路径稀疏化处理，减少全局路径点数量
            from local_planner.algorithms.apf_local_planner import APFLocalPlanner
            apf_planner = APFLocalPlanner()
            apf_planner.reset_planner()
            # 对spline_path进行稀疏化处理，增大距离阈值以获取更少的关键节点
            sparse_path = apf_planner.discretize_path(spline_result, distance_threshold=250.0)
            print(f"🎯 使用稀疏化后的路径进行局部规划：{len(spline_result)} → {len(sparse_path)} 点")

            path_for_motion = sparse_path

            # 在规划图层上绘制用于实时航行的优化路径（淡黄色），便于与实时局部路径区分
            if sparse_path and len(sparse_path) >= 2:
                light_yellow = (255, 255, 153)
                # 先清理旧的规划路径显示
                self.plan_surface.fill(self.back_color)
                for i in range(len(sparse_path) - 1):
                    x1, y1 = sparse_path[i]
                    x2, y2 = sparse_path[i + 1]
                    pygame.draw.line(self.plan_surface, light_yellow,
                                     (int(x1), int(y1)), (int(x2), int(y2)), 3)
        else:
            # 情况二：无全局路径结果，进入“终点直航”模式
            self.main_window.printf("未检测到全局路径，已切换为以终点为目标的实时航行模式。")
            # 仅使用终点作为路径，交给 USVMotionSimulator 自动识别无全局路径模式
            path_for_motion = [self.end_point]

        # 调试信息：打印动态障碍物数量
        if hasattr(self, 'dynamic_obstacles'):
            print(f"📊 开始实时航行前 - 当前动态障碍物数量: {len(self.dynamic_obstacles)}")
            for i, obs in enumerate(self.dynamic_obstacles):
                print(f"   障碍物{i+1}: 位置={obs.position}, 形状={obs.shape}, 大小={obs.size}")
        else:
            print("❌ 未找到dynamic_obstacles属性")

        # 检查起点是否与任何障碍物发生碰撞
        if hasattr(self, 'motion_simulator'):
            print(f"🔍 进行起点碰撞检测 - 起点位置: {self.start_point}")
            # 确保motion_simulator可以访问最新的障碍物信息
            if hasattr(self.motion_simulator, 'widget'):
                self.motion_simulator.widget = self  # 重新赋值widget引用，确保获取最新状态

            # 同时检查所有类型的障碍物（与航行过程中的碰撞检测逻辑保持一致）
            collision, info = self.motion_simulator.check_all_obstacles_collision(self.start_point)
            if collision:
                obstacle_type = info.get('obstacle_type', '未知类型')
                self.main_window.printf(f"无法开始航行！起点位置与{obstacle_type}障碍物发生碰撞，请重新设置起点或移除该障碍物")
                print(f"🚨 起点碰撞检测：与{obstacle_type}障碍物在距离 {info['collision_distance']:.2f} 处碰撞")
                return
            else:
                print("✅ 起点碰撞检测通过，无碰撞")
        else:
            print("⚠️ 未找到motion_simulator，跳过起点碰撞检测")

        # 调用实时运动模拟器
        self.motion_simulator.start_journey(
            self.start_point,
            self.end_point,
            path_for_motion  # 可能是稀疏化路径，也可能仅包含终点
        )

        # 启动实时航行时，同步最新的快照开关与目录
        self.snapshot_enabled = getattr(self.main_window, 'snapshot_enabled', False)
        self.snapshot_dir = getattr(self.main_window, 'snapshot_dir', "")
        # 重置终点快照标记
        self.final_snapshot_taken = False
        # 若勾选了快照且目录有效，则记录开始时间并确保定时器开启
        if self.snapshot_enabled and self.snapshot_dir:
            self.snapshot_start_time = time.time()
            if hasattr(self, 'snapshot_timer') and not self.snapshot_timer.isActive():
                self.snapshot_timer.start(500)
        else:
            self.snapshot_start_time = None
    # A*算法
    def startAstar(self):
        self.result = None
        # self.obs_surface.fill(PygameWidget.BACK_COLOR)
        self.search = astar(self)
        self.result = self.search.process()

        if self.result is not None:
            for k in self.result:
                pygame.draw.circle(self.plan_surface, (0, 100, 255), (k.x, k.y), 3)
                # time.sleep(0.1)

            # 路径优化：对原始路径进行稀疏化处理
            if len(self.result) > 2:
                path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
                path_coords = [(p.x, p.y) for p in self.result]
                key_points = path_optimizer.extract_key_points(path_coords)
                from arithmetic.Astar.Node import point
                optimized_result = [point(x, y) for x, y in key_points]
                print(f"🔄 Astar路径优化: {len(self.result)} → {len(optimized_result)} 点")
                self.result = optimized_result

    # RRT算法
    def startRtt(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.search = Rrt(self)
        self.result, time1 = self.search.plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            # 创建路径优化器
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)

            # 转换路径格式
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            # 创建新的优化路径（复制新对象）
            from arithmetic.RRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]

            # 打印优化统计
            print(f"🔄 RRT路径优化: {len(self.result)} → {len(optimized_result)} 点")

            # 使用优化后的路径
            self.result = optimized_result

        if self.result is not None:
            for k in self.result:
                pygame.draw.circle(self.plan_surface, (0, 100, 255), (k.x, k.y), 3)
        if self.result is not None:
            for k in range(len(self.result) - 1):
                pygame.draw.line(self.plan_surface, (0, 100, 255), (self.result[k].x, self.result[k].y),
                                 (self.result[k + 1].x, self.result[k + 1].y), 3)
        track = []
        for point in self.result:
            x = point.x
            y = point.y
            track.append((x, y))
        # self.save_result(time1,track)
        return track, time1
    def startRRTStar(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.search = RrtStar(self)
        self.result, time = self.search.plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.RRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 RRT*路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            x = point.x
            y = point.y
            track.append((x, y))
        return track, time
    def startBiRRT(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.search = BiRrt(self)
        self.result, time = self.search.plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.RRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 BiRRT路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        if self.result is not None:
            for k in range(len(self.result) - 1):
                pygame.draw.line(self.plan_surface, (0, 100, 255), (self.result[k].x, self.result[k].y),
                                 (self.result[k + 1].x, self.result[k + 1].y), 3)
        track = []
        for point in self.result:
            x = point.x
            y = point.y
            #print(x,y);
            track.append((x, y))
        return track, time
    def startApf(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.search = apf(self)
        self.result, time = self.search.plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p[0], p[1]) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            optimized_result = [(x, y) for x, y in key_points]
            print(f"🔄 APF路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            track.append((point[0], point[1]))
        return track, time

    def startApfRrt(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.result, time = APFRRT(self).plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.APFRRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 APFRRT路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            track.append((point.x, point.y))

        # 转换原始路径为坐标列表
        original_track = []
        if original_path is not None:
            for point in original_path:
                original_track.append((point.x, point.y))

        return track, time, original_track
    def startApfRrt_dyn(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.result, time = APFRRT_dyn(self).plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.APFRRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 APFRRT_dyn路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            track.append((point.x, point.y))
        return track, time
    def startDbvsPRrt(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.result, time = dbvsAPFRRT_dyn(self).plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.APFRRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 dbvsAPFRRT_dyn路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            track.append((point.x, point.y))
        return track, time
    def startPRm(self):
        self.result = None
        self.result, time = prm(self).plan(self.plan_surface)

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.RRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 PRM路径优化: {len(self.result)} → {len(optimized_result)} 点")
            self.result = optimized_result

        track = []
        for point in self.result:
            track.append((point.x, point.y))
        return track, time
    def startcostRrt(self):
        self.plan_surface.fill(self.back_color)
        self.result = None
        self.result, time = Cost_Rrt(self).plan(self.plan_surface)

        # 保存原始路径的副本
        original_path = self.result.copy() if self.result is not None else None

        # 路径优化：对原始路径进行稀疏化处理，确保保留起点和终点
        if self.result is not None and len(self.result) > 2:
            path_optimizer = PathOptimizer(min_segment_length=80.0, angle_threshold=25.0)
            path_coords = [(p.x, p.y) for p in self.result]
            key_points = path_optimizer.extract_key_points(path_coords,self.obstacles)

            # 确保起点和终点
            if key_points:
                if key_points[0] != (self.start_point[0], self.start_point[1]):
                    key_points.insert(0, (self.start_point[0], self.start_point[1]))
                if key_points[-1] != (self.end_point[0], self.end_point[1]):
                    key_points.append((self.end_point[0], self.end_point[1]))

            from arithmetic.RRT.Node import point
            optimized_result = [point(x, y) for x, y in key_points]
            print(f"🔄 CostRRT路径优化: {len(self.result)} → {len(optimized_result)} 点")

            # 使用三次样条插值法进一步平滑路径
            spline_result = path_optimizer.cubic_spline_interpolation(key_points, num_interpolated_points=100)
            print(f"🔄 三次样条插值: {len(optimized_result)} → {len(spline_result)} 点")

            # 保存三次样条插值路径到类属性，避免重复计算
            self.spline_path = spline_result

            # 绘制优化后的路径
            for k in spline_result:
                pygame.draw.circle(self.plan_surface, (0, 255, 0), (int(k[0]), int(k[1])), 1)

            # 更新结果为原始优化结果（保留稀疏点用于其他处理）
            self.result = optimized_result

        if self.result is not None:
            for k in self.result:
                pygame.draw.circle(self.plan_surface, (0, 100, 255), (k.x, k.y), 3)
        if self.result is not None:
            for k in range(len(self.result) - 1):
                pygame.draw.line(self.plan_surface, (0, 100, 255), (self.result[k].x, self.result[k].y),
                                 (self.result[k + 1].x, self.result[k + 1].y), 3)
        track = []
        # 返回三次样条插值后的路径点
        if self.result is not None:
            path_coords = [(p.x, p.y) for p in self.result]
            path_optimizer = PathOptimizer()
            spline_result = path_optimizer.cubic_spline_interpolation(path_coords, num_interpolated_points=100)
            track = spline_result
        # 转换原始路径为坐标列表
        original_track = []
        if original_path is not None:
            for point in original_path:
                original_track.append((point.x, point.y))

        return track, time, original_track

    def save_result(self, time1, track, file_path, original_track=None):
        """
        保存结果文件，包括地图
        :param time1: 运行的时间
        :param track: 优化后的路径 [(x,y),(x,y),...]
        :param file_path: 文件保存路径
        :param original_track: 原始路径 [(x,y),(x,y),...]
        :return:
        """


        if file_path is None:
            self.main_window.printf("路径未选择！", None, None)
            return

        # 生成第二个文件的路径（优化路径文件）
        import os
        file_dir, file_name = os.path.split(file_path)
        file_base, file_ext = os.path.splitext(file_name)
        optimized_file_path = os.path.join(file_dir, f"{file_base}_optimized{file_ext}")

        # 创建地图和障碍物的feature_collection
        def create_map_features():
            features = []
            if self.obstacles is not None:
                for obstacle in self.obstacles:
                    if obstacle[0] != obstacle[-1]:
                        obstacle.append(obstacle[0])
                    features.append(
                        geojson.Feature(geometry=geojson.Polygon([obstacle]), properties={"name": "障碍物"}))

            if self.start_point is not None and self.end_point is not None:
                start_point = Point(self.start_point)
                end_point = Point(self.end_point)
                features.append(Feature(geometry=start_point, properties={"name": "起始点"}))
                features.append(Feature(geometry=end_point, properties={"name": "终点"}))
            if self.dynamic_obstacles is not None:
                for dynamic_obstacle in self.dynamic_obstacles:
                    shape=dynamic_obstacle.shape
                    position = dynamic_obstacle.position
                    direction = dynamic_obstacle.direction
                    speed = dynamic_obstacle.speed
                    size = dynamic_obstacle.size
                    # 创建动态障碍物的GeoJSON对象
                    dynamic_feature = geojson.Feature(
                        geometry=geojson.Point(position),
                        properties={
                            "type": "dynamic_obstacle",
                            "name": "动态障碍物",
                            "shape":shape,
                            "direction": direction,
                            "speed": speed,
                            "size": size
                        }
                    )
                    features.append(dynamic_feature)
            return FeatureCollection(features)

        # 保存第一个文件：地图信息和原始路径
        feature_collection = create_map_features()
        js2 = json.loads(str(feature_collection))

        # 如果有原始路径，则保存到第一个文件
        if original_track is not None:
            r_original = Result_Demo(self.start_point, self.end_point, time1, self.obstacles, self.dynamic_obstacles, original_track)
            js_original = dict(time=time1, track=original_track, smoothness=r_original.smoothness, pathlen=r_original.pathlen)
            js2.update(js_original)
        else:
            # 如果没有原始路径，使用优化路径作为备用
            r = Result_Demo(self.start_point, self.end_point, time1, self.obstacles, self.dynamic_obstacles, track)
            js = dict(time=time1, track=track, smoothness=r.smoothness, pathlen=r.pathlen)
            js2.update(js)

        # 将FeatureCollection保存为GeoJSON格式的字符串
        geojson_str = json.dumps(js2, indent=4)
        with open(file_path, 'w') as file:
            file.write(geojson_str)
        self.main_window.printf(f"地图和原始路径已成功保存到 {file_path}！", None, None)

        # 保存第二个文件：原始路径和优化后的路径（如果有优化路径）
        if track is not None and track != original_track:
            feature_collection_optimized = create_map_features()
            js2_optimized = json.loads(str(feature_collection_optimized))

            # 保存原始路径和优化后的路径
            r_optimized = Result_Demo(self.start_point, self.end_point, time1, self.obstacles, self.dynamic_obstacles, track)
            js_optimized = {
                "time": time1,
                "original_track": original_track if original_track is not None else [],
                "optimized_track": track,
                "smoothness": r_optimized.smoothness,
                "pathlen": r_optimized.pathlen
            }
            js2_optimized.update(js_optimized)

            # 将FeatureCollection保存为GeoJSON格式的字符串
            geojson_str_optimized = json.dumps(js2_optimized, indent=4)
            with open(optimized_file_path, 'w') as file:
                file.write(geojson_str_optimized)
            self.main_window.printf(f"原始路径和优化后的路径已成功保存到 {optimized_file_path}！", None, None)

    # 保存地图文件
    def save_map(self):
        # 创建文件对话框
        dialog = QFileDialog()
        # 设置文件对话框为保存文件模式
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        # 设置对话框标题
        dialog.setWindowTitle('保存地图')
        # 设置文件过滤器
        dialog.setNameFilter('地图文件 (*.txt)')
        # 设置默认文件名，包含文件类型后缀
        dialog.setDefaultSuffix('txt')

        # 打开文件对话框，并返回保存的文件路径
        file_path, _ = dialog.getSaveFileName(self, '保存地图', '', '地图文件 (*.txt)')

        if file_path is None:
            self.main_window.printf("路径未选择！", None, None)
            return

        features = []

        # feature_collection = None

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                if obstacle[0] != obstacle[-1]:
                    obstacle.append(obstacle[0])
                features.append(
                    geojson.Feature(geometry=geojson.Polygon([obstacle]), properties={"name": "障碍物"}))

        if self.start_point is not None and self.end_point is not None:
            start_point = Point(self.start_point)
            end_point = Point(self.end_point)
            features.append(Feature(geometry=start_point, properties={"name": "起始点"}))
            features.append(Feature(geometry=end_point, properties={"name": "终点"}))

            # 处理动态障碍物
        if self.dynamic_obstacles is not None:
            for dynamic_obstacle in self.dynamic_obstacles:
                shape=dynamic_obstacle.shape
                position = dynamic_obstacle.position
                direction = dynamic_obstacle.direction
                speed = dynamic_obstacle.speed
                size = dynamic_obstacle.size
                # 创建动态障碍物的GeoJSON对象
                dynamic_feature = geojson.Feature(
                    geometry=geojson.Point(position),
                    properties={
                        "type": "dynamic_obstacle",
                        "name": "动态障碍物",
                        "shape":shape,
                        "direction": direction,
                        "speed": speed,
                        "size": size
                    }
                )
                features.append(dynamic_feature)
        feature_collection = FeatureCollection(features)

        # 将FeatureCollection保存为GeoJSON格式的字符串
        geojson_str = json.dumps(feature_collection, indent=4)

        print(geojson_str)

        with open(file_path, 'w') as file:
            file.write(geojson_str)
        self.main_window.printf("地图已成功保存！", None, None)

        # 关闭文件保存对话框
        # self.win_main.quit()

    def update_dynamic_obstacles(self):
        """
        更新动态障碍物位置，检测碰撞并处理反弹或消失。
        """
        # 如果船舶已经发生碰撞，则保持当前动态障碍物状态，不再更新其位置
        if hasattr(self, 'motion_simulator') and getattr(self.motion_simulator, 'has_collided', False):
            return

        # 记录需要移除的障碍物
        obstacles_to_remove = []

        for obstacle in self.dynamic_obstacles:
            # 更新尾迹
            obstacle.update_trail()
            
            # 更新位置
            dx, dy = obstacle.direction
            x, y = obstacle.position
            new_x, new_y = x + dx * obstacle.speed, y + dy * obstacle.speed

            # 临时更新位置用于碰撞检测
            obstacle.position = (new_x, new_y)
            dynamic_polygon = Polygon(obstacle.to_polygon())

            # 检测与边界碰撞
            hit_boundary = False
            if new_x - obstacle.size < 0 or new_x + obstacle.size > self.width:
                hit_boundary = True
            if new_y - obstacle.size < 0 or new_y + obstacle.size > self.height:
                hit_boundary = True

            # 检测与静态障碍物碰撞
            collision = False
            for static_obstacle in self.obstacles:
                static_polygon = Polygon(static_obstacle)
                if dynamic_polygon.intersects(static_polygon):
                    collision = True
                    break

            if hit_boundary or collision:
                if obstacle.bounce:
                    # 反弹模式：恢复原位置并反向
                    obstacle.position = (x, y)
                    # 反向运动方向
                    if hit_boundary:
                        if new_x - obstacle.size < 0 or new_x + obstacle.size > self.width:
                            obstacle.direction = (-dx, dy)  # 水平方向反弹
                        if new_y - obstacle.size < 0 or new_y + obstacle.size > self.height:
                            obstacle.direction = (dx, -dy)  # 垂直方向反弹
                    if collision:
                        obstacle.direction = (-dx, -dy)  # 与静态障碍物碰撞反弹
                else:
                    # 消失模式：标记为需要移除
                    obstacle.should_remove = True
                    obstacles_to_remove.append(obstacle)
            else:
                # 如果无碰撞，则保持新位置
                obstacle.position = (new_x, new_y)

        # 移除标记为消失的障碍物
        for obstacle in obstacles_to_remove:
            if obstacle in self.dynamic_obstacles:
                self.dynamic_obstacles.remove(obstacle)

    def draw_dynamic_obstacles(self):
        """
        绘制动态障碍物到动态障碍物表面。
        """
        # 清空动态障碍物表面
        #self.dynamic_surface.fill((0, 0, 0, 0))  # 透明背景
        self.dynamic_surface.fill(self.back_color)

        # 绘制每个动态障碍物
        for obstacle in self.dynamic_obstacles:
            # 绘制尾迹
            if len(obstacle.trail) > 1:
                # 尾迹颜色：渐变的半透明效果
                for i in range(len(obstacle.trail) - 1):
                    alpha = (i + 1) / len(obstacle.trail)  # 渐变透明度
                    trail_color = (
                        int(255 * alpha),  # R
                        int(165 * alpha),  # G  
                        int(0 * alpha)     # B
                    )
                    # 绘制尾迹线段
                    start_pos = (int(obstacle.trail[i][0]), int(obstacle.trail[i][1]))
                    end_pos = (int(obstacle.trail[i + 1][0]), int(obstacle.trail[i + 1][1]))
                    pygame.draw.line(self.dynamic_surface, trail_color, start_pos, end_pos, 2)
                    
                    # 绘制尾迹点
                    pygame.draw.circle(self.dynamic_surface, trail_color, start_pos, 3)

            # 绘制障碍物本体
            x, y = obstacle.position
            if obstacle.shape == "圆形":
                # 绘制圆形障碍物
                pygame.draw.circle(self.dynamic_surface, PygameWidget.DYN_ObsColor, (int(x), int(y)),
                                   int(obstacle.size))
            elif obstacle.shape == "正方形":
                # 绘制正方形障碍物
                half_size = obstacle.size / 2
                points = [
                    (x - half_size, y - half_size),
                    (x + half_size, y - half_size),
                    (x + half_size, y + half_size),
                    (x - half_size, y + half_size)
                ]
                pygame.draw.polygon(self.dynamic_surface, PygameWidget.DYN_ObsColor, points)
            else:
                raise ValueError(f"未知动态障碍物形状: {obstacle.shape}")

    #创建动态障碍物定义，并加入列表中
    def create_dynamic_obstacle(self, x, y, shape, direction, speed, size=20, bounce=True):
        # 创建障碍物图形（可以使用 QLabel 模拟）
        # 定义运动方向（dx, dy）
        direction_map = {"向上": (0, -1), "向下": (0, 1), "向左": (-1, 0), "向右": (1, 0)}
        dx, dy = direction_map[direction]
        self.dynamic_obstacles.append(DynamicObstacle(
            shape,
            (x, y),
            (dx,dy),  # 水平方向
            speed,
            size,  # 使用传入的大小参数
            bounce  # 反弹控制参数
        ))
        #self.draw_dynamic_obstacles()



    # def move_obstacle(self, grid_widget, obstacle, dx, dy, speed):
    #     # 获取障碍物当前位置
    #     current_x = obstacle.x()
    #     current_y = obstacle.y()
    #
    #     # 更新位置
    #     new_x = current_x + dx * speed
    #     new_y = current_y + dy * speed
    #     grid_widget.dynamic_collision(new_x, new_y)
    #     # # 检测边界碰撞
    #     # if new_x < 0 or new_x + obstacle.width() > grid_widget.width():
    #     #     dx *= -1  # 水平方向反向
    #     # if new_y < 0 or new_y + obstacle.height() > grid_widget.height():
    #     #     dy *= -1  # 垂直方向反向
    #     # 设置新位置
    #     obstacle.move(new_x, new_y)
    # 打开地图文件
    def open_map(self):
        # 创建文件对话框
        dialog = QFileDialog()
        # 设置文件对话框标题
        dialog.setWindowTitle('打开地图')
        # 设置文件过滤器
        dialog.setNameFilter('地图文件 (*.txt *.csv)')

        # 打开文件对话框，并返回选择的文件路径
        file_path, _ = dialog.getOpenFileName(self, '打开地图', '', '地图文件 (*.txt *.csv)')

        # with open('file_path', 'r') as file:
        #     data = json.load(file)
        # 打开并读取txt文件内容
        if not file_path:
            self.main_window.printf("路径未选择！", None, None)
            return
        # 如果选择了文件路径，则进行地图识别的操作
        with open(file_path, 'r') as file:
            geojson_str = file.read()

        # 将字符串解析为GeoJSON对象
        geojson_obj = geojson.loads(geojson_str)
        print(geojson_obj)
        # 打开地图前需要先清空地图
        # self.clearObstacles()
        # # 清空地图后，重新设置地图分辨率
        # self.modifyMap(int(self.cell_size))
        obstacles = []
        dynamic_obstacles = []
        for feature in geojson_obj['features']:
            geometry = feature['geometry']
            properties = feature['properties']
            if geometry['type'] == 'Point' and properties['name'] == "\u8d77\u59cb\u70b9":
                self.start_point = geometry['coordinates']

            if geometry['type'] == 'Point' and properties['name'] == "\u7ec8\u70b9":
                self.end_point = geometry['coordinates']
            elif geometry['type'] == 'Polygon':
                obstacles.append(tuple(geometry['coordinates'][0]))
                # 处理动态障碍物
            elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
                shape = properties.get('shape', '正方形')
                position = tuple(geometry['coordinates'])
                direction = tuple(properties.get('direction', (1, 0)))  # 默认方向为 (1, 0)
                speed = properties.get('speed', 1.0)  # 默认速度为 1.0
                size = properties.get('size', 20.0)  # 默认大小为 5.0

                dynamic_obstacle = DynamicObstacle(shape, position, direction, speed, size)
                dynamic_obstacles.append(dynamic_obstacle)

        self.obstacles = obstacles
        self.dynamic_obstacles = dynamic_obstacles
        # print(obstacles)

        self.obs_surface.fill(self.back_color)

        for obs in obstacles:
            pygame.draw.polygon(self.obs_surface, PygameWidget.OBS_COLOR, obs)

        # 清空point_surface
        self.point_surface.fill(self.back_color)
        
        # 绘制起始点和终点到point_surface上，而不是obs_surface
        if self.start_point:
            pygame.draw.circle(self.point_surface, (0, 255, 0), self.start_point, self.point_radius)
        if self.end_point:
            pygame.draw.circle(self.point_surface, (255, 0, 0), self.end_point, self.point_radius)
        
        # 更新界面
        self.update()



    # 鼠标点击事件处理
    def mousePressEvent(self, event):
        pos = (event.pos().x(), event.pos().y())

        # 若存在待生成的图形障碍物配置，则优先在点击位置放置图形障碍物
        if event.button() == Qt.LeftButton and self.pending_graph_shape is not None:
            shape_index = self.pending_graph_shape
            size = self.pending_graph_size or 40
            # 在点击位置生成图形障碍物
            self.paint_random_one(shape_index, center=pos, size=size)
            # 使用后清空配置
            self.pending_graph_shape = None
            self.pending_graph_size = None
            self.update()
            return

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_pos = pos
            print(pos)
            pygame.draw.circle(self.obs_surface, self.obs_color, pos, self.obs_radius)
        elif event.button() == Qt.RightButton:
            pos = (event.pos().x(), event.pos().y())
            if self.obstacles:
                # 判断起点存在且不在障碍物内部
                if self.start_point is None:
                    if all(not shapely.Polygon(item).contains(shapely.Point(list(pos))) for item in self.obstacles):
                        self.start_point = pos
                        print(self.start_point)
                        pygame.draw.circle(self.point_surface, (0, 255, 0), pos, self.point_radius)
                        self.main_window.printf("设置起点", event.pos().x(), event.pos().y())
                    else:
                        self.main_window.text_result.append("起点不能设置在障碍物内部")
                else:
                    if self.start_point and self.start_point != pos and self.end_point is None:
                        if all(not shapely.Polygon(item).contains(shapely.Point(list(pos))) for item in self.obstacles):
                            self.end_point = pos
                            print(self.end_point)
                            pygame.draw.circle(self.point_surface, (255, 0, 0), pos, self.point_radius)
                            self.main_window.printf("设置终点", event.pos().x(), event.pos().y())
                        else:
                            self.main_window.text_result.append("终点不能设置在障碍物内部")
            else:
                self.main_window.text_result.append("请添加障碍物后再设置起始点")

    # 鼠标移动事件处理
    def mouseMoveEvent(self, event):

        if self.drawing:
            current_pos = (event.pos().x(), event.pos().y())
            if self.last_pos:
                draw_line(self.obs_surface, self.obs_color, self.last_pos, current_pos, self.obs_radius)
            self.last_pos = current_pos

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_pos = None
            self.get_obs_vertices()

    # 绘制pygame界面
    def paintEvent(self, event):

        self.surface.fill(PygameWidget.BACK_COLOR)
        # 更新动态障碍物位置
        self.update_dynamic_obstacles()
        # 绘制动态障碍物
        self.draw_dynamic_obstacles()
        #将多个表面叠加
        self.surface.blit(self.obs_surface, (0, 0))
        self.surface.blit(self.dynamic_surface, (0, 0))
        self.surface.blit(self.point_surface, (0, 0))
        self.surface.blit(self.plan_surface, (0, 0))
        if self.grid:
            self.surface.blit(self.grid_surface, (0, 0))


        # 绘制船舶
        if hasattr(self, 'motion_simulator') and self.motion_simulator:
            self.motion_simulator.draw_ship(self.surface)

        # 将pygame surface转换为QImage
        surface_string = pygame.image.tostring(self.surface, 'RGB')
        # width, height = self.surface.get_size()
        image = QImage(surface_string, self.width, self.height, QImage.Format_RGB888)

        # 将QImage转换为QPixmap
        pixmap = QPixmap.fromImage(image)

        # 使用QPainter绘制pixmap
        painter = QPainter(self)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

    def _maybe_save_snapshot(self):
        """在满足条件时保存当前画面快照。

        条件：
        - 主窗口勾选了实时快照（snapshot_enabled=True）
        - 已设置有效的保存目录（snapshot_dir 非空）
        - 模拟器存在且当前处于运动状态（motion_simulator.is_moving=True）
        """
        # 同步主窗口最新配置（允许途中取消勾选）
        if hasattr(self, 'main_window'):
            self.snapshot_enabled = getattr(self.main_window, 'snapshot_enabled', self.snapshot_enabled)
            self.snapshot_dir = getattr(self.main_window, 'snapshot_dir', self.snapshot_dir)

        # 基本前置条件：未勾选或无目录或无模拟器，则不保存
        if (not self.snapshot_enabled or
                not self.snapshot_dir or
                not hasattr(self, 'motion_simulator') or
                not self.motion_simulator):
            return

        ms = self.motion_simulator

        # 初始化开始时间（以防定时器在 start_realtime_simulation 之前被启动）
        if self.snapshot_start_time is None:
            self.snapshot_start_time = time.time()

        # 时间上限：超过 10 秒自动停止保存并重置勾选
        elapsed = time.time() - self.snapshot_start_time
        if elapsed > 40.0:
            # 停止快照定时器并关闭主窗口快照开关
            if hasattr(self, 'snapshot_timer') and self.snapshot_timer.isActive():
                self.snapshot_timer.stop()
            if hasattr(self, 'main_window'):
                if hasattr(self.main_window, 'snapshot_enabled'):
                    self.main_window.snapshot_enabled = False
                if hasattr(self.main_window, 'checkbox_snapshot'):
                    self.main_window.checkbox_snapshot.setChecked(False)
            # 自身状态重置
            self.snapshot_enabled = False
            self.snapshot_start_time = None
            return

        # 航行过程中，仅在 is_moving=True 时按 0.5s 保存
        if not getattr(ms, 'is_moving', False):
            return

        # 确保存储目录存在
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        except Exception:
            return

        # 组合当前完整画面
        snapshot_surface = pygame.Surface((self.width, self.height))
        snapshot_surface.fill(self.back_color)
        snapshot_surface.blit(self.obs_surface, (0, 0))
        snapshot_surface.blit(self.dynamic_surface, (0, 0))
        snapshot_surface.blit(self.point_surface, (0, 0))
        snapshot_surface.blit(self.plan_surface, (0, 0))
        if self.grid:
            snapshot_surface.blit(self.grid_surface, (0, 0))
        if hasattr(self, 'motion_simulator') and self.motion_simulator:
            self.motion_simulator.draw_ship(snapshot_surface)

        # 生成文件名：snapshot_0001.png 这样的格式
        self.snapshot_index += 1
        filename = f"snapshot_{self.snapshot_index:04d}.png"
        filepath = os.path.join(self.snapshot_dir, filename)

        try:
            pygame.image.save(snapshot_surface, filepath)
        except Exception:
            # 捕获保存异常，避免影响主循环
            pass

    def handle_journey_completed_for_snapshot(self):
        """由 USVMotionSimulator 在航行完成时调用：

        - 如启用快照且未保存过终点快照，则立即保存一张终点画面
        - 保存后停止快照定时器并关闭主窗口快照勾选
        """
        # 若未开启快照或未设置目录，直接返回
        if (not getattr(self, 'snapshot_enabled', False)) or not self.snapshot_dir:
            return

        # 已保存过终点快照，则不再重复
        if self.final_snapshot_taken:
            return

        self.final_snapshot_taken = True

        # 确保存储目录存在
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
        except Exception:
            return

        # 组合当前完整画面
        snapshot_surface = pygame.Surface((self.width, self.height))
        snapshot_surface.fill(self.back_color)
        snapshot_surface.blit(self.obs_surface, (0, 0))
        snapshot_surface.blit(self.dynamic_surface, (0, 0))
        snapshot_surface.blit(self.point_surface, (0, 0))
        snapshot_surface.blit(self.plan_surface, (0, 0))
        if self.grid:
            snapshot_surface.blit(self.grid_surface, (0, 0))
        if hasattr(self, 'motion_simulator') and self.motion_simulator:
            self.motion_simulator.draw_ship(snapshot_surface)

        # 生成文件名
        self.snapshot_index += 1
        filename = f"snapshot_{self.snapshot_index:04d}_final.png"
        filepath = os.path.join(self.snapshot_dir, filename)

        try:
            pygame.image.save(snapshot_surface, filepath)
        except Exception:
            pass

        # 保存完终点快照后，停止快照定时器并关闭主窗口勾选
        if hasattr(self, 'snapshot_timer') and self.snapshot_timer.isActive():
            self.snapshot_timer.stop()
        if hasattr(self, 'main_window'):
            if hasattr(self.main_window, 'snapshot_enabled'):
                self.main_window.snapshot_enabled = False
            if hasattr(self.main_window, 'checkbox_snapshot'):
                self.main_window.checkbox_snapshot.setChecked(False)

    # 输入始末点坐标
    def ori_end_input(self, ):  # 输入起始点终点函数
        # print("111111111111111111111111111")
        coordinate = self.main_window.text_input.text()
        print(coordinate)
        pattern = r"\((\d+),(\d+)\)"  # 匹配坐标的正则表达式模式
        match = re.match(pattern, coordinate)
        # print(match)
        if match:
            keyx = int(match.group(1))  # 提取横坐标
            keyy = int(match.group(2))  # 提取纵坐标
            print("起点横坐标:", keyx)
            print("起点纵坐标:", keyy)
            self.painting_ori(keyx, keyy)
        else:
            print("坐标格式不正确")
        coordinate_2 = self.main_window.text_input_2.text()
        # print(coordinate_2)
        pattern_2 = r"\((\d+),(\d+)\)"  # 匹配坐标的正则表达式模式
        match_2 = re.match(pattern, coordinate_2)
        if match_2:
            # global keyx_2, keyy_2
            keyx_2 = int(match_2.group(1))  # 提取横坐标
            keyy_2 = int(match_2.group(2))  # 提取纵坐标
            print("终点横坐标:", keyx_2)
            print("终点纵坐标:", keyy_2)
            self.painting_end(keyx_2, keyy_2)
        else:
            print("坐标格式不正确")

    # 栅格化地图
    def rasterize_map(self):

        obs_array = pygame.surfarray.array3d(self.obs_surface)
        # obs_array = numpy.transpose(obs_array, (1, 0, 2))
        for y in range(self.rows):
            for x in range(self.cols):
                if (x, y) == self.start_point or (x, y) == self.end_point:
                    continue
                # 获取当前栅格对应的像素块
                pixel_block = obs_array[x * self.cell_size:(x + 1) * self.cell_size,
                              y * self.cell_size:(y + 1) * self.cell_size]

                # 检查像素块中是否存在非白色像素（障碍物）
                if numpy.any(numpy.any(pixel_block != list(self.back_color), axis=2)):
                    self.grid_map[x, y] = 1  # 标记为障碍物
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size,
                                       self.cell_size)
                    pygame.draw.rect(self.obs_surface, self.obs_color, rect)

        # 绘制格线
        for y in range(self.rows + 1):
            pygame.draw.line(self.grid_surface, self.grid_color, (0, y * self.cell_size),
                             (self.width, y * self.cell_size))
        for x in range(self.cols + 1):
            pygame.draw.line(self.grid_surface, self.grid_color, (x * self.cell_size, 0),
                             (x * self.cell_size, self.height))

        # for y in range(0, self.height + 1, self.cell_size):
        #     pygame.draw.line(self.grid_surface, self.grid_color, (0, y), (self.width, y))
        # for x in range(0, self.width + 1, self.cell_size):
        #     pygame.draw.line(self.grid_surface, self.grid_color, (x, 0), (x, self.height))

        self.grid = True

    # 画起始点
    def painting_ori(self, x, y):
        # 输入新的起始点需要判断是否在之前已经生成起始点了，生成起始点就需要将原来的擦除以及self。start_point换成新的值
        if (self.start_point != None):
            pygame.draw.circle(self.obs_surface, (255, 255, 255), self.start_point, self.obs_radius)
            self.start_point = None
        self.start_point = (x, y)
        self.main_window.printf("设置起点", x, y)
        if self.start_point:
            pygame.draw.circle(self.point_surface, (0, 255, 0), (x, y), self.point_radius)
            self.update()

    def painting_end(self, x1, y1):
        # 绘画终点 假设输入终止点前地图为空，所以直接填色即可
        # 创建一个新的surface
        screen = pygame.Surface((self.width, self.height))

        # 设置背景颜色（当前逻辑中不直接使用 screen，仅保留以便后续扩展）
        # screen.fill(PygameWidget.WHITE)

        self.end_point = (x1, y1)
        self.main_window.printf("设置终点", x1, y1)
        if self.end_point:
            pygame.draw.circle(self.point_surface, (255, 0, 0), (x1, y1), self.point_radius)
            self.update()

    # 修改地图分辨率[OK]
    def modifyMap(self, size):
        if self.start_point is None and self.end_point is None and len(self.obstacles) == 0:
            if not isinstance(size, str):
                self.main_window.printf("请输入正确的分辨率！", None, None)
                if int(size) > 0:
                    new_size = int(size)
                    self.cell_size = new_size
                    print(self.cell_size)
                    self.cols = self.width // self.cell_size
                    self.rows = self.height // self.cell_size
                    self.main_window.printf("分辨率调整成功！", None, None)
                else:
                    self.main_window.printf("请输入正确的分辨率！", None, None)
        else:
            self.main_window.printf("当前地图已起始点或障碍点不可调整地图分辨率，请清空地图后再次调整分辨率！", None,
                                    None)

    # 随机起始点方法[Ok]
    def generateRandomStart(self):
        if self.start_point is None:
            # 生成随机的x和y坐标
            x = random.choice(range(0, self.width))
            y = random.choice(range(0, self.height))
            # if [x, y] not in self.obstacles:
            while any(shapely.Polygon(item).contains(shapely.Point([x, y])) for item in self.obstacles):
                # 生成随机的x和y坐标
                x = random.choice(range(0, self.width))
                y = random.choice(range(0, self.height))
            # if all(not shapely.Polygon(item).contains(shapely.Point([x, y])) for item in self.obstacles):
            self.start_point = (x, y)
            print(self.start_point)
            self.main_window.printf("添加起始点：", x, y)
            pygame.draw.circle(self.point_surface, (0, 255, 0), (x, y), self.point_radius)
            self.update()
        else:
            self.main_window.printf("error: 已经设置起点！", None, None)

        if self.end_point is None and self.end_point != self.start_point:
            # 生成随机的x和y坐标
            x_1 = random.choice(range(0, self.width))
            y_1 = random.choice(range(0, self.height))
            while any(shapely.Polygon(item).contains(shapely.Point([x_1, y_1])) for item in self.obstacles):
                # 生成随机的x和y坐标
                x_1 = random.choice(range(0, self.width))
                y_1 = random.choice(range(0, self.height))
            # if [x_1, y_1] not in self.obstacles:
            # if all(not shapely.Polygon(item).contains(shapely.Point([x_1, y_1])) for item in self.obstacles):
            self.end_point = (x_1, y_1)
            pygame.draw.circle(self.point_surface, (255, 0, 0), (x_1, y_1), self.point_radius)
            self.main_window.printf("添加终点：", y_1, x_1)
            self.update()
        else:
            self.main_window.printf("error: 已经设置终点！", None, None)
        self.update()

    def update_obs_surface(self):
        self.obs_surface.fill(self.back_color)
        for obs in self.obstacles:
            pygame.draw.polygon(self.obs_surface, self.obs_color, obs)

    # 清空地图方法[OK]
    def clear_map(self):
        self.start_point = None  # 清除起点
        self.end_point = None  # 清除终点
        self.obstacles = []  # 清空障碍物列表
        self.dynamic_obstacles = []
        self.dynamic_surface.fill(self.back_color)
        self.obs_surface.fill(self.back_color)
        self.plan_surface.fill(self.back_color)
        self.point_surface.fill(self.back_color)
        self.grid_surface.fill(self.back_color)

        # 清除全局路径规划结果与样条/稀疏路径，防止下次航行沿用旧路径
        self.result = None
        self.spline_path = None

        # 清除实时航行路径
        if hasattr(self, 'motion_simulator'):
            self.motion_simulator.history_path = []  # 清空历史轨迹
            self.motion_simulator.local_path = []  # 清空局部路径
            self.motion_simulator.path = []        # 清空用于实时航行的全局路径
            self.motion_simulator.path_index = 0   # 重置路径索引
            self.motion_simulator.current_pos = None
            self.motion_simulator.target_pos = None
            self.motion_simulator.is_moving = False  # 停止运动
            self.motion_simulator.journey_completed = False  # 重置完成状态
            self.motion_simulator.has_collided = False  # 重置碰撞状态

        self.update_map()
        self.main_window.printf("已经清空地图")
        # self.update()  # 更新界面

    # 清空地图起始点[OK]
    def clearStartAndEnd(self):
        self.start_point = None  # 清除起点
        self.end_point = None  # 清除终点
        self.update_map()
        self.main_window.printf("已经清空起始点")
        self.update()  # 更新界面

    #  更新地图界面方法[OK]
    def update_map(self):
        # 清空obs_surface
        self.obs_surface.fill(self.back_color)
        # 清空point_surface
        self.point_surface.fill(self.back_color)
        # 清空plan_surface
        self.plan_surface.fill(self.back_color)

        # 重新绘制障碍物
        for obs in self.obstacles:
            pygame.draw.polygon(self.obs_surface, self.obs_color, obs)

        # 重新绘制起点和终点，确保它们在清空图层后保持正确颜色显示
        if hasattr(self, 'start_point') and self.start_point is not None:
            # 绿色绘制起点
            pygame.draw.circle(self.point_surface, (0, 255, 0),
                              (int(self.start_point[0]), int(self.start_point[1])),
                              self.point_radius)
        if hasattr(self, 'end_point') and self.end_point is not None:
            # 红色绘制终点
            pygame.draw.circle(self.point_surface, (255, 0, 0),
                              (int(self.end_point[0]), int(self.end_point[1])),
                              self.point_radius)

    # 未栅格化地图障碍点获取
    def get_obs_vertices(self):
        capture_bgr = surface_to_cv_bgr(self.obs_surface)
        capture_gray = cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(capture_gray, 254, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.obstacles = []

        # 排除起始点和终点的半径范围
        excluded_points = []
        if self.start_point:
            excluded_points.append((self.start_point, self.point_radius + 2))
        if self.end_point:
            excluded_points.append((self.end_point, self.point_radius + 2))

        for obs in contours:
            # 检查轮廓是否过小（可能是起始点或终点）
            area = cv2.contourArea(obs)
            if area < 100:  # 过小的轮廓可能是起始点或终点
                # 检查轮廓中心点是否在排除点范围内
                M = cv2.moments(obs)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    is_excluded = False
                    for (point, radius) in excluded_points:
                        distance = math.sqrt((cx - point[0])**2 + (cy - point[1])**2)
                        if distance <= radius:
                            is_excluded = True
                            break
                    if is_excluded:
                        continue
            
            obstacle = obs.squeeze().tolist()
            # 确保obstacle是列表格式且不为空
            if isinstance(obstacle, list) and len(obstacle) > 0:
                # 确保是二维列表格式
                if isinstance(obstacle[0], int):  # 如果是一维列表，转为二维
                    obstacle = [obstacle]
                pygame.draw.polygon(self.obs_surface, self.obs_color, obstacle)
                self.obstacles.append(obstacle)

        # print(self.obstacles)

        # cv2.drawContours(capture_bgr, contours, -1, (0, 0, 255), 2)
        # self.obs_surface = cv_bgr_to_surface(capture_bgr)
        # self.obs_surface.set_colorkey(WHITE)
        # cv2.imshow('contours', capture_bgr)
        # print(contours[0].reshape(-1, 2).tolist())
        # print(contours)
        # print(hierarchy)

        # return contours

    # 图形障碍物（单个，可指定中心和大小）
    def paint_random_one(self, current_index, center, size):
        cx, cy = center
        if current_index == 1:  # 矩形
            half = size // 2
            rect = pygame.Rect(cx - half, cy - half, size, size)
            pygame.draw.rect(self.obs_surface, BLACK, rect)
        elif current_index == 2:  # 圆形
            pygame.draw.circle(self.obs_surface, BLACK, (cx, cy), size, 0)
        elif current_index == 3:  # 三角形
            height = size * math.sqrt(3) / 2
            tri_vertices = [
                (cx, cy - size / 2),
                (cx - size / 2, cy + height),
                (cx + size / 2, cy + height)
            ]
            pygame.draw.polygon(self.obs_surface, BLACK, tri_vertices)
        elif current_index == 4:  # 椭圆
            points = []
            a = size
            b = size * 0.6
            for i in range(360):
                angle_rad = math.radians(i)
                x = cx + a * math.cos(angle_rad)
                y = cy + b * math.sin(angle_rad)
                points.append((int(x), int(y)))
            if len(points) >= 3:
                pygame.draw.polygon(self.obs_surface, BLACK, points)
        elif current_index == 5:  # 菱形
            half = size // 2
            dia_vertices = [
                (cx, cy - half),
                (cx + half, cy),
                (cx, cy + half),
                (cx - half, cy)
            ]
            pygame.draw.polygon(self.obs_surface, BLACK, dia_vertices)
        elif current_index == 6:  # 五角星
            points = []
            outer_r = size
            inner_r = size * 0.4
            for i in range(10):
                r = outer_r if i % 2 == 0 else inner_r
                angle_rad = math.radians(90 + i * 36)
                x = cx + r * math.cos(angle_rad)
                y = cy - r * math.sin(angle_rad)
                points.append((int(x), int(y)))
            pygame.draw.polygon(self.obs_surface, BLACK, points)
        self.get_obs_vertices()

    #  随机多边形障碍物
    def random_graph(self, count):
        for _ in range(count):
            shape_type = random.randrange(6)  # 0表示圆形，1表示矩形
            if shape_type == 0:
                radius = random.randrange(10, 51)
                x = random.randrange(radius, self.width - radius)
                y = random.randrange(radius, self.height - radius)
                pygame.draw.circle(self.obs_surface, self.obs_color, (x, y), radius)
            elif shape_type == 1:
                width = random.randrange(20, 81)
                height = random.randrange(20, 81)
                color = (random.randrange(256), random.randrange(256), random.randrange(256))
                x = random.randrange(0, self.width - width)
                y = random.randrange(0, self.height - height)
                pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
            elif shape_type == 2:
                side_length = random.randrange(20, 81)
                x = random.randrange(0, self.width - side_length)
                y = random.randrange(0, self.height - side_length)
                triangle_points = [(x, y), (x + side_length, y), (x + side_length // 2, y + int(0.866 * side_length))]
                pygame.draw.polygon(self.obs_surface, self.obs_color, triangle_points)
            elif shape_type == 3:
                # 绘制椭圆
                width = random.randrange(20, 81)
                height = random.randrange(20, 81)
                x = random.randrange(0, self.width - width)
                y = random.randrange(0, self.height - height)
                pygame.draw.ellipse(self.obs_surface, self.obs_color, (x, y, width, height))
            elif shape_type == 4:
                # 绘制菱形
                side_length = random.randrange(20, 81)
                x = random.randrange(0, self.width - side_length)
                y = random.randrange(0, self.height - side_length)
                diamond_points = [(x + side_length // 2, y), (x + side_length, y + side_length // 2),
                                  (x + side_length // 2, y + side_length), (x, y + side_length // 2)]
                pygame.draw.polygon(self.obs_surface, self.obs_color, diamond_points)
            elif shape_type == 5:
                # 绘制五角星
                side_length = random.randrange(20, 81)
                x = random.randrange(0, self.width - side_length)
                y = random.randrange(0, self.height - side_length)
                star_points = []
                for i in range(5):
                    angle = i * 2 * math.pi / 5
                    if i % 2 == 0:
                        star_points.append((x + side_length // 2 + side_length * math.cos(angle) / 2,
                                            y + side_length * math.sin(angle) / 2))
                    else:
                        star_points.append(
                            (x + side_length // 2 + side_length * math.cos(angle), y + side_length * math.sin(angle)))
                pygame.draw.polygon(self.obs_surface, self.obs_color, star_points)
            elif shape_type == 3:
                # 绘制不重叠的椭圆
                width = random.randrange(20, 81)
                height = random.randrange(20, 81)
                x, y = self.get_non_overlapping_position(width, height)
                pygame.draw.ellipse(self.obs_surface, self.obs_color, (x, y, width, height))
            elif shape_type == 4:
                # 绘制不重叠的菱形
                side_length = random.randrange(20, 81)
                x, y = self.get_non_overlapping_position(side_length, side_length)
                diamond_points = [(x + side_length // 2, y), (x + side_length, y + side_length // 2),
                                  (x + side_length // 2, y + side_length), (x, y + side_length // 2)]
                pygame.draw.polygon(self.obs_surface, self.obs_color, diamond_points)
            elif shape_type == 5:
                # 绘制不重叠的五角星
                side_length = random.randrange(20, 81)
                x, y = self.get_non_overlapping_position(side_length, side_length)
                star_points = []
                for i in range(5):
                    angle = i * 2 * math.pi / 5
                    if i % 2 == 0:
                        star_points.append((x + side_length // 2 + side_length * math.cos(angle) / 2,
                                            y + side_length * math.sin(angle) / 2))
                    else:
                        star_points.append(
                            (x + side_length // 2 + side_length * math.cos(angle), y + side_length * math.sin(angle)))
                pygame.draw.polygon(self.obs_surface, self.obs_color, star_points)

        self.get_obs_vertices()
        self.main_window.text_result.append("成功生成 %s 个随机障碍物" % count)

    #  随机多边形不重叠障碍物
    def random_graph_new(self, count):
        self.clear_map()  # 重置地图
        obstacles = []
        max_retries = 200  # 障碍物最大寻址次数，保证每个障碍物不重叠

        # 检查一个障碍物是否与障碍物列表中的任何障碍物重叠
        def is_overlapping(x, y, width, height):
            for obstacle in obstacles:
                obstacle_x, obstacle_y, obstacle_width, obstacle_height = obstacle
                if x < obstacle_x + obstacle_width and x + width > obstacle_x and y < obstacle_y + obstacle_height and y + height > obstacle_y:
                    return True
            return False

        # 根据用户输入的数量输出障碍物
        for _ in range(count):
            retries = 0
            while retries < max_retries:
                shape_type = random.randrange(4)
                if shape_type == 0:
                    radius = random.randrange(10, 51)
                    x = random.randrange(radius, self.width - radius)
                    y = random.randrange(radius, self.height - radius)
                    if not is_overlapping(x - radius, y - radius, 2 * radius, 2 * radius):
                        obstacles.append((x - radius, y - radius, 2 * radius, 2 * radius))
                        pygame.draw.circle(self.obs_surface, self.obs_color, (x, y), radius)
                        break
                elif shape_type == 1:
                    width = random.randrange(20, 81)
                    height = random.randrange(20, 81)
                    x = random.randrange(0, self.width - width)
                    y = random.randrange(0, self.height - height)
                    if not is_overlapping(x, y, width, height):
                        obstacles.append((x, y, width, height))
                        pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
                        break
                # 添加椭圆形障碍物生成逻辑
                elif shape_type == 2:
                    width = random.randrange(20, 81)
                    height = random.randrange(20, 81)
                    x = random.randrange(0, self.width - width)
                    y = random.randrange(0, self.height - height)
                    if not is_overlapping(x, y, width, height):
                        obstacles.append((x, y, width, height))
                        pygame.draw.ellipse(self.obs_surface, self.obs_color, (x, y, width, height))
                        break
                # 添加正菱形障碍物生成逻辑
                elif shape_type == 3:
                    side_length = random.randrange(20, 81)
                    x = random.randrange(0, self.width - side_length)
                    y = random.randrange(0, self.height - side_length)
                    if not is_overlapping(x, y, side_length, side_length):
                        obstacles.append((x, y, side_length, side_length))
                        diamond_points = [(x + side_length // 2, y),
                                          (x + side_length, y + side_length // 2),
                                          (x + side_length // 2, y + side_length),
                                          (x, y + side_length // 2)]
                        pygame.draw.polygon(self.obs_surface, self.obs_color, diamond_points)
                        break
                retries += 1

        self.get_obs_vertices()
        return obstacles

    # 根据参数生成障碍物
    def graph_setting(self, quantity, size, types, overlap):
        self.clear_map()  # 重置地图
        obstacles = []
        max_retries = 200  # 障碍物最大寻址次数，保证每个障碍物不重叠
        my_array = types
        if overlap == "F":  # 不重叠障碍物
            # 检查一个障碍物是否与障碍物列表中的任何障碍物重叠
            def is_overlapping(x, y, width, height):
                for obstacle in obstacles:
                    obstacle_x, obstacle_y, obstacle_width, obstacle_height = obstacle
                    if x < obstacle_x + obstacle_width and x + width > obstacle_x and y < obstacle_y + obstacle_height and y + height > obstacle_y:
                        return True
                return False

            # 生成不规则障碍物

            def generate_smooth_blob(center, max_radius, irregularity=0.1, spikeyness=0.1, num_vertices=20):
                angle_steps = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
                angle_steps += np.random.normal(0, irregularity, num_vertices)

                points = []
                for angle in angle_steps:
                    radius = max_radius * (1 + np.random.uniform(-spikeyness, spikeyness))
                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    points.append((x, y))

                points.append(points[0])  # Closing the loop

                points = np.array(points)
                tck, u = splprep([points[:, 0], points[:, 1]], s=0.5, per=True)
                u_new = np.linspace(u.min(), u.max(), 100)
                x_new, y_new = splev(u_new, tck, der=0)

                return list(zip(x_new, y_new))

            # 根据用户输入的数量输出障碍物
            for _ in range(int(quantity)):
                retries = 0
                while retries < max_retries:
                    shape_type = random.choice(my_array)
                    if shape_type == 0:
                        # radius = random.randrange(10, 51)
                        radius = int(size) // 2  # 统一半径
                        x = random.randrange(radius, self.width - radius)
                        y = random.randrange(radius, self.height - radius)
                        if not is_overlapping(x - radius, y - radius, 2 * radius, 2 * radius):
                            obstacles.append((x - radius, y - radius, 2 * radius, 2 * radius))
                            pygame.draw.circle(self.obs_surface, self.obs_color, (x, y), radius)
                            break
                    elif shape_type == 1:
                        width = int(size)
                        height = int(size)
                        x = random.randrange(0, self.width - width)
                        y = random.randrange(0, self.height - height)
                        if not is_overlapping(x, y, width, height):
                            obstacles.append((x, y, width, height))
                            pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
                            break
                    # 添加椭圆形障碍物生成逻辑
                    elif shape_type == 2:
                        width = int(size)
                        height = int(size) // 2
                        x = random.randrange(0, self.width - width)
                        y = random.randrange(0, self.height - height)
                        if not is_overlapping(x, y, width, height):
                            obstacles.append((x, y, width, height))
                            pygame.draw.ellipse(self.obs_surface, self.obs_color, (x, y, width, height))
                            break
                    # 添加正菱形障碍物生成逻辑
                    elif shape_type == 3:
                        side_length = int(size)
                        x = random.randrange(0, self.width - side_length)
                        y = random.randrange(0, self.height - side_length)
                        if not is_overlapping(x, y, side_length, side_length):
                            obstacles.append((x, y, side_length, side_length))
                            diamond_points = [(x + side_length // 2, y),
                                              (x + side_length, y + side_length // 2),
                                              (x + side_length // 2, y + side_length),
                                              (x, y + side_length // 2)]
                            pygame.draw.polygon(self.obs_surface, self.obs_color, diamond_points)
                            break
                    elif shape_type == 4:
                        width = int(size)
                        height = int(size) // 2
                        x = random.randrange(0, self.width - width)
                        y = random.randrange(0, self.height - height)
                        if not is_overlapping(x, y, width, height):
                            obstacles.append((x, y, width, height))
                            pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
                            break
                    elif shape_type == 5:  # 不规则形状
                        max_radius = random.randrange(15,50)
                        center = (max_radius, max_radius)
                        points = generate_smooth_blob(center, max_radius)
                        if len(points) >= 3:  # 确保有足够的点数
                            offset_x = random.randrange(0, self.width - int(size))
                            offset_y = random.randrange(0, self.height - int(size))
                            points = [(x + offset_x, y + offset_y) for x, y in points]
                            bounding_box = pygame.Rect(min(x for x, y in points), min(y for x, y in points),
                                                       max(x for x, y in points) - min(x for x, y in points),
                                                       max(y for x, y in points) - min(y for x, y in points))
                            if not is_overlapping(bounding_box.left, bounding_box.top, bounding_box.width,
                                                  bounding_box.height):
                                obstacles.append(
                                    (bounding_box.left, bounding_box.top, bounding_box.width, bounding_box.height))
                                pygame.draw.polygon(self.obs_surface, self.obs_color, points)
                                break
                    retries += 1


        elif overlap == "T":  # 重叠障碍物

            for _ in range(int(quantity)):
                shape_type = random.choice(my_array)  # 0表示圆形，1表示矩形
                if shape_type == 0:
                    radius = int(size)
                    x = random.randrange(radius, self.width - radius)
                    y = random.randrange(radius, self.height - radius)
                    pygame.draw.circle(self.obs_surface, self.obs_color, (x, y), radius)
                elif shape_type == 1:
                    width = int(size)
                    height = int(size)
                    color = (random.randrange(256), random.randrange(256), random.randrange(256))
                    x = random.randrange(0, self.width - width)
                    y = random.randrange(0, self.height - height)
                    pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
                elif shape_type == 2:
                    # 绘制椭圆
                    width = int(size)
                    height = int(size) // 2
                    x = random.randrange(0, self.width - width)
                    y = random.randrange(0, self.height - height)
                    pygame.draw.ellipse(self.obs_surface, self.obs_color, (x, y, width, height))
                elif shape_type == 3:
                    # 绘制菱形
                    side_length = int(size)
                    x = random.randrange(0, self.width - side_length)
                    y = random.randrange(0, self.height - side_length)
                    diamond_points = [(x + side_length // 2, y), (x + side_length, y + side_length // 2),
                                      (x + side_length // 2, y + side_length), (x, y + side_length // 2)]
                    pygame.draw.polygon(self.obs_surface, self.obs_color, diamond_points)
                elif shape_type == 4:
                    width = int(size)
                    height = int(size) // 2
                    color = (random.randrange(256), random.randrange(256), random.randrange(256))
                    x = random.randrange(0, self.width - width)
                    y = random.randrange(0, self.height - height)
                    pygame.draw.rect(self.obs_surface, self.obs_color, (x, y, width, height))
        self.get_obs_vertices()
        return obstacles


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题
        self.setWindowTitle('Pygame in PyQt')

        # 创建Pygame部件
        self.pygame_widget = PygameWidget()

        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(self.pygame_widget)

        # 创建主窗口部件
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
