import json
import os

import geojson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from geojson import Feature, FeatureCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from shapely import Polygon, Point

import re
import pygame
from shapely.geometry import LineString
from shapely.ops import unary_union
from arithmetic.PRM.MapComplexity import analyze_map

from DynamicObstacle import DynamicObstacle


def load_demo(file_name):
    """
    加载result_demo(路径规划问题结果类)
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as file:
        geojson_str = file.read()
        # 替换字符串中的NaN为null
    geojson_str = re.sub(r'NaN', 'null', geojson_str)
    try:
        geojson_obj = geojson.loads(geojson_str)
    except ValueError as e:
        print(f"Error loading GeoJSON: {e}")
        return None  # 或者根据需要进行其他错误处理
    obstacles = []
    dynamic_obstacles = []
    for feature in geojson_obj['features']:
        geometry = feature['geometry']
        properties = feature['properties']
        if geometry['type'] == 'Point' and properties['name'] == "起始点":
            start_point = geometry['coordinates']
        if geometry['type'] == 'Point' and properties['name'] == "终点":
            end_point = geometry['coordinates']
        elif geometry['type'] == 'Polygon':
            obstacles.append(Polygon(geometry['coordinates'][0]))
        elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
            shape = properties.get('shape', '正方形')
            position = tuple(geometry['coordinates'])
            direction = tuple(properties.get('direction', (1, 0)))  # 默认方向为 (1, 0)
            speed = properties.get('speed', 1.0)  # 默认速度为 1.0
            size = properties.get('size', 20.0)  # 默认大小为 5.0
            bounce = properties.get('bounce', True)  # 读取反弹属性，默认为True

            dynamic_obstacle = DynamicObstacle(shape, position, direction, speed, size, bounce)
            dynamic_obstacles.append(dynamic_obstacle)
    # 尝试从不同的键中获取路径数据，以兼容新旧格式
    time = geojson_obj.get('time')

    # 优先使用optimized_track（如果存在），否则使用track
    track = geojson_obj.get('optimized_track') or geojson_obj.get('track')

    smoothness = geojson_obj.get('smoothness')
    pathlen = geojson_obj.get('pathlen', 0)

    # 如果没有找到track，尝试获取original_track
    if track is None:
        track = geojson_obj.get('original_track')

    # 如果仍然没有找到路径，返回一个空列表
    if track is None:
        track = []
    r = Result_Demo(start=start_point, end=end_point, time=time, obstacles=obstacles,
                    dynamic_obstacles=dynamic_obstacles, track=track,
                    smoothness=smoothness, pathlen=pathlen)
    if 'complexity' in geojson_obj:
        r.set_complexity_from_dict(geojson_obj['complexity'])
    return r


def load_look(file_name):
    """
    加载result_demo(路径规划问题结果类)
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as file:
        geojson_str = file.read()
        # 替换字符串中的NaN为null
    geojson_str = re.sub(r'NaN', 'null', geojson_str)
    try:
        geojson_obj = geojson.loads(geojson_str)
    except ValueError as e:
        print(f"Error loading GeoJSON: {e}")
        return None  # 或者根据需要进行其他错误处理
    obstacles = []
    dynamic_obstacles = []
    for feature in geojson_obj['features']:
        geometry = feature['geometry']
        properties = feature['properties']
        if geometry['type'] == 'Point' and properties['name'] == "起始点":
            start_point = geometry['coordinates']
        if geometry['type'] == 'Point' and properties['name'] == "终点":
            end_point = geometry['coordinates']
        elif geometry['type'] == 'Polygon':
            obstacles.append(Polygon(geometry['coordinates'][0]))
        elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
            shape = properties.get('shape', '正方形')
            position = tuple(geometry['coordinates'])
            direction = tuple(properties.get('direction', (1, 0)))  # 默认方向为 (1, 0)
            speed = properties.get('speed', 1.0)  # 默认速度为 1.0
            size = properties.get('size', 20.0)  # 默认大小为 5.0
            bounce = properties.get('bounce', True)  # 读取反弹属性，默认为True

            dynamic_obstacle = DynamicObstacle(shape, position, direction, speed, size, bounce)
            dynamic_obstacles.append(dynamic_obstacle)

    # 尝试从不同的键中获取路径数据，以兼容新旧格式
    track = None
    if geojson_obj:
        # 优先使用optimized_track（如果存在），否则使用track
        track = geojson_obj.get('optimized_track') or geojson_obj.get('track')

        # 如果没有找到track，尝试获取original_track
        if track is None:
            track = geojson_obj.get('original_track')

        # 如果仍然没有找到路径，返回一个空列表
        if track is None:
            track = []

    return Result_Demo(start=start_point, end=end_point, time=None, obstacles=obstacles,
                       dynamic_obstacles=dynamic_obstacles, track=track,
                       smoothness=None, pathlen=0)


class Result_Demo:
    """
    路径规划结果类
    """

    def __init__(self, start, end, time, obstacles, dynamic_obstacles, track, smoothness=None, pathlen=0, obs_surface=None, node_count=None, edge_count=None):
        """
        路径规划的结果分析类（中文增强版）
        :param start: 起点 [x,y]
        :param end: 终点 [x,y]
        :param time: 运行时间（秒）
        :param obstacles: 静态障碍物 Polygon[]
        :param dynamic_obstacles: 动态障碍物 DynamicObstacle[]
        :param track: 路径 [(x,y),(x,y)...]
        :param smoothness: 平滑度（可选，若为空将自动计算）
        :param pathlen: 路径的长度（可选，若为0将自动计算）
        :param obs_surface: 可选的pygame.Surface，用于计算复杂度指标
        :param node_count: 可选的采样节点数
        :param edge_count: 可选的连边数
        """
        # 原始字段
        self.start = start
        self.end = end
        self.time = time
        self.obstacles = obstacles or []
        self.dynamic_obstacles = dynamic_obstacles or []
        self.track = track or []
        self.smoothness = smoothness
        self.pathlen = pathlen
        self.node_count = node_count
        self.edge_count = edge_count

        # 兼容：自动计算常用指标
        if not self.pathlen:
            self.calculate_path_length()
        if self.smoothness is None and self.track:
            try:
                self.compute_smoothness()
            except Exception:
                self.smoothness = None

        # 计算扩展评估指标
        self._curv_stats = self._compute_curvature_stats()
        self._turn_stats = self._compute_turn_stats()
        self._clearance_stats, self._collision_count = self._compute_clearance_and_collisions()

        # 复杂度：若提供了障碍物Surface，则自动评估
        self.complexity = None
        if obs_surface is not None:
            try:
                self.complexity = analyze_map(obs_surface)
            except Exception:
                self.complexity = None

        # —— 中文别名（属性）
        self.起点 = self.start
        self.终点 = self.end
        self.用时 = self.time
        self.路径 = self.track
        self.路径长度 = self.pathlen
        self.路径平滑度 = self.smoothness
        self.转角数 = self._turn_stats.get('turn_count', 0)
        self.平均转角 = self._turn_stats.get('mean_abs_deg', None)
        self.曲率均值 = self._curv_stats.get('mean', None)
        self.曲率最大值 = self._curv_stats.get('max', None)
        self.曲率标准差 = self._curv_stats.get('std', None)
        self.最小安全裕度 = self._clearance_stats.get('min', None)
        self.平均安全裕度 = self._clearance_stats.get('mean', None)
        self.碰撞段数 = self._collision_count
        self.直线率 = self._straightness_ratio()
        self.曲折度 = None if (self.直线率 in (0, None)) else float(1.0 / self.直线率)
        self.节点数 = self.node_count
        self.连边数 = self.edge_count

    def draw_start(self):
        """
        画出起点
        :return:
        """
        plt.scatter(self.start[0], self.start[1], marker='o', color='blue', s=20)

    def draw_end(self):
        """
        画出终点
        :return:
        """
        plt.scatter(self.end[0], self.end[1], marker='*', color='red', s=20)

    def draw_obstacles(self):
        """
        画障碍物
        :return:
        """
        for polygon in self.obstacles:
            x, y = polygon.exterior.xy
            plt.plot(x, y, color='black')
            plt.fill(x, y, alpha=1, color='black')
        # 绘制动态障碍物
        for dynamic_obstacle in self.dynamic_obstacles:
            # 获取动态障碍物的多边形形状
            polygon = Polygon(dynamic_obstacle.to_polygon())
            x, y = polygon.exterior.xy
            plt.plot(x, y, color='red')  # 红色轮廓线
            plt.fill(x, y, alpha=1, color='red')  # 半透明红色填充

            # 绘制运动方向箭头
            position = dynamic_obstacle.position
            direction = dynamic_obstacle.direction
            arrow_scale = dynamic_obstacle.size*2   # 箭头的缩放因子
            plt.arrow(
                position[0], position[1],  # 箭头起点
                direction[0] * arrow_scale, direction[1] * arrow_scale,  # 箭头方向和长度
                head_width=0.3 * arrow_scale, head_length=0.5 * arrow_scale, fc='red', ec='red'
            )



    def draw_track(self):
        """
        画出路径
        :return:
        """
        figure = plt.figure()
        self.draw_end()
        self.draw_start()
        self.draw_obstacles()
        a = [x for (x, y) in self.track]
        b = [y for (x, y) in self.track]
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()  # 将x轴刻度显示在上方
        plt.plot(a, b, marker='o', color='blue')
        plt.title("track")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        return FigureCanvas(figure)

    def compute_curvature(self, x, y):
        # 计算路径的曲率
        dx = np.diff(x)
        dy = np.diff(y)
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        curvature = np.abs(ddx * dy[1:] - dx[1:] * ddy) / (dx[1:] ** 2 + dy[1:] ** 2) ** (3 / 2)
        return curvature

    def compute_linearity(self, x, y):
        # 计算起点到终点的直线距离
        start_point = np.array([x[0], y[0]])
        end_point = np.array([x[-1], y[-1]])
        straight_line_distance = np.linalg.norm(end_point - start_point)

        # 计算路径的总长度
        path_length = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

        # 线性度，即路径长度与直线距离的比值
        linearity = straight_line_distance / path_length
        print(linearity)
        # 平滑度为线性度的补数，1 - linearity
        smoothness = 1 - linearity

        return smoothness

    def compute_smoothness(self):
        """计算路径平滑度"""
        path_x = [x for (x, y) in self.track]
        path_y = [y for (x, y) in self.track]
        # 计算曲率
        curvature = self.compute_curvature(path_x, path_y)

        # 计算平均曲率作为路径平滑度的指标
        average_curvature = np.mean(curvature)
        self.smoothness = round(average_curvature, 2)

    def draw_curvature(self):
        # 绘制曲率
        figure = plt.figure()
        path_x = [x for (x, y) in self.track]
        path_y = [y for (x, y) in self.track]
        t = np.linspace(0, self.time, len(path_x))
        plt.plot(t[1:], self.compute_curvature(path_x, path_y), marker='o', color='r')

        plt.title("curvature")
        plt.xlabel("X")
        plt.ylabel("curvature")

        plt.tight_layout()
        return FigureCanvas(figure)

    def calculate_path_length(self):
        """计算路径总长度"""
        a = [x for (x, y) in self.track]
        b = [y for (x, y) in self.track]
        self.pathlen = np.sum(np.sqrt(np.diff(a) ** 2 + np.diff(b) ** 2))

    # ————— 扩展评估计算 —————
    def _straightness_ratio(self):
        """直线率 = 起终点直线距离 / 路径长度 ∈ (0,1]；若路径长度为0返回None。"""
        if not self.track or self.pathlen in (0, None):
            return None
        x = [p[0] for p in self.track]; y = [p[1] for p in self.track]
        straight = float(((x[-1]-x[0])**2 + (y[-1]-y[0])**2) ** 0.5)
        if straight <= 1e-9 or self.pathlen <= 1e-9:
            return None
        return float(straight / self.pathlen)

    def _compute_curvature_stats(self):
        """返回曲率统计: mean/max/std（若无路径则为空字典）"""
        try:
            if not self.track or len(self.track) < 3:
                return {}
            x = [p[0] for p in self.track]; y = [p[1] for p in self.track]
            kappa = self.compute_curvature(x, y)
            if kappa is None or len(kappa) == 0:
                return {}
            arr = np.asarray(kappa, dtype=float)
            return dict(mean=float(np.mean(arr)), max=float(np.max(arr)), std=float(np.std(arr)))
        except Exception:
            return {}

    def _compute_turn_stats(self, deg_threshold: float = 15.0):
        """
        计算转角统计：
        - turn_count: 绝对转角>阈值(默认15°)的节点数
        - mean_abs_deg: 平均绝对转角（度）
        """
        try:
            pts = self.track
            if not pts or len(pts) < 3:
                return dict(turn_count=0, mean_abs_deg=None)
            import math
            def angle(a,b,c):
                ax, ay = a; bx, by = b; cx, cy = c
                v1x, v1y = bx-ax, by-ay
                v2x, v2y = cx-bx, cy-by
                n1 = (v1x*v1x + v1y*v1y) ** 0.5
                n2 = (v2x*v2x + v2y*v2y) ** 0.5
                if n1<1e-9 or n2<1e-9: return 0.0
                cosang = max(-1.0, min(1.0, (v1x*v2x + v1y*v2y) / (n1*n2)))
                return math.degrees(math.acos(cosang))
            angs = [abs(angle(pts[i-1], pts[i], pts[i+1])) for i in range(1, len(pts)-1)]
            turn_count = int(sum(1 for a in angs if a > deg_threshold))
            mean_abs = float(sum(angs)/len(angs)) if angs else None
            return dict(turn_count=turn_count, mean_abs_deg=mean_abs)
        except Exception:
            return dict(turn_count=0, mean_abs_deg=None)

    def _compute_clearance_and_collisions(self):
        """
        计算路径安全裕度（到最近障碍物的距离）与碰撞段数。
        返回: ({{min, mean}}, collision_count)
        """
        try:
            # 收集静态+动态障碍物
            polys = []
            for poly in (self.obstacles or []):
                if isinstance(poly, Polygon):
                    polys.append(poly)
            for dob in (self.dynamic_obstacles or []):
                try:
                    poly = Polygon(dob.to_polygon())
                    polys.append(poly)
                except Exception:
                    pass
            if not polys:
                return (dict(min=None, mean=None), 0)

            u = unary_union(polys)
            # 计算每个点的距离
            dists = []
            pts = self.track or []
            for p in pts:
                dists.append(Point(p).distance(u))
            min_d = float(min(dists)) if dists else None
            mean_d = float(sum(dists)/len(dists)) if dists else None

            # 统计碰撞段数
            collision = 0
            for i in range(len(pts)-1):
                seg = LineString([pts[i], pts[i+1]])
                try:
                    if seg.intersects(u) or seg.touches(u):
                        # touches 视为贴边，可按需调整
                        collision += 1
                except Exception:
                    pass
            return (dict(min=min_d, mean=mean_d), int(collision))
        except Exception:
            return (dict(min=None, mean=None), 0)

    # ———— 复杂度注入/导出 ————
    def set_complexity_from_surface(self, obs_surface: pygame.Surface):
        """基于障碍物Surface计算复杂度指标。"""
        try:
            self.complexity = analyze_map(obs_surface)
        except Exception:
            self.complexity = None

    def set_complexity_from_dict(self, metrics: dict):
        """直接注入复杂度字典（需符合 MapComplexity.analyze_map 的返回格式）。"""
        self.complexity = dict(metrics) if metrics else None

    # ———— 导出指标字典（英文/中文） ————
    def to_metrics_dict(self, zh: bool = False):
        """
        返回一个包含规划评估指标与复杂度指标的字典。
        zh=False 时字段使用英文，便于与旧代码兼容。
        """
        # 常规
        straight = self._straightness_ratio()
        tortuosity = None if (straight in (0, None)) else float(1.0/straight)
        base = dict(
            time=self.time,
            success=bool(self.track and len(self.track) >= 2 and self.碰撞段数 == 0),
            pathlen=self.pathlen,
            smoothness=self.smoothness,
            curvature_mean=self._curv_stats.get('mean'),
            curvature_max=self._curv_stats.get('max'),
            curvature_std=self._curv_stats.get('std'),
            turn_count=self._turn_stats.get('turn_count'),
            mean_turn_deg=self._turn_stats.get('mean_abs_deg'),
            straightness=straight,
            tortuosity=tortuosity,
            clearance_min=self._clearance_stats.get('min'),
            clearance_mean=self._clearance_stats.get('mean'),
            node_count=self.node_count,
            edge_count=self.edge_count,
            track=self.track  # 便于保存
        )
        # 复杂度
        if self.complexity:
            comp = {k:self.complexity.get(k) for k in ['S','level','rho','nu','B','mu_v','var_v','C','p_edge','B_tilde','cell','cols','rows','r_robot']}
            base['complexity'] = comp

        if zh:
            # 转中文键
            cn = dict(
                用时=base['time'],
                成功=base['success'],
                路径长度=base['pathlen'],
                平滑度=base['smoothness'],
                曲率均值=base['curvature_mean'],
                曲率最大值=base['curvature_max'],
                曲率标准差=base['curvature_std'],
                转角数=base['turn_count'],
                平均转角=base['mean_turn_deg'],
                直线率=base['straightness'],
                曲折度=base['tortuosity'],
                最小安全裕度=base['clearance_min'],
                平均安全裕度=base['clearance_mean'],
                节点数=base['node_count'],
                连边数=base['edge_count'],
                复杂度=base.get('complexity'),
                轨迹=base['track']
            )
            return cn
        return base


# —— 中文类名别名（不破坏旧导入） ——
class 结果(Result_Demo):
    """与 Result_Demo 等价的中文别名；新增中文属性与导出方法可直接使用。"""
    pass


class Category_Demo:
    """
    结果类的集合
    """
    def __init__(self):
        self.results = []
        self.file_name = []
        self.ave_smooth = None
        self.ave_path_length = None
        self.ave_time = None
        self.name = None

    def read(self, path):
        """
        已知完整路径,从文件中读取category
        :param path:完整路径
        :return:None
        """
        with open(path, 'r') as file:
            geojson_str = file.read()
        geojson_obj = geojson.loads(geojson_str)
        self.name = geojson_obj['name']
        self.ave_smooth = geojson_obj['ave_smooth']
        self.ave_path_length = geojson_obj['ave_path_length']
        self.ave_time = geojson_obj['ave_time']

    def read_file(self, path):
        """
        读取文件，从多个结果文件生成一个category
        :param path: 文件夹的路径
        :return: None
        """
        files = os.listdir(path)
        for file in files:
            self.results.append(load_demo(os.path.join(path, file)))  # 读取结果
            self.file_name.append(file)  # 保存文件名
        self.calculate()

    def save_file(self, file_path, name):
        """
        保存category文件:1)地图 2)算法名 3)平均平滑度 4)平均路径长度 5)平均用时
        :param name: 算法的名称
        :param file_path: 保存的路径
        :return: None
        """
        self.calculate()
        features = []
        for r in self.results:
            a = [x for (x, y) in r.track]
            b = [y for (x, y) in r.track]
            # 将障碍物也转化为GeoJSON
            for polygon in r.obstacles:
                coords = list(polygon.exterior.coords)
                features.append(Feature(geometry={"type": "Polygon", "coordinates": [coords]},
                                        properties={"name": "障碍物"}))
            feature_collection = FeatureCollection(features)
        geojson_obj = json.loads(str(feature_collection))
        # 添加路径相关的信息
        geojson_obj.update({
            'name': name,
            'ave_smooth': self.ave_smooth,
            'ave_path_length': self.ave_path_length,
            'ave_time': self.ave_time,
        })
        with open(file_path, 'w') as file:
            file.write(str(geojson_obj))

    def track_compare(self):
        """
        绘制不同路径的对比
        :return: null
        """
        figure = plt.figure()
        self.results[0].draw_obstacles()
        self.results[0].draw_end()
        self.results[0].draw_start()
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()  # 将x轴刻度显示在上方
        # 定义颜色和线型列表
        colors = ['b', 'orangered', 'g', 'm', 'm', 'r', 'k']  # 颜色列表，例如：蓝、绿、红、青、品红、黄、黑
        line_styles = ['--', '-', '-.', ':']  # 线型列表，例如：实线、虚线、点划线、点线
        for i, (r, n) in enumerate(zip(self.results, self.file_name)):
            if n.endswith('.txt'):
                n = n[:-4]  # 去除最后四个字符
            a = [x for (x, y) in r.track]
            b = [y for (x, y) in r.track]

            # 绘制路径，使用不同的颜色和线型
            plt.plot(a, b, label=n, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
        plt.legend(loc='upper left')
        return FigureCanvas(figure)

    def calculate(self):
        """
        计算
        :return: None
        """
        self.calculate_average_time()
        self.calculate_average_smoothness()
        self.calculate_average_length()

    def calculate_average_smoothness(self):
        """
        求多个结果的平均路径平滑度
        :return:None
        """
        total_smoothness = 0
        result_count = len(self.results)
        for r in self.results:
            total_smoothness += r.smoothness
        self.ave_smooth = total_smoothness / result_count

    def calculate_average_length(self):
        """
        求多个结果的平均路径长度
        :return:None
        """
        total_length = 0
        result_count = len(self.results)
        for r in self.results:
            total_length += r.pathlen
        self.ave_path_length = total_length / result_count

    def calculate_average_time(self):
        """
        求多个结果的平均用时
        :return:None
        """
        total_time = 0
        result_count = len(self.results)
        for r in self.results:
            total_time += r.time
        self.ave_time = total_time / result_count


class Category_Compare:
    """
    Category 的比较
    """
    def __init__(self):
        self.category = []
        self.df = None  # 生成的数据表格

    def read_category(self, files):
        """
        读取一系列category文件
        :param files: 完整路径数组
        :return:
        """
        for file in files:
            category = Category_Demo()
            category.read(file)
            self.category.append(category)
        data = []
        for c in self.category:
            data.append([c.name, c.ave_smooth, c.ave_path_length, c.ave_time])
        self.df = pd.DataFrame(data, columns=['算法名', '平均平滑度', '平均轨迹长度', '平均用时'])

    def save_image(self, file_path):
        df = self.df

        fig, ax = plt.subplots(figsize=(8, 4))

        # 去除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        # 使用table函数将DataFrame绘制到axes对象上
        the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        # 自动调整列宽
        the_table.auto_set_column_width(col=list(range(len(df.columns))))
        # 自动调整行高
        the_table.scale(1, 1.5)
        # 隐藏坐标轴
        ax.axis('off')
        # 将figure保存为图片
        canvas = FigureCanvas(fig)
        canvas.print_figure(file_path, dpi=300)

    def save_csv(self, file_path):
        self.df.to_csv(file_path, index=False, encoding='utf-8-sig')
