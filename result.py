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
        if geometry['type'] == 'Point' and properties['name'] == "\u8d77\u59cb\u70b9":
            start_point = geometry['coordinates']
        if geometry['type'] == 'Point' and properties['name'] == "\u7ec8\u70b9":
            end_point = geometry['coordinates']
        elif geometry['type'] == 'Polygon':
            obstacles.append(Polygon(geometry['coordinates'][0]))
        elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
            shape = properties.get('shape', '正方形')
            position = tuple(geometry['coordinates'])
            direction = tuple(properties.get('direction', (1, 0)))  # 默认方向为 (1, 0)
            speed = properties.get('speed', 1.0)  # 默认速度为 1.0
            size = properties.get('size', 20.0)  # 默认大小为 5.0

            dynamic_obstacle = DynamicObstacle(shape, position, direction, speed, size)
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
    return Result_Demo(start=start_point, end=end_point, time=time, obstacles=obstacles,
                       dynamic_obstacles=dynamic_obstacles, track=track,
                       smoothness=smoothness, pathlen=pathlen)
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
        if geometry['type'] == 'Point' and properties['name'] == "\u8d77\u59cb\u70b9":
            start_point = geometry['coordinates']
        if geometry['type'] == 'Point' and properties['name'] == "\u7ec8\u70b9":
            end_point = geometry['coordinates']
        elif geometry['type'] == 'Polygon':
            obstacles.append(Polygon(geometry['coordinates'][0]))
        elif geometry['type'] == 'Point' and properties.get('type') == "dynamic_obstacle":
            shape = properties.get('shape', '正方形')
            position = tuple(geometry['coordinates'])
            direction = tuple(properties.get('direction', (1, 0)))  # 默认方向为 (1, 0)
            speed = properties.get('speed', 1.0)  # 默认速度为 1.0
            size = properties.get('size', 20.0)  # 默认大小为 5.0

            dynamic_obstacle = DynamicObstacle(shape, position, direction, speed, size)
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

    def __init__(self, start, end, time, obstacles, dynamic_obstacles, track, smoothness=None, pathlen=0):
        """
        路径规划的结果分析类
        :param start: 起点 [x,y]
        :param end: 终点 [x,y]
        :param time: 时间
        :param obstacles: 障碍物 Polygon[]
        :param track: 路径 [(x,y),(x,y)...]
        :param smoothness: 平滑度
        :param pathlen: 路径的长度
        """
        self.start = start
        self.end = end
        self.time = time
        self.obstacles = obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.track = track
        self.smoothness = smoothness
        if smoothness is None:
            self.compute_smoothness()
        self.pathlen = pathlen
        if pathlen == 0:
            self.calculate_path_length()

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
        plt.plot(a, b)
        return FigureCanvas(figure)
    def draw_map(self):
        """
        画出路径
        :return:
        """
        figure = plt.figure()
        self.draw_end()
        self.draw_start()
        self.draw_obstacles()
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()  # 将x轴刻度显示在上方
        return FigureCanvas(figure)
    # -----------------------计算路径平滑度-----------------------------------
    def compute_curvature(self, x, y):
        """计算曲率"""
        slopes = np.diff(y) / np.diff(x)

        # 计算曲率，这里使用曲率的定义 k = |dy/dx| / (1 + (dy/dx)^2)^(3/2)
        curvature = np.abs(slopes) / np.sqrt(1 + slopes ** 2)
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

    # -----------------------------------计算路径长度-------------------------------
    def calculate_path_length(self):
        # 计算路径长度

        for i in range(len(self.track) - 1):
            # 计算相邻两点之间的欧几里得距离
            dx = self.track[i + 1][0] - self.track[i][0]
            dy = self.track[i + 1][1] - self.track[i][1]
            distance = math.sqrt(dx * dx + dy * dy)
            # 累加到路径长度
            self.pathlen += distance
        print(f"路径长度为:{self.pathlen}")


class Category_Demo:
    """
    多结果分析:同地图同算法运行多次
    画图:1)路径比较2)平滑度比较
    数据：1)计算平滑度2)计算平均路径长度3)平均用时
    """

    def __init__(self):
        """
        初始化
        """
        self.results = []  # 定义数组类型
        self.file_name = []
        self.name = None
        self.ave_smooth = None
        self.ave_path_length = 0
        self.ave_time = 0

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
        # 1)地图都是一样的,就直接取出results[0]的就可以
        obstacles = self.results[0].obstacles
        start_point = self.results[0].start
        end_point = self.results[0].end
        # 2)算法名:name
        self.name = name
        # 3)平均平滑度：self.ave_smooth
        # 4)平均路径长度: self.ave_path_length
        # 5)平均用时：self.ave.time
        features = []
        if obstacles is not None:
            for obstacle in obstacles:
                features.append(
                    geojson.Feature(geometry=obstacle, properties={"name": "障碍物"}))
        if start_point is not None and end_point is not None:
            features.append(Feature(geometry=Point(start_point), properties={"name": "起始点"}))
            features.append(Feature(geometry=Point(end_point), properties={"name": "终点"}))
        feature_collection = FeatureCollection(features)
        # 先封装经过计算再存入文件
        js2 = json.loads(str(feature_collection))
        js = dict(name=name, ave_smooth=self.ave_smooth, ave_path_length=self.ave_path_length, ave_time=self.ave_time)
        js2.update(js)
        # 将FeatureCollection保存为GeoJSON格式的字符串
        geojson_str = json.dumps(js2, indent=4)
        with open(file_path, 'w') as file:
            file.write(geojson_str)

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
        求平均用时
        :return:None
        """
        times = [r.time for r in self.results]
        self.ave_time = sum(times) / len(times)


class Category_Compare:
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
        data = {
            'name': [c.name for c in self.category],
            'average time': [c.ave_time for c in self.category],
            'average path length': [c.ave_path_length for c in self.category],
            'average smoothness': [c.ave_smooth for c in self.category]
        }
        self.df = pd.DataFrame(data)

    def save_image(self, file_path):
        df = self.df
        # 创建一个figure和axes对象
        fig, ax = plt.subplots()
        # 去掉x轴和y轴的刻度
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
