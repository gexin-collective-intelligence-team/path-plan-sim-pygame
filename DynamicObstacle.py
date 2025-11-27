# 动态障碍物数据结构
import math

from shapely import Point


class DynamicObstacle:
    def __init__(self, shape, position, direction, speed, size, bounce=True):
        """
        :param shape: str, 形状类型（如 '圆形', '正方形', '椭圆' 等）
        :param position: tuple(float, float), 障碍物中心位置 (x, y)
        :param direction: tuple(float, float), 方向向量 (dx, dy)
        :param speed: float, 运动速度
        :param size: float, 障碍物大小（半径或边长）
        :param bounce: bool, 是否在边界反弹（True为反弹，False为消失）
        """
        self.shape = shape
        self.position = position
        self.direction = direction
        self.speed = speed
        self.size = size
        self.bounce = bounce
        
        # 尾迹轨迹：存储历史位置点
        self.trail = []
        self.max_trail_length = 30  # 最大尾迹长度
        self.trail_counter = 0  # 用于控制尾迹记录频率
        
        # 标记是否应该被移除（用于不反弹的情况）
        self.should_remove = False

    def update_trail(self):
        """更新尾迹轨迹"""
        # 每隔几帧记录一次位置，避免尾迹过密
        self.trail_counter += 1
        if self.trail_counter % 2 == 0:  # 每2帧记录一次
            self.trail.append(self.position)
            # 限制尾迹长度
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)

    def predict_future_position(self, time_delta):
        """根据速度和时间预测未来的位置"""
        # 计算速度向量 (velocity)
        velocity_x = self.direction[0] * self.speed
        velocity_y = self.direction[1] * self.speed

        # 计算未来位置
        future_x = self.position[0] + velocity_x * time_delta
        future_y = self.position[1] + velocity_y * time_delta
        return Point(future_x, future_y)  # 返回预测的点
    def to_polygon(self):
        """
        将障碍物转换为多边形形式，便于碰撞检测。
        """
        x, y = self.position
        if self.shape == "圆形":
            # 将圆形近似为多边形（例如正十二边形）
            num_segments = 12  # 分段数量
            return [
                (
                    x + self.size * math.cos(2 * math.pi * i / num_segments),
                    y + self.size * math.sin(2 * math.pi * i / num_segments),
                )
                for i in range(num_segments)
            ]
        elif self.shape == "正方形":
            half_size = self.size / 2
            return [
                (x - half_size, y - half_size),
                (x + half_size, y - half_size),
                (x + half_size, y + half_size),
                (x - half_size, y + half_size),
            ]
        else:
            raise ValueError(f"未知形状: {self.shape}")

    def contains_point(self, point):
        """
        检查给定点是否在动态障碍物内部
        :param point: tuple(float, float) or shapely.Point, 要检查的点
        :return: bool, 点是否在障碍物内部
        """
        if isinstance(point, tuple):
            x, y = point
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            x, y = point.x, point.y
        else:
            raise ValueError("点必须是tuple格式或shapely.Point对象")
        
        obstacle_x, obstacle_y = self.position
        
        if self.shape == "圆形":
            # 圆形：计算距离
            distance = math.sqrt((x - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
            return distance <= self.size
        elif self.shape == "正方形":
            # 正方形：检查点是否在正方形内
            half_size = self.size / 2
            return (obstacle_x - half_size <= x <= obstacle_x + half_size and 
                    obstacle_y - half_size <= y <= obstacle_y + half_size)
        else:
            raise ValueError(f"未知形状: {self.shape}")

