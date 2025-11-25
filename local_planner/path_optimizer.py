"""
全局路径点优化器
用于将密集的全局路径转换为稀疏的关键路径点，便于局部路径规划
"""

import numpy as np
from typing import List, Tuple
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon, LineString


class PathOptimizer:
    """路径点优化器，提取关键拐点"""
    
    def __init__(self, min_segment_length=50.0, angle_threshold=30.0):
        """
        初始化路径优化器
        
        Args:
            min_segment_length: 最小路径段长度，小于此值的点将被合并
            angle_threshold: 角度变化阈值（度），大于此值认为是拐点
        """
        self.min_segment_length = min_segment_length
        self.angle_threshold = np.radians(angle_threshold)
    
    def extract_key_points(self, path: List[Tuple[float, float]], obstacles: List = None) -> List[Tuple[float, float]]:
        """
        从密集路径中提取关键拐点，同时验证每段路径的碰撞情况
        
        Args:
            path: 原始路径点列表 [(x1,y1), (x2,y2), ...]
            obstacles: 障碍物列表（ shapely.Polygon）
            
        Returns:
            关键路径点列表，稀疏但包含所有重要拐点，且路径段均可行
        """
        if len(path) < 3:
            return path

        # 第一步：去除过近的点
        filtered_path = self._remove_close_points(path)

        # 第二步：提取拐点（只基于角度，不强行加入碰撞点）
        key_points = [filtered_path[0]]
        for i in range(1, len(filtered_path) - 1):
            prev_vec = np.array(filtered_path[i]) - np.array(filtered_path[i-1])
            next_vec = np.array(filtered_path[i+1]) - np.array(filtered_path[i])

            prev_len = np.linalg.norm(prev_vec)
            next_len = np.linalg.norm(next_vec)

            if prev_len > 0 and next_len > 0:
                prev_vec = prev_vec / prev_len
                next_vec = next_vec / next_len

                angle = np.arccos(np.clip(np.dot(prev_vec, next_vec), -1.0, 1.0))

                if angle > self.angle_threshold:
                    # 只在不碰撞的情况下直接添加
                    if not obstacles or not self._line_collision(key_points[-1], filtered_path[i], obstacles):
                        key_points.append(filtered_path[i])
                    # 否则跳过，留给 validate_segments 处理

        # 第三步：加上终点
        if key_points[-1] != filtered_path[-1]:
            key_points.append(filtered_path[-1])

        # 第四步：统一进行段验证，移除/补充不可行的段
        if obstacles:
            key_points = self._validate_segments(key_points, obstacles)

        return key_points
    
    def _remove_close_points(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """去除距离过近的点"""
        if len(path) < 2:
            return path
        
        filtered = [path[0]]
        for i in range(1, len(path)):
            prev_point = filtered[-1]
            curr_point = path[i]
            distance = np.linalg.norm(np.array(curr_point) - np.array(prev_point))
            
            if distance >= self.min_segment_length or i == len(path) - 1:
                filtered.append(curr_point)
        
        return filtered
    
    def _extract_turning_points(self, path: List[Tuple[float, float]], obstacles: List = None) -> List[Tuple[float, float]]:
        """提取路径中的拐点，并实时验证与前一个关键点的连接是否碰撞"""
        if len(path) < 3:
            return path
        
        key_points = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev_vec = np.array(path[i]) - np.array(path[i-1])
            next_vec = np.array(path[i+1]) - np.array(path[i])
            
            # 计算角度变化
            prev_len = np.linalg.norm(prev_vec)
            next_len = np.linalg.norm(next_vec)
            
            if prev_len > 0 and next_len > 0:
                prev_vec = prev_vec / prev_len
                next_vec = next_vec / next_len
                
                # 计算夹角
                angle = np.arccos(np.clip(np.dot(prev_vec, next_vec), -1.0, 1.0))
                
                # 如果角度变化大于阈值，认为是拐点
                if angle > self.angle_threshold:
                    # 实时验证与前一个关键点的连接是否碰撞
                    if obstacles:
                        if not self._line_collision(key_points[-1], path[i], obstacles):
                            # 如果直接连接不碰撞，则添加该拐点
                            key_points.append(path[i])
                        else:
                            # 如果直接连接碰撞，尝试添加中间点来绕开障碍物
                            mid_point = ((key_points[-1][0] + path[i][0])/2, (key_points[-1][1] + path[i][1])/2)
                            # 检查中间点是否在路径中
                            mid_index = -1
                            for j in range(len(path)):
                                if abs(path[j][0] - mid_point[0]) < 1 and abs(path[j][1] - mid_point[1]) < 1:
                                    mid_index = j
                                    break
                            
                            if mid_index != -1:
                                # 验证两段路径
                                if (not self._line_collision(key_points[-1], path[mid_index], obstacles) and
                                    not self._line_collision(path[mid_index], path[i], obstacles)):
                                    key_points.append(path[mid_index])
                                    key_points.append(path[i])
                            else:
                                # 如果中间点不在路径中，直接添加当前点，后续_validate_segments会处理
                                key_points.append(path[i])
                    else:
                        # 没有障碍物信息，直接添加拐点
                        key_points.append(path[i])
        
        return key_points
    
    def optimize(self, original_path: List[Tuple[float, float]], 
                        obstacles: List = None) -> List[Tuple[float, float]]:
        """
        专门为APF优化路径点
        
        Args:
            original_path: 原始密集路径
            obstacles: 障碍物列表，用于实时验证路径段的可行性
            
        Returns:
            优化后的稀疏路径点
        """
        if not original_path or len(original_path) < 2:
            return original_path
        
        # 提取关键拐点并实时验证碰撞（不再需要后续验证）
        key_points = self.extract_key_points(original_path, obstacles)
        
        print(f"🔍 路径优化完成：{len(original_path)}个点 → {len(key_points)}个关键点")
        
        return key_points
    
    def _validate_segments(self, key_points: List[Tuple[float, float]], obstacles: List) -> List[Tuple[float, float]]:
        """
        验证路径段之间的碰撞情况，并在必要时添加补充点
        """
        if not obstacles or len(key_points) < 2:
            return key_points

        validated_path = [key_points[0]]
        for i in range(1, len(key_points)):
            current_point = validated_path[-1]
            next_point = key_points[i]
            
            # 检查从当前有效点到下一个关键点的路径是否有碰撞
            if not self._line_collision(current_point, next_point, obstacles):
                # 如果无碰撞，直接添加该点
                validated_path.append(next_point)
            else:
                # 如果碰撞，使用更简单可靠的方法：逐个检查中间点
                # 从当前有效点到下一个点之间，尝试添加中间点
                # 首先记录当前有效点的位置
                current_idx = -1
                for j, p in enumerate(key_points):
                    if abs(p[0] - current_point[0]) < 1e-6 and abs(p[1] - current_point[1]) < 1e-6:
                        current_idx = j
                        break
                
                # 逐个检查中间点
                safe_point_found = False
                for j in range(current_idx + 1, i):
                    mid_point = key_points[j]
                    if not self._line_collision(current_point, mid_point, obstacles):
                        # 找到一个安全的中间点
                        validated_path.append(mid_point)
                        safe_point_found = True
                        # 递归检查从中间点到目标点
                        if not self._line_collision(mid_point, next_point, obstacles):
                            validated_path.append(next_point)
                        else:
                            # 如果从中间点到目标点仍有碰撞，继续添加更多中间点
                            for k in range(j + 1, i + 1):
                                if not self._line_collision(validated_path[-1], key_points[k], obstacles):
                                    validated_path.append(key_points[k])
                        break
                
                # 如果没有找到安全的中间点，仍然添加目标点（可能需要进一步处理）
                if not safe_point_found and validated_path[-1] != next_point:
                    validated_path.append(next_point)
        
        return validated_path
    
    def _find_point_in_path(self, point: Tuple[float, float], path: List[Tuple[float, float]]) -> int:
        """
        在路径中查找指定点的索引（考虑浮点数精度）
        """
        for i, p in enumerate(path):
            if abs(p[0] - point[0]) < 1e-6 and abs(p[1] - point[1]) < 1e-6:
                return i
        return -1
    
    def normalize(self, dx, dy):
        """将向量归一化"""
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude == 0:
            return 0, 0
        return dx / magnitude, dy / magnitude
    
    def dist(self, point1, point2):
        """计算两点间的欧氏距离"""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])** 2)
    
    def _line_collision(self, start: Tuple[float, float], end: Tuple[float, float], obstacles: List) -> bool:
        """
        检查线段是否与障碍物碰撞
        支持 pygame.Surface 和 shapely.Polygon
        """
        if not obstacles:
            return False

        # 🎯 shapely.Polygon 情况
        if isinstance(obstacles, list) and isinstance(obstacles[0], Polygon):
            line = LineString([start, end])
            for obs in obstacles:
                if line.intersects(obs):
                    return True
            return False

        # 🎯 pygame.Surface 情况
        if hasattr(obstacles, "get_at"):
            vx, vy = self.normalize(end[0] - start[0], end[1] - start[1])
            curr = list(start)
            step_size = 1  # 像素级步长

            while self.dist(curr, end) > step_size:
                int_curr = int(curr[0]), int(curr[1])
                try:
                    if obstacles.get_at(int_curr) == (0, 0, 0):  # 黑色障碍
                        return True
                except IndexError:
                    pass
                curr[0] += vx * step_size
                curr[1] += vy * step_size

            int_end = int(end[0]), int(end[1])
            try:
                if obstacles.get_at(int_end) == (0, 0, 0):
                    return True
            except IndexError:
                pass

            return False

        # 默认返回不碰撞
        return False
    
    def get_path_statistics(self, original_path: List[Tuple[float, float]], 
                          optimized_path: List[Tuple[float, float]]) -> dict:
        """获取路径优化统计信息"""
        original_length = self._calculate_path_length(original_path)
        optimized_length = self._calculate_path_length(optimized_path)
        
        return {
            'original_points': len(original_path),
            'optimized_points': len(optimized_path),
            'compression_ratio': len(optimized_path) / len(original_path),
            'original_length': original_length,
            'optimized_length': optimized_length,
            'length_change': (optimized_length - original_length) / original_length
        }
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """计算路径总长度"""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(path)):
            segment_length = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            total_length += segment_length
        
        return total_length
    
    def cubic_spline_interpolation(self, path_points: List[Tuple[float, float]], num_interpolated_points: int = 100) -> List[Tuple[float, float]]:
        """
        使用三次样条插值法平滑路径
        
        Args:
            path_points: 原始路径点列表 [(x1,y1), (x2,y2), ...]
            num_interpolated_points: 插值后的总点数
            
        Returns:
            平滑后的路径点列表
        """
        if len(path_points) < 2:
            return path_points
        
        # 将路径点转换为numpy数组
        points = np.array(path_points)
        x = points[:, 0]
        y = points[:, 1]
        
        # 计算每个点到起点的累积距离，作为参数化变量
        distances = np.zeros(len(x))
        for i in range(1, len(x)):
            distances[i] = distances[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])** 2)
        
        # 归一化距离
        t = distances / distances[-1]
        
        # 创建三次样条插值函数
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        
        # 生成新的参数点
        t_new = np.linspace(0, 1, num_interpolated_points)
        
        # 计算插值后的x和y坐标
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)
        
        # 转换回列表格式
        interpolated_path = [(float(x_new[i]), float(y_new[i])) for i in range(len(x_new))]
        
        return interpolated_path
    
    def optimize_with_spline(self, original_path: List[Tuple[float, float]], 
                             num_interpolated_points: int = 100, 
                             obstacles: List = None) -> List[Tuple[float, float]]:
        """
        结合关键拐点提取和三次样条插值进行路径优化
        
        Args:
            original_path: 原始密集路径
            num_interpolated_points: 插值后的总点数
            obstacles: 障碍物列表，用于验证路径段的可行性
            
        Returns:
            优化并平滑后的路径点
        """
        # 1. 提取关键拐点（带碰撞检测）
        key_points = self.extract_key_points(original_path, obstacles)
        
        # 2. 对关键拐点进行三次样条插值，生成平滑路径
        smooth_path = self.cubic_spline_interpolation(key_points, num_interpolated_points)
        
        # 3. 验证路径是否与障碍物碰撞
        if obstacles:
            # 检查平滑后的路径是否与障碍物碰撞
            valid_smooth_path = [smooth_path[0]]
            collision_count = 0
            
            for i in range(1, len(smooth_path)):
                start_point = valid_smooth_path[-1]
                end_point = smooth_path[i]
                
                if not self._line_collision(start_point, end_point, obstacles):
                    valid_smooth_path.append(end_point)
                else:
                    collision_count += 1
            
            # 如果碰撞点太多，考虑保留更多原始关键拐点
            if collision_count > len(smooth_path) * 0.1:  # 如果超过10%的点发生碰撞
                # 增加关键拐点的数量
                # 临时降低最小段长度和角度阈值
                original_min_segment = self.min_segment_length
                original_angle_threshold = self.angle_threshold
                
                self.min_segment_length = max(10.0, original_min_segment * 0.5)
                self.angle_threshold = max(np.radians(10.0), original_angle_threshold * 0.5)
                
                # 重新提取更多关键拐点（带碰撞检测）
                more_key_points = self.extract_key_points(original_path, obstacles)
                
                # 重新插值
                smooth_path = self.cubic_spline_interpolation(more_key_points, num_interpolated_points)
                
                # 恢复原始参数
                self.min_segment_length = original_min_segment
                self.angle_threshold = original_angle_threshold
                
                print(f"⚠️ 检测到碰撞，已重新生成更安全的路径")
            elif collision_count > 0:
                smooth_path = valid_smooth_path
                print(f"⚠️ 已移除 {collision_count} 个碰撞点")
        
        print(f"🔄 三次样条路径优化: {len(original_path)} → {len(smooth_path)} 点")
        
        return smooth_path