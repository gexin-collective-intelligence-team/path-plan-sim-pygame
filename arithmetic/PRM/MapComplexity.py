# -*- coding: utf-8 -*-
"""
MapComplexity.py (COMMENTED EDITION)
===================================
本文件实现对二值/灰度障碍图的“复杂度评估”，用于驱动 APRM 的自适应采样与连边参数。
核心指标：
- ρ (rho): 自由体积分比，越大越开阔；
- ν (nu): 伪距离场(SDF)低于阈值像素占比，反映“狭窄/拥挤”；
- B: 边界复杂度（单位面积的障碍边界长度密度）；
- μv, Var(V): 可见性均值与方差（从网格点向多方向投射的无碰撞射线的归一化长度统计）；
- C: 自由区连通分量个数（越多越碎）；
- p_edge: 用稀疏PRM探针估计的“连边碰撞率”；
- \tilde{B}: 分块异质性，粗分块(如4x4)的自由率方差归一化。

综合评分 S ∈ [0,1]：
    S = α(1-ρ) + βν + γ(1-μv) + δφ(C) + εp_edge + ζ\tilde{B}
并据 S 分档为 open/normal/complex/high，供 APRM 调参使用。

工程注意：
- 所有度量基于同一张“障碍底图”surface，障碍≈接近黑色像素；
- 栅格化 cell 越大越快但越粗糙，常用 6~12；
- 计算量大的指标（可见性/探针PRM）采用采样子集以控制耗时。
"""

# -*- coding: utf-8 -*-
"""
MapComplexity.py
基于截图要求实现的地图复杂度评估：
  ρ: 自由体积分比（开阔度）
  ν: 伪SDF阈下比例（窄通道/拥挤度）
  B: 边界复杂度（障碍物周长密度）
  μv, Var(V): 可见性均值/方差（多方位射线长度比）
  C, p_edge: 自由区连通分量数C + PRM粗探针的碰撞率
  \tilde{B}: 分块异质性（简化成 4x4 子块的占比方差归一）

综合分值：
  S = α(1-ρ) + βν + γ(1-μv) + δφ(C) + εp_edge + ζ\tilde{B}
复杂度档位：
  S < 0.25 开阔; 0.25–0.5 一般; 0.5–0.75 复杂; ≥0.75 高度复杂
"""
import math
import random
from typing import Dict, Tuple, List

import numpy as np
import pygame
from scipy.spatial import KDTree
# --- 将障碍 surface 栅格化成 0/1 网格 ---
# 输入: surface(障碍图), cell(像素栅格尺寸)
# 输出: grid[np.uint8]: 形状 (cols, rows)，1 表示障碍、0 表示自由；以及网格尺寸参数。
# 说明: 以 cell×cell 的块统计是否“存在黑像素”判障碍；加速后续度量。



def _build_grid_from_surface(surface: pygame.Surface, cell: int = 8) -> Tuple[np.ndarray, int, int, int]:
    """把障碍物Surface粗栅格化为0/1网格(1为障碍)，默认障碍是黑色像素。"""
    w, h = surface.get_width(), surface.get_height()
    cols, rows = max(1, w // cell), max(1, h // cell)
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3)
    grid = np.zeros((cols, rows), dtype=np.uint8)
    for cx in range(cols):
        xs = slice(cx * cell, min((cx + 1) * cell, w))
        for cy in range(rows):
            ys = slice(cy * cell, min((cy + 1) * cell, h))
            block = arr[xs, ys, :]
            # 认为(0,0,0)像素为障碍（与原 is_obstacle 保持一致）
            if np.any(np.all(block == 0, axis=2)):
                grid[cx, cy] = 1
    return grid, cols, rows, cell
# --- 自由体积分比 ρ ---
# 计算 grid 中 0 的比例；越大越开阔。
# 返回: ρ ∈ [0,1]



def _free_ratio(grid: np.ndarray) -> float:
    total = grid.size
    occ = grid.sum()
    return float(total - occ) / float(total + 1e-9)


def _kd_on_obstacles(grid: np.ndarray):
    obs = np.argwhere(grid == 1)
    if len(obs) == 0:
        # 没障碍物：虚设一个远点，避免KDTree报错
        obs = np.array([[10**6, 10**6]])
    return KDTree(obs.astype(np.float32)), obs


def _nu_clearance_ratio(grid: np.ndarray, cell: int, r_robot: float = 8.0, k_tau: float = 1.5, n_sample: int = 800) -> Tuple[float, float]:
    """ν = Pr[d(x) < τ], d 用障碍格最近距离近似（欧氏），τ = k * r_robot。"""
    kd, _ = _kd_on_obstacles(grid)
    free = np.argwhere(grid == 0)
    if len(free) == 0:
        return 1.0, 0.0
    if len(free) > n_sample:
        idx = np.random.choice(len(free), n_sample, replace=False)
        free = free[idx]
    d, _ = kd.query(free.astype(np.float32), k=1)
    d_px = d * cell
    tau = k_tau * float(r_robot)
    nu = float(np.mean(d_px < tau))
    avg_d = float(np.mean(d_px)) if len(d_px) else 0.0
    return nu, avg_d
# --- 边界复杂度 B ---
# 方法: 对障碍网格做 4-邻域差分统计边界长度并归一化为单位面积密度；
# 直觉: 障碍轮廓越锯齿/越曲折，B 越大。



def _boundary_complexity(grid: np.ndarray) -> float:
    """B ≈ 边界长度密度：相邻格差分统计（水平+垂直）。归一化到[0,1]"""
    gx = np.sum(grid[:, 1:] != grid[:, :-1])
    gy = np.sum(grid[1:, :] != grid[:-1, :])
    L = gx + gy
    # 最大边界 ~ 2*W*H（锯齿上界），据此粗归一
    max_L = 2.0 * grid.shape[0] * grid.shape[1] + 1e-9
    return float(L / max_L)


# 4-联通栈式泛洪
def _components_count(grid: np.ndarray) -> int:
    visited = np.zeros_like(grid, dtype=bool)
    w, h = grid.shape
    comp = 0
    for sx in range(w):
        for sy in range(h):
            if grid[sx, sy] == 0 and not visited[sx, sy]:
                comp += 1
                stack = [(sx, sy)]
                visited[sx, sy] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < w and 0 <= ny < h and grid[nx, ny] == 0 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
    return comp


def _bresenham_collision(grid: np.ndarray, a: Tuple[int,int], b: Tuple[int,int]) -> bool:
    """栅格直线是否穿越障碍，含端点。"""
    x0, y0 = a; x1, y1 = b
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if not (0 <= x0 < grid.shape[0] and 0 <= y0 < grid.shape[1]):
            return True
        if grid[x0, y0] == 1:
            return True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2*err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return False
# --- 粗PRM探针：连边碰撞率 p_edge ---
# 在随机自由点之间尝试少量直线连边，统计被障碍阻断的比例；
# 反映“局部碰撞风险”/“连边难度”。



def _probe_prm_edge_collision_rate(grid: np.ndarray, k: int = 8, n_pts: int = 200) -> float:
    free = np.argwhere(grid == 0)
    if len(free) < 2:
        return 1.0
    if len(free) > n_pts:
        idx = np.random.choice(len(free), n_pts, replace=False)
        free = free[idx]
    kd = KDTree(free.astype(np.float32))
    total, collided = 0, 0
    for p in free:
        d, idxs = kd.query(p.astype(np.float32), k=min(k+1, len(free)))
        # idxs[0] 是自己
        for j in idxs[1:]:
            q = free[j]
            total += 1
            if _bresenham_collision(grid, (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))):
                collided += 1
    return float(collided / (total + 1e-9))
# --- 可见性指标 μv / Var(V) ---
# 思路: 在若干采样点处，向多方向射线推进直到撞到障碍，记录无碰撞长度；
# 将长度归一到 [0,1] 后求均值与方差。
# 提示: 采样数量与方向数越大越稳定但越慢，可根据分辨率调节。



def _visibility_metrics(grid: np.ndarray, dirs: int = 16, M: int = 50) -> Tuple[float, float]:
    """从随机M个自由格出发，沿dirs个方向投射，长度比值均值 μv 与方差。"""
    free = np.argwhere(grid == 0)
    if len(free) == 0:
        return 0.0, 0.0
    if len(free) > M:
        free = free[np.random.choice(len(free), M, replace=False)]
    angles = [2*math.pi*i/dirs for i in range(dirs)]
    max_r = math.hypot(grid.shape[0], grid.shape[1])  # 以对角线为最大视距
    ratios: List[float] = []
    for x0, y0 in free:
        for th in angles:
            x, y = float(x0), float(y0)
            dx, dy = math.cos(th), math.sin(th)
            step = 0.0
            while 0 <= int(x) < grid.shape[0] and 0 <= int(y) < grid.shape[1] and grid[int(x), int(y)] == 0:
                x += dx; y += dy; step += 1.0
                if step > max_r: break
            ratios.append(float(step / (max_r + 1e-9)))
    mu = float(np.mean(ratios)) if ratios else 0.0
    var = float(np.var(ratios)) if ratios else 0.0
    return mu, var


def _heterogeneity(grid: np.ndarray, blocks: int = 4) -> float:
    """把网格切 blocks×blocks 统计各块障碍占比的方差并归一化到[0,1]"""
    w, h = grid.shape
    vs = []
    for i in range(blocks):
        for j in range(blocks):
            xs = slice(int(i*w/blocks), int((i+1)*w/blocks))
            ys = slice(int(j*h/blocks), int((j+1)*h/blocks))
            sub = grid[xs, ys]
            occ = float(sub.sum()) / float(sub.size + 1e-9)
            vs.append(occ)
    var = float(np.var(vs)) if vs else 0.0
    # 最大方差理论上限 < 0.25，这里粗归一
    return min(1.0, var / 0.25)
# --- 综合复杂度评估 analyze_map ---
# 输入: surface(障碍底图), cell(粗栅格像素), r_robot(机器人半径像素, 影响保守性)
# 流程: 栅格化 → 逐项指标 → 归一化加权 → 档位划分
# 输出: dict，包括所有中间量与 S、level，可直接被 Aprm.plan() 使用。
# 调参: α..ζ 权重可按任务偏好修改，例如强调窄通道可增大 β。



def analyze_map(mapdata) -> Dict:
    """外部主入口：传入现有的 mapdata（即 PygameWidget 实例）"""
    cell = getattr(mapdata, "cell_size", 8)
    grid, cols, rows, cell = _build_grid_from_surface(mapdata.obs_surface, cell)

    rho = _free_ratio(grid)
    r_robot = getattr(mapdata, "ship_radius", 8.0)
    nu, avg_d = _nu_clearance_ratio(grid, cell, r_robot=r_robot, k_tau=1.5, n_sample=800)
    B = _boundary_complexity(grid)
    mu_v, var_v = _visibility_metrics(grid, dirs=16, M=50)
    C = _components_count(grid)
    p_edge = _probe_prm_edge_collision_rate(grid, k=8, n_pts=220)
    B_tilde = _heterogeneity(grid, blocks=4)

    # 经验权重：β,γ,ε ~ 0.2–0.25；其余 0.1–0.15
    α, β, γ, δ, ε, ζ = 0.15, 0.22, 0.20, 0.12, 0.20, 0.11
    Cmax = 8.0
    phiC = min(1.0, (max(1, C) - 1) / Cmax)

    S = (α*(1.0 - rho) +
         β*nu +
         γ*(1.0 - mu_v) +
         δ*phiC +
         ε*p_edge +
         ζ*B_tilde)
    S = float(max(0.0, min(1.0, S)))

    if S < 0.25: level = "open"         # 开阔
    elif S < 0.50: level = "normal"     # 一般
    elif S < 0.75: level = "complex"    # 复杂
    else: level = "high"                # 高度复杂

    return dict(
        grid=grid, cell=cell, cols=cols, rows=rows,
        rho=rho, nu=nu, B=B, mu_v=mu_v, var_v=var_v, C=C, p_edge=p_edge, B_tilde=B_tilde,
        S=S, level=level, r_robot=r_robot
    )
