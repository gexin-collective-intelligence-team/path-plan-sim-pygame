# -*- coding: utf-8 -*-
import math
import time
import random
import copy
import pygame
from PyQt5.QtWidgets import QApplication
from scipy.spatial import KDTree

from .Node import point
from .MapComplexity import analyze_map


# ==== 分区域自适应采样（内置到 Aprm.py，无需外部文件） ====
def _is_free_px(surface, x, y):
    """Return True if pixel at (x,y) is free (non-obstacle).
    Treat (near) black as obstacle; robust to float coords and bounds."""
    W, H = surface.get_width(), surface.get_height()
    ix, iy = int(round(x)), int(round(y))
    if ix < 0 or iy < 0 or ix >= W or iy >= H:
        return False
    r, g, b, *rest = surface.get_at((ix, iy))
    # obstacle ~ dark pixel; threshold can be tuned
    return not (r < 10 and g < 10 and b < 10)
    return surface.get_at((ix, iy)) != (0, 0, 0)


def _is_segment_free(surface, ax, ay, bx, by, inflate=0):
    """Bresenham-style grid walk with optional obstacle inflation (Manhattan radius).
    Returns True if all traversed pixels are free."""
    W, H = surface.get_width(), surface.get_height()
    x0, y0 = int(round(ax)), int(round(ay))
    x1, y1 = int(round(bx)), int(round(by))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    def free_with_inflate(ix, iy):
        if inflate <= 0:
            return _is_free_px(surface, ix, iy)
        for ox in range(-inflate, inflate + 1):
            for oy in range(-inflate, inflate + 1):
                if abs(ox) + abs(oy) > inflate:
                    continue
                jx, jy = ix + ox, iy + oy
                if jx < 0 or jy < 0 or jx >= W or jy >= H:
                    return False
                r, g, b, *rest = surface.get_at((jx, jy))
                if r < 10 and g < 10 and b < 10:
                    return False
        return True

    while True:
        if not free_with_inflate(x0, y0):
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True


def _sample_in_box_free(surface, x0, y0, x1, y1, max_try=200):
    for _ in range(max_try):
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        if _is_free_px(surface, x, y):
            return (x, y)
    return None


def _bridge_sample_box(surface, x0, y0, x1, y1, max_try=200):
    w = max(1.0, x1 - x0);
    h = max(1.0, y1 - y0)
    for _ in range(max_try):
        ax = x0 + random.random() * w
        ay = y0 + random.random() * h
        bx = x0 + random.random() * w
        by = y0 + random.random() * h
        if (not _is_free_px(surface, ax, ay)) and (not _is_free_px(surface, bx, by)):
            mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
            if _is_free_px(surface, mx, my):
                return (mx, my)
    return None


def _compute_complexity_heatmap(surface, tile_px=40):
    W, H = surface.get_width(), surface.get_height()
    cols = int(math.ceil(W / float(tile_px)))
    rows = int(math.ceil(H / float(tile_px)))
    Hmap = [[0.0 for _ in range(cols)] for __ in range(rows)]
    Fmap = [[0.0 for _ in range(cols)] for __ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_px, r * tile_px
            x1, y1 = min((c + 1) * tile_px, W), min((r + 1) * tile_px, H)
            area = max(1, (x1 - x0) * (y1 - y0))
            free_cnt = 0
            boundary_cnt = 0
            S = max(60, int(area / 50))
            for _ in range(S):
                x = random.uniform(x0, x1)
                y = random.uniform(y0, y1)
                free = _is_free_px(surface, x, y)
                if free:
                    free_cnt += 1
                    nb = 0
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if not _is_free_px(surface, x + dx, y + dy):
                            nb += 1
                    if nb >= 2:
                        boundary_cnt += 1
            free_ratio = free_cnt / float(S)
            boundary_ratio = boundary_cnt / float(S)
            comp = (1.0 - free_ratio) * 0.65 + boundary_ratio * 0.35
            Hmap[r][c] = min(1.0, max(0.0, comp))
            Fmap[r][c] = free_ratio
    flat = [Hmap[r][c] for r in range(rows) for c in range(cols)]
    mx = max(flat) if flat else 1.0
    if mx > 0:
        for r in range(rows):
            for c in range(cols):
                Hmap[r][c] = Hmap[r][c] / mx
    return {"heat": Hmap, "free": Fmap, "rows": rows, "cols": cols, "tile_px": tile_px, "W": W, "H": H}


def _allocate_budgets(heatpack, total_samples, min_per_free=2):
    Hmap = heatpack["heat"];
    Fmap = heatpack["free"]
    rows, cols = heatpack["rows"], heatpack["cols"]
    weights = []
    for r in range(rows):
        for c in range(cols):
            if Fmap[r][c] < 0.02:
                w = 0.0
            else:
                w = 0.2 + 0.6 * Hmap[r][c] + 0.2 * (1.0 - Fmap[r][c])
            weights.append(w)
    S = sum(weights)
    if S == 0:
        return {(r, c): 0 for r in range(rows) for c in range(cols)}, 0
    raw = [w / S * total_samples for w in weights]
    budgets = {(r, c): 0 for r in range(rows) for c in range(cols)}
    remain = total_samples
    # 保底
    for r in range(rows):
        for c in range(cols):
            if Fmap[r][c] >= 0.02 and remain > 0:
                give = min(min_per_free, remain)
                budgets[(r, c)] += give;
                remain -= give
    # 按比例
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if remain <= 0: break
            if Fmap[r][c] < 0.02:
                idx += 1
                continue
            add = int(raw[idx]);
            idx += 1
            if add <= 0:
                continue
            give = min(add, remain)
            budgets[(r, c)] += give;
            remain -= give
    # 零头加到最复杂
    while remain > 0:
        best = None;
        best_score = -1
        for r in range(rows):
            for c in range(cols):
                if Fmap[r][c] < 0.02:
                    continue
                score = 0.7 * Hmap[r][c] + 0.3 * (1.0 - Fmap[r][c])
                if score > best_score:
                    best_score = score;
                    best = (r, c)
        if best is None: break
        budgets[best] += 1;
        remain -= 1
    return budgets, total_samples - remain


def _stitch_samples(surface, heatpack, budgets, border_k=1):
    Hmap, rows, cols, tp = heatpack["heat"], heatpack["rows"], heatpack["cols"], heatpack["tile_px"]
    pts = []
    for r in range(rows):
        for c in range(cols):
            me = Hmap[r][c]
            if c + 1 < cols:
                nb = Hmap[r][c + 1]
                if (me > 0.4 or nb > 0.4):
                    x = (c + 1) * tp
                    y0 = r * tp;
                    y1 = min((r + 1) * tp, heatpack["H"])
                    for _ in range(border_k):
                        cand = _sample_in_box_free(surface, x - 2, y0, x + 2, y1)
                        if cand: pts.append(cand)
            if r + 1 < rows:
                nb = Hmap[r + 1][c]
                if (me > 0.4 or nb > 0.4):
                    y = (r + 1) * tp
                    x0 = c * tp;
                    x1 = min((c + 1) * tp, heatpack["W"])
                    for _ in range(border_k):
                        cand = _sample_in_box_free(surface, x0, y - 2, x1, y + 2)
                        if cand: pts.append(cand)
    return pts


def regional_adaptive_sample(surface, total_samples, tile_px=40, min_per_free=2, border_k=1):
    heatpack = _compute_complexity_heatmap(surface, tile_px=tile_px)
    budgets, _ = _allocate_budgets(heatpack, total_samples, min_per_free=min_per_free)
    W, H = heatpack["W"], heatpack["H"];
    tp = heatpack["tile_px"]
    pts = []
    for (r, c), quota in budgets.items():
        if quota <= 0:
            continue
        x0, y0 = c * tp, r * tp
        x1, y1 = min((c + 1) * tp, W), min((r + 1) * tp, H)
        free_ratio = heatpack["free"][r][c]
        use_bridge_first = free_ratio < 0.15
        for _ in range(quota):
            p = None
            if use_bridge_first:
                p = _bridge_sample_box(surface, x0, y0, x1, y1, max_try=200)
            if p is None:
                p = _sample_in_box_free(surface, x0, y0, x1, y1, max_try=200)
            if p is not None:
                pts.append(p)
    pts += _stitch_samples(surface, heatpack, budgets, border_k=border_k)
    need = total_samples - len(pts)
    while need > 0:
        p = _sample_in_box_free(surface, 0, 0, W, H, max_try=400)
        if p:
            pts.append(p);
            need -= 1
        else:
            break
    return pts, heatpack


class prm:
    def __init__(self, mapdata):
        self.map = mapdata
        self.start = point(mapdata.start_point[0], mapdata.start_point[1])
        self.end = point(mapdata.end_point[0], mapdata.end_point[1])

        self.result = []
        self.width = mapdata.width
        self.height = mapdata.height

        # 注意：原 is_obstacle 依赖 obs_surface 黑像素判断
        self.obstacle = mapdata.obs_surface

        # 以下参数将被复杂度自适应覆写
        self.max_point_num = 1000
        self.nodes = [self.start, self.end]
        self.edges = []
        self.step = 40  # 连边半径（像素）

        # —— 分区域自适应采样参数（可从GUI调整）
        self.tile_px = 40  # 单瓦片像素边长，复杂度网格粒度
        self.min_per_free = 2  # 每个自由瓦片的最低采样数
        self.border_k = 1  # 邻接瓦片公共边拼接点密度
        self.debug_draw_heat = True  # 调试：是否在右侧绘制复杂度热力图

        # 采样/探测缓存
        self._comp = None
        self._obs_kd = None
        self._obs_cells = None

    # ---------------- 基本几何与碰撞 ----------------
    def dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def collision(self, src, dst):
        """Return True if the straight line src->dst collides with obstacle surface."""
        # Inflate by 1 pixel to account for robot footprint / aliasing
        ok = _is_segment_free(self.obstacle, src.x, src.y, dst.x, dst.y, inflate=1)
        return (not ok)

    def normalize(self, vx, vy):
        norm = math.hypot(vx, vy)
        if norm > 1e-6:
            return vx / norm, vy / norm
        return 0.0, 0.0

    # ---------------- 路径平滑（两步：直连捷径 + Chaikin） ----------------
    def _angle_rad(self, a, b, c):
        """计算∠ABC（弧度）。a,b,c 为 (x,y)。"""
        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return math.acos(cosv)

    def _curvature_ok(self, p0, p1, p2, kappa_max=0.25):
        """离散曲率近似：κ≈转角/相邻段较大长度；控制转弯不过猛。"""
        d1 = max(1e-6, math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        d2 = max(1e-6, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        ang = self._angle_rad(p0, p1, p2)
        kappa = ang / max(d1, d2)
        return kappa <= kappa_max

    def _los(self, a, b):
        """Line-Of-Sight：使用障碍底图检测 a→b 是否无遮挡（含1像素膨胀）。"""
        return _is_segment_free(self.obstacle, a[0], a[1], b[0], b[1], inflate=1)

    def smooth_shortcut(self, pts, max_iter=600, kappa_max=0.25):
        """随机-贪心混合的直连平滑：
        - 一次随机挑选 (i, j), i+1<j，若 a=pts[i], b=pts[j] 之间无遮挡且曲率满足，就删掉中间拐点。
        - 重复 max_iter 次，线性时间可控，效果显著。
        - 输入/输出：[(x,y), ...]
        """
        if len(pts) <= 2:
            return pts[:]
        import random
        P = pts[:]
        n_iter = max(0, int(max_iter))
        for _ in range(n_iter):
            if len(P) <= 2:
                break
            i = random.randint(0, len(P) - 3)
            j = random.randint(i + 2, len(P) - 1)
            a, b = P[i], P[j]
            if not self._los(a, b):
                continue
            ok = True
            if i - 1 >= 0:
                ok &= self._curvature_ok(P[i - 1], a, b, kappa_max)
            if j + 1 < len(P):
                ok &= self._curvature_ok(a, b, P[j + 1], kappa_max)
            if ok:
                P = P[:i + 1] + [b] + P[j + 1:]
        return P

    def chaikin_once(self, pts, alpha=0.25):
        """一次 Chaikin 角切。保持端点，生成更圆润的折线。"""
        if len(pts) < 2:
            return pts[:]
        Q = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            Q.append(((1 - alpha) * p[0] + alpha * q[0], (1 - alpha) * p[1] + alpha * q[1]))
            Q.append((alpha * p[0] + (1 - alpha) * q[0], alpha * p[1] + (1 - alpha) * q[1]))
        Q.append(pts[-1])
        return Q

    def _polyline_collision_free(self, pts):
        """逐段检查折线是否无碰撞。"""
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if not _is_segment_free(self.obstacle, a[0], a[1], b[0], b[1], inflate=1):
                return False
        return True

    def chaikin_smooth_with_check(self, pts, passes=1, alpha=0.25):
        """进行 1~k 次 Chaikin 角切，并在每次后做碰撞校验，若发生穿障则回退停止。"""
        P = pts[:]
        k = max(0, int(passes))
        for _ in range(k):
            C = self.chaikin_once(P, alpha=alpha)
            if self._polyline_collision_free(C):
                P = C
            else:
                break
        return P

    # ---------------- 采样方法 ----------------
    @staticmethod
    def _halton(index, base):
        """Halton 低差异序列一维"""
        f, r = 1.0, 0.0
        i = index
        while i > 0:
            f = f / base
            r = r + f * (i % base)
            i = i // base
        return r

    # ---------------- 角点圆滑（Quadratic Bézier 圆角） ----------------
    def _quad_bezier(self, p0, p1, p2, t):
        """二次B样条曲线 p(t) = (1-t)^2 p0 + 2(1-t)t p1 + t^2 p2"""
        u = 1.0 - t
        return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])

    def _try_round_corner(self, a, b, c, r_px, samples=6):
        """尝试把拐点 b 圆滑为二次贝塞尔片段。
        返回: (ok, [p0,...,pk])，ok=True 表示生成的曲线段无碰撞。
        说明: 取 ab, bc 上距离 b 为 s 的点作为端点，b 为控制点，s 基于 r_px 自动截断。
        """
        import math
        # 向量与长度
        v1 = (a[0] - b[0], a[1] - b[1]);
        d1 = math.hypot(v1[0], v1[1])
        v2 = (c[0] - b[0], c[1] - b[1]);
        d2 = math.hypot(v2[0], v2[1])
        if d1 < 1e-6 or d2 < 1e-6:
            return False, []
        # 单位方向
        u1 = (v1[0] / d1, v1[1] / d1)
        u2 = (v2[0] / d2, v2[1] / d2)
        # 取偏移距离 s，避免超过半段长度
        s_max = max(4.0, min(r_px, 0.35 * min(d1, d2)))
        s = s_max
        # 试着从大到小递减，直到无碰撞
        while s >= 4.0:
            p0 = (b[0] + u1[0] * s, b[1] + u1[1] * s)  # 在 ab 上、靠近 b 的点
            p2 = (b[0] + u2[0] * s, b[1] + u2[1] * s)  # 在 bc 上、靠近 b 的点
            # 采样曲线
            curve = [self._quad_bezier(p0, b, p2, t / float(samples)) for t in range(0, samples + 1)]
            # 检查段段无碰撞
            ok = True
            for i in range(len(curve) - 1):
                x0, y0 = curve[i];
                x1, y1 = curve[i + 1]
                if not _is_segment_free(self.obstacle, x0, y0, x1, y1, inflate=1):
                    ok = False;
                    break
            if ok:
                return True, curve
            s *= 0.6  # 缩小半径再试
        return False, []

    def round_corners_bezier_with_check(self, pts, r_px=12, samples=6):
        """对整条折线进行角点圆滑（二次贝塞尔），保持端点不动。
        - r_px: 目标圆角半径（像素），会按段长自动截断；
        - samples: 曲线离散的采样点数（越大越圆滑，耗时也稍增）。
        - 返回: 新的点列（折线），若某角无法圆滑则保留原角点。
        """
        if len(pts) <= 2:
            return pts[:]
        new_pts = [pts[0]]
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            ok, curve = self._try_round_corner(a, b, c, r_px=r_px, samples=samples)
            if ok:
                # 用曲线替换拐点：连接上一点到curve首点的直线也需可行
                if _is_segment_free(self.obstacle, new_pts[-1][0], new_pts[-1][1], curve[0][0], curve[0][1], inflate=1):
                    new_pts += curve  # curve含首尾点
                else:
                    new_pts.append(b)
            else:
                new_pts.append(b)
        new_pts.append(pts[-1])
        return new_pts

    def sample_halton(self, i):
        """二维Halton -> 屏幕像素"""
        x = self._halton(i, 2) * self.width
        y = self._halton(i, 3) * self.height
        return point(x, y)

    def _build_obs_kd(self):
        if self._comp is None:
            self._comp = analyze_map(self.map)
        grid = self._comp["grid"]
        obs = (grid == 1)
        obs_idx = [(int(x), int(y)) for x, y in zip(*obs.nonzero())]
        if not obs_idx:
            obs_idx = [(10 ** 6, 10 ** 6)]
        self._obs_cells = obs_idx
        self._obs_kd = KDTree([[x, y] for x, y in obs_idx])

    def _cell_from_xy(self, x, y):
        c = self._comp["cell"]
        return int(x // c), int(y // c)

    def _dist_to_obstacle_px(self, x, y):
        """返回像素级近障距离（用栅格KDTree近似）"""
        if self._obs_kd is None:
            self._build_obs_kd()
        cx, cy = self._cell_from_xy(x, y)
        d, _ = self._obs_kd.query([cx, cy], k=1)
        return float(d * self._comp["cell"])

    def sample_near_obstacle(self, tau_px, max_try=50):
        """靠障样本：d(x) < τ"""
        for _ in range(max_try):
            x = random.random() * self.width
            y = random.random() * self.height
            if self._dist_to_obstacle_px(x, y) < tau_px:
                return point(x, y)
        # 退化：随便给一个
        return point(random.random() * self.width, random.random() * self.height)

    def sample_bridge(self, max_try=60):
        """
        Bridge Test：随机取两个障碍内点，若中点在自由区则纳入。
        这里用像素判断(0,0,0)为障碍。找障碍点用随机击中法。
        """
        for _ in range(max_try):
            ax = random.random() * self.width
            ay = random.random() * self.height
            bx = random.random() * self.width
            by = random.random() * self.height
            if self.obstacle.get_at((int(ax), int(ay))) == (0, 0, 0) and \
                    self.obstacle.get_at((int(bx), int(by))) == (0, 0, 0):
                mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
                if self.obstacle.get_at((int(mx), int(my))) != (0, 0, 0):
                    return point(mx, my)
        # 退化
        return point(random.random() * self.width, random.random() * self.height)

    def generate_random_point(self):
        return point(random.random() * self.width, random.random() * self.height)

    # ---------------- 连边与搜索 ----------------

    def _local_free_ratio(self, p):
        hp = getattr(self, '_heatpack', None)
        if not hp:
            return 1.0
        tp = hp['tile_px']
        r = int(max(0, min(hp['rows'] - 1, int(p.y // tp))))
        c = int(max(0, min(hp['cols'] - 1, int(p.x // tp))))
        return float(hp['free'][r][c])

    def connect_nodes(self):
        # 自适应半径 + 节点度上限 + 懒验证（Lazy-PRM）
        coords = [(n.x, n.y) for n in self.nodes]
        kdtree = KDTree(coords)
        self._edge_cache = {}  # (ida,idb) -> True/False (是否已验证为可通过)
        self.edges = []  # 候选边（未验证）。仅存一次 (a,b)
        seen = set()

        N = len(self.nodes)
        base_r = float(self.step)
        for i, node in enumerate(self.nodes):
            free = self._local_free_ratio(node)
            # 局部半径：稀疏区更大，密集区更小
            r_local = base_r * (0.7 + 0.6 * max(0.0, min(1.0, free)))
            # 度上限：密集区上限略高，开阔区更低
            deg_cap = 6 if free < 0.5 else 4
            # 预取最近若干个候选（避免 query_ball_point 返回过大集合）
            k_init = min(N, max(8, deg_cap * 4))
            dists, idxs = kdtree.query((node.x, node.y), k=k_init)
            # 统一为列表
            if k_init == 1:
                dists = [dists];
                idxs = [idxs]
            # 去掉自身并按距离排序
            pairs = [(d, j) for d, j in zip(dists, idxs) if j != i and d <= r_local]
            pairs.sort(key=lambda t: t[0])
            added = 0
            for _, j in pairs:
                if added >= deg_cap:
                    break
                a, b = (i, j) if i < j else (j, i)
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                self.edges.append((self.nodes[a], self.nodes[b]))
                added += 1

    def get_neighbors(self, node):
        # 懒验证：只有在取邻居时才做碰撞检测，并缓存结果
        nb = []
        if not hasattr(self, '_node_index'):
            self._node_index = {n: idx for idx, n in enumerate(self.nodes)}
        i = self._node_index[node]
        for a, b in self.edges:
            if a is node:
                jnode = b
            elif b is node:
                jnode = a
            else:
                continue
            j = self._node_index[jnode]
            key = (i, j) if i < j else (j, i)
            ok = self._edge_cache.get(key)
            if ok is None:
                ok = not self.collision(node, jnode)
                self._edge_cache[key] = ok
            if ok:
                nb.append(jnode)
        return nb

    def a_star(self):
        open_set = {self.start}
        came_from = {}
        g_score = {n: float('inf') for n in self.nodes}
        f_score = {n: float('inf') for n in self.nodes}
        g_score[self.start] = 0.0
        f_score[self.start] = self.dist(self.start, self.end)
        while open_set:
            current = min(open_set, key=lambda n: f_score[n])
            if current == self.end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            open_set.remove(current)
            for nb in self.get_neighbors(current):
                tg = g_score[current] + self.dist(current, nb)
                if tg < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tg
                    f_score[nb] = tg + self.dist(nb, self.end)
                    open_set.add(nb)
        return []

    # ---------------- 主流程：自适应采样与连边 ----------------
    def _compute_radius(self, n, eta=1.6, scale=0.30):
        # r(n) = η (log n / n)^(1/2) * (scale * domain_diag)
        diag = math.hypot(self.width, self.height)
        base = scale * diag
        rn = eta * (max(1.0, math.log(max(2, n))) / max(2.0, float(n))) ** 0.5
        return max(10.0, rn * base)

    def plan(self, plan_surface: pygame.Surface):
        st = time.time()

        # === 1) 地图复杂度评估 ===
        self._comp = analyze_map(self.map)
        level = self._comp["level"]
        S = self._comp["S"];
        nu = self._comp["nu"];
        rho = self._comp["rho"]
        tau_px = 1.5 * float(self._comp["r_robot"])

        # === 2) 档位 -> 采样/连边参数 ===
        # 配比：Halton / 靠障 / Bridge
        if level == "open":
            self.max_point_num = 700
            mix = (0.8, 0.15, 0.05)
            eta = 1.8
        elif level == "normal":
            self.max_point_num = 1100
            mix = (0.55, 0.30, 0.15)
            eta = 1.7
        elif level == "complex":
            self.max_point_num = 1600
            mix = (0.30, 0.45, 0.25)  # 窄通道更偏桥采+靠障
            eta = 1.6
        else:  # high
            self.max_point_num = 2200
            mix = (0.20, 0.55, 0.25)
            eta = 1.5

        # 控制靠障权重再受 ν 的一次缩放（ν大→更挤→靠障更多）
        halton_w, near_w, bridge_w = mix
        near_w = min(0.85, near_w * (1.0 + 0.8 * nu))
        # 归一
        s = halton_w + near_w + bridge_w
        halton_w, near_w, bridge_w = halton_w / s, near_w / s, bridge_w / s

        print(f"[MapComplexity] S={S:.3f}, level={level}, rho={rho:.2f}, nu={nu:.2f}")
        print(f"[SamplerMix] Halton={halton_w:.2f}, NearObs={near_w:.2f}, Bridge={bridge_w:.2f}")

        # === 3) 生成节点（分区域自适应采样） ===
        # 使用分块复杂度热力图为每个瓦片分配采样配额：复杂区多采，开阔区少采
        pts, heatpack = regional_adaptive_sample(
            self.obstacle,
            total_samples=self.max_point_num,
            tile_px=self.tile_px,
            min_per_free=self.min_per_free,
            border_k=self.border_k
        )
        self._heatpack = heatpack
        # 可选：调试热力图叠加
        if getattr(self, 'debug_draw_heat', False):
            try:
                self._draw_heatmap(plan_surface, heatpack)
            except Exception as _:
                pass

        self.nodes = []  # 覆盖旧的 nodes
        for (x, y) in pts:
            p = point(x, y)
            # 只要是自由区像素就加入
            if _is_free_px(self.obstacle, p.x, p.y):
                self.nodes.append(p)
                pygame.draw.circle(plan_surface, (0, 100, 255), (int(p.x), int(p.y)), 2)
                QApplication.processEvents()

        # 确保起终点在集合内
        if self.start not in self.nodes: self.nodes.append(self.start)
        if self.end not in self.nodes: self.nodes.append(self.end)

        # === 4) 自适应连边半径 r(n) ===
        self.step = self._compute_radius(len(self.nodes), eta=eta)
        # 适度再放大，ν大→密集→半径略增，保证连通；开阔则半径略减
        self.step *= (0.9 + 0.6 * (0.5 + nu - rho * 0.3))
        self.step = max(15.0, min(self.step, 120.0))
        print(f"[ConnectRadius] step={self.step:.1f} px, n={len(self.nodes)}")

        # === 5) 连边/求解 ===
        self.connect_nodes()
        for a, b in self.edges:
            if _is_segment_free(self.obstacle, a.x, a.y, b.x, b.y, inflate=1):
                pygame.draw.line(plan_surface, (0, 255, 0), (int(a.x), int(a.y)), (int(b.x), int(b.y)), 1)

        path = self.a_star()
        # --- 路径平滑：LOS捷径 + Chaikin(1次) ---
        if path:
            pts = [(p.x, p.y) for p in path]
            pts = self.smooth_shortcut(pts, max_iter=600, kappa_max=0.25)
            pts = self.chaikin_smooth_with_check(pts, passes=1, alpha=0.25)
            # 角点圆滑（二次贝塞尔），进一步消除钝角
            pts = self.round_corners_bezier_with_check(pts, r_px=12, samples=6)
            path = [point(x, y) for (x, y) in pts]
        et = time.time()
        print("规划用时：", et - st, "s")
        if not path:
            print("未找到可行路径！")

        # 红线绘制
        for k in range(len(path) - 1):
            pygame.draw.line(plan_surface, (255, 0, 0),
                             (int(path[k].x), int(path[k].y)),
                             (int(path[k + 1].x), int(path[k + 1].y)), 2)
        return path, et - st
