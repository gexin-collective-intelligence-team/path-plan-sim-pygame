# -*- coding: utf-8 -*-
# AdaptiveSampling.py
import random
import math

try:
    import numpy as np
except:
    np = None

# —— 工具：检测一个像素是否可通行（非障碍）
def _is_free(surface, x, y):
    ix, iy = int(x), int(y)
    if ix < 0 or iy < 0 or ix >= surface.get_width() or iy >= surface.get_height():
        return False
    return surface.get_at((ix, iy)) != (0, 0, 0)

# —— 工具：在矩形框内“拒绝采样”（采到障碍就重试）
def _sample_in_box_free(surface, x0, y0, x1, y1, max_try=200):
    for _ in range(max_try):
        x = random.uniform(x0, x1)
        y = random.uniform(y0, y1)
        if _is_free(surface, x, y):
            return (x, y)
    return None

# —— 工具：Bridge Test，复杂区很窄时增加命中通道的概率
def _bridge_sample(surface, x0, y0, x1, y1, max_try=200):
    w = x1 - x0
    h = y1 - y0
    for _ in range(max_try):
        ax = x0 + random.random() * w
        ay = y0 + random.random() * h
        bx = x0 + random.random() * w
        by = y0 + random.random() * h
        if (not _is_free(surface, ax, ay)) and (not _is_free(surface, bx, by)):
            mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
            if _is_free(surface, mx, my):
                return (mx, my)
    return None

# —— 计算分块复杂度热力图（0~1），tile_px 可调
def compute_complexity_heatmap(surface, tile_px=40):
    W, H = surface.get_width(), surface.get_height()
    cols = math.ceil(W / tile_px)
    rows = math.ceil(H / tile_px)
    Hmap = [[0.0 for _ in range(cols)] for __ in range(rows)]
    Fmap = [[0.0 for _ in range(cols)] for __ in range(rows)]  # 自由占比，做正规化与兜底

    # 简化指标：以“自由占比 + 边界像素占比”近似复杂度（越拥挤/边界越多 -> 越复杂）
    # 自由占比低 -> 复杂；边界占比高 -> 复杂
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * tile_px, r * tile_px
            x1, y1 = min((c + 1) * tile_px, W), min((r + 1) * tile_px, H)
            area = max(1, (x1 - x0) * (y1 - y0))

            # 采样估计
            free_cnt = 0
            boundary_cnt = 0
            S = max(60, int(area / 50))  # 采样点数
            for _ in range(S):
                x = random.uniform(x0, x1)
                y = random.uniform(y0, y1)
                free = _is_free(surface, x, y)
                if free:
                    free_cnt += 1
                    # 邻域若含障碍像素，认为靠近边界
                    nb = 0
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        if not _is_free(surface, x+dx, y+dy):
                            nb += 1
                    if nb >= 2:
                        boundary_cnt += 1
            free_ratio = free_cnt / float(S)
            boundary_ratio = boundary_cnt / float(S)

            # 复杂度：低自由 + 高边界
            comp = (1.0 - free_ratio) * 0.65 + boundary_ratio * 0.35
            Hmap[r][c] = min(1.0, max(0.0, comp))
            Fmap[r][c] = free_ratio

    # 正规化一下，避免全 0 的图
    flat = [Hmap[r][c] for r in range(rows) for c in range(cols)]
    mx = max(flat) if flat else 1.0
    if mx > 0:
        for r in range(rows):
            for c in range(cols):
                Hmap[r][c] = Hmap[r][c] / mx

    return {
        "heat": Hmap,     # 0~1
        "free": Fmap,     # 0~1
        "rows": rows,
        "cols": cols,
        "tile_px": tile_px,
        "W": W,
        "H": H,
    }

# —— 将总采样数分配到各个 tile：复杂度越高配额越多；每个可通行 tile 至少 min_per_free
def allocate_budgets(heatpack, total_samples, min_per_free=2):
    Hmap = heatpack["heat"]
    Fmap = heatpack["free"]
    rows, cols = heatpack["rows"], heatpack["cols"]

    # 权重 = 0.2（均匀基线） + 0.6*复杂度 + 0.2*(1-自由占比)  —— 更拥挤/复杂的区域权重更大
    weights = []
    for r in range(rows):
        for c in range(cols):
            if Fmap[r][c] < 0.02:            # 几乎全障碍，权重置零
                w = 0.0
            else:
                w = 0.2 + 0.6 * Hmap[r][c] + 0.2 * (1.0 - Fmap[r][c])
            weights.append(w)
    S = sum(weights)
    if S == 0:
        # 全障碍或异常，兜底：均匀分配
        n = rows * cols
        return {(r, c): 0 for r in range(rows) for c in range(cols)}, 0

    # 先按权重分配，再保证“每个可通行 tile 至少 min_per_free”
    raw = [w / S * total_samples for w in weights]
    budgets = {}
    idx = 0
    remain = total_samples
    # 先全部置零
    for r in range(rows):
        for c in range(cols):
            budgets[(r, c)] = 0

    # 先分配地板值
    for r in range(rows):
        for c in range(cols):
            if Fmap[r][c] >= 0.02 and remain > 0:
                give = min(min_per_free, remain)
                budgets[(r, c)] += give
                remain -= give

    # 再按比例分配剩余
    for r in range(rows):
        for c in range(cols):
            if remain <= 0: break
            if Fmap[r][c] < 0.02:
                continue
            add = int(raw[idx])
            idx += 1
            if add <= 0:
                continue
            give = min(add, remain)
            budgets[(r, c)] += give
            remain -= give

    # 把零头补齐
    while remain > 0:
        # 选取当前最复杂的 tile 继续+1
        best = None
        best_score = -1
        for r in range(rows):
            for c in range(cols):
                if Fmap[r][c] < 0.02:
                    continue
                score = 0.7 * Hmap[r][c] + 0.3 * (1.0 - Fmap[r][c])
                if score > best_score:
                    best_score = score
                    best = (r, c)
        if best is None:
            break
        budgets[best] += 1
        remain -= 1

    return budgets, total_samples - remain

# —— 沿 tile 边界投放少量“拼接点”，提升跨 tile 连通性（只在复杂邻接处）
def _stitch_samples(surface, heatpack, budgets, border_k=1):
    Hmap, rows, cols, tp = heatpack["heat"], heatpack["rows"], heatpack["cols"], heatpack["tile_px"]
    pts = []
    for r in range(rows):
        for c in range(cols):
            me = Hmap[r][c]
            # 右邻居
            if c + 1 < cols:
                nb = Hmap[r][c + 1]
                if (me > 0.4 or nb > 0.4):  # 至少有一侧较复杂
                    x = (c + 1) * tp
                    y0 = r * tp
                    y1 = min((r + 1) * tp, heatpack["H"])
                    for _ in range(border_k):
                        cand = _sample_in_box_free(surface, x - 2, y0, x + 2, y1)
                        if cand: pts.append(cand)
            # 下邻居
            if r + 1 < rows:
                nb = Hmap[r + 1][c]
                if (me > 0.4 or nb > 0.4):
                    y = (r + 1) * tp
                    x0 = c * tp
                    x1 = min((c + 1) * tp, heatpack["W"])
                    for _ in range(border_k):
                        cand = _sample_in_box_free(surface, x0, y - 2, x1, y + 2)
                        if cand: pts.append(cand)
    return pts

# —— 主函数：按预算分配进行分区域采样（带 Bridge 与拼接兜底）
def regional_adaptive_sample(surface, total_samples, tile_px=40, min_per_free=2, border_k=1):
    heatpack = compute_complexity_heatmap(surface, tile_px=tile_px)
    budgets, _ = allocate_budgets(heatpack, total_samples, min_per_free=min_per_free)

    W, H = heatpack["W"], heatpack["H"]
    tp = heatpack["tile_px"]
    pts = []

    for (r, c), quota in budgets.items():
        if quota <= 0:
            continue
        x0, y0 = c * tp, r * tp
        x1, y1 = min((c + 1) * tp, W), min((r + 1) * tp, H)

        # 自由占比很低（但非零）→ 优先做 Bridge 采样
        free_ratio = heatpack["free"][r][c]
        use_bridge_first = free_ratio < 0.15

        for _ in range(quota):
            p = None
            if use_bridge_first:
                p = _bridge_sample(surface, x0, y0, x1, y1, max_try=200)
            if p is None:
                p = _sample_in_box_free(surface, x0, y0, x1, y1, max_try=200)
            if p is not None:
                pts.append(p)

    # 加入少量边界拼接点
    pts += _stitch_samples(surface, heatpack, budgets, border_k=border_k)

    # 若因为极端地图导致采样不足，做全局兜底
    need = total_samples - len(pts)
    while need > 0:
        p = _sample_in_box_free(surface, 0, 0, W, H, max_try=400)
        if p:
            pts.append(p)
            need -= 1
        else:
            break

    return pts, heatpack  # 返回点 + 热力图信息供可视化/调试
