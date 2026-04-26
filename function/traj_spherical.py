# -*- coding: utf-8 -*-
"""球面轨迹（凸/凹，栅形/螺旋线，含法向量）"""
import numpy as np
from ._traj_common import generate_raster_rect, generate_spiral_2d


def generate_spherical(R, zc=0.0, surf_type="convex", h=None,
                       traj_type="G", direction="X",
                       step_len=1.0, line_spacing=5.0,
                       pitch=5.0, arc_step=1.0):
    if R <= 0:
        raise ValueError("球体半径R必须为正数")
    if h is None or h <= 0 or h > 2 * R:
        raise ValueError(f"球冠高度 h 必须在 (0, 2R={2*R:.4f}] 范围内")

    if surf_type == "convex":
        z_cut = zc + R - h
        r_proj = np.sqrt(max(0.0, R ** 2 - (z_cut - zc) ** 2))
        z_min_region, z_max_region = z_cut, zc + R
    else:
        z_cut = zc - R
        z_top = z_cut + h
        r_proj = np.sqrt(max(0.0, R ** 2 - (z_top - zc) ** 2))
        z_min_region, z_max_region = z_cut, z_top

    if r_proj < 1e-9:
        raise ValueError("投影圆半径为零，请调整 h 值")

    if traj_type == "G":
        p2d = generate_raster_rect(-r_proj, r_proj, -r_proj, r_proj, direction, step_len, line_spacing)
        p2d = [[x, y] for x, y in p2d if x ** 2 + y ** 2 <= r_proj ** 2 + 1e-6]
    else:
        p2d = generate_spiral_2d(pitch, arc_step, r_proj, 0.0, 0.0)
        p2d = [[x, y] for x, y in p2d if x ** 2 + y ** 2 <= r_proj ** 2 + 1e-6]

    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    result = []
    for x, y in p2d:
        r2 = x * x + y * y
        sq = np.sqrt(max(0.0, R ** 2 - r2))
        z_abs = (zc + sq) if surf_type == "convex" else (zc - sq)
        if not (z_min_region - 1e-6 <= z_abs <= z_max_region + 1e-6):
            continue
        z_rel = z_abs - z_cut
        if surf_type == "convex":
            nx, ny, nz = x / R, y / R, sq / R
        else:
            nx, ny, nz = -x / R, -y / R, -sq / R
        result.append([round(x, 6), round(y, 6), round(z_rel, 6),
                       round(nx, 6), round(ny, 6), round(nz, 6)])
    return result
