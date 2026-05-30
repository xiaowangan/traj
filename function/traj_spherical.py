# -*- coding: utf-8 -*-
"""球面轨迹（凸/凹，栅形/螺旋线，含法向量）"""
import numpy as np
from ._traj_common import generate_spiral_2d


def _angle_values(start, stop, step):
    vals = np.arange(start, stop + 1e-12, step)
    if len(vals) == 0:
        return np.array([start, stop], dtype=float)
    if vals[-1] < stop - step * 0.01:
        vals = np.append(vals, stop)
    return vals


def _generate_spherical_raster(R, r_proj, direction, step_len, line_spacing):
    """按球面弧长生成栅形投影点，避免投影平面栅格在边界处产生大缺口。"""
    u_max = np.arcsin(min(1.0, r_proj / R))
    du = line_spacing / R
    rows = _angle_values(-u_max, u_max, du)
    points = []
    for idx, u in enumerate(rows):
        fixed = R * np.sin(u)
        rho = max(0.0, R * np.cos(u))
        limit = np.sqrt(max(0.0, r_proj * r_proj - fixed * fixed))
        if rho < 1e-9 or limit < 1e-9:
            moving_vals = np.array([0.0])
        else:
            beta_max = np.arcsin(min(1.0, limit / rho))
            db = step_len / rho
            beta_vals = _angle_values(-beta_max, beta_max, db)
            moving_vals = rho * np.sin(beta_vals)
        if idx % 2:
            moving_vals = moving_vals[::-1]
        for moving in moving_vals:
            if direction == "X":
                points.append([float(moving), float(fixed)])
            else:
                points.append([float(fixed), float(moving)])
    return points


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
        p2d = _generate_spherical_raster(R, r_proj, direction, step_len, line_spacing)
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
