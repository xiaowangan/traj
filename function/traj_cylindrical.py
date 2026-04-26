# -*- coding: utf-8 -*-
"""柱面轨迹（凸/凹，轴线沿X或Y，含法向量）"""
import numpy as np
from ._traj_common import generate_raster_rect, generate_spiral_2d


def generate_cylindrical(R, zc=0.0, k_cut=0.0, axis_dir="Y", surf_type="C",
                          axis_min=0.0, axis_max=100.0,
                          proj_shape="R", proj_R=0.0,
                          traj_type="G", direction="X",
                          step_len=1.0, line_spacing=5.0,
                          pitch=5.0, arc_step=1.0):
    if R <= 0:
        raise ValueError("圆柱半径R必须为正数")
    if axis_min >= axis_max:
        raise ValueError("轴线范围参数无效")
    delta_z = k_cut - zc
    if abs(delta_z) > R:
        raise ValueError("切割平面与柱面无交线，请调整 k 或 R")
    d_max = np.sqrt(R ** 2 - delta_z ** 2)
    z0_new = k_cut if surf_type == "C" else (zc - R)

    if axis_dir == "Y":
        y_min_p, y_max_p = axis_min, axis_max
        if proj_shape == "C":
            if proj_R <= 0: raise ValueError("投影圆半径必须为正数")
            eff_R = min(proj_R, d_max)
            x_min_p, x_max_p = -eff_R, eff_R
        else:
            x_min_p, x_max_p = -d_max, d_max
    else:
        x_min_p, x_max_p = axis_min, axis_max
        if proj_shape == "C":
            if proj_R <= 0: raise ValueError("投影圆半径必须为正数")
            eff_R = min(proj_R, d_max)
            y_min_p, y_max_p = -eff_R, eff_R
        else:
            y_min_p, y_max_p = -d_max, d_max

    if traj_type == "G":
        p2d = generate_raster_rect(x_min_p, x_max_p, y_min_p, y_max_p, direction, step_len, line_spacing)
        if proj_shape == "C":
            p2d = [[x, y] for x, y in p2d if x ** 2 + y ** 2 <= eff_R ** 2 + 1e-6]
    else:
        if proj_shape == "C":
            R_sp, xc_sp, yc_sp = eff_R, 0.0, 0.0
        else:
            R_sp = np.hypot((x_max_p - x_min_p) / 2, (y_max_p - y_min_p) / 2)
            xc_sp = (x_min_p + x_max_p) / 2; yc_sp = (y_min_p + y_max_p) / 2
        p2d = generate_spiral_2d(pitch, arc_step, R_sp, xc_sp, yc_sp)
        if proj_shape == "C":
            p2d = [[x, y] for x, y in p2d if x ** 2 + y ** 2 <= eff_R ** 2 + 1e-6]
        else:
            p2d = [[x, y] for x, y in p2d if x_min_p - 1e-6 <= x <= x_max_p + 1e-6 and y_min_p - 1e-6 <= y <= y_max_p + 1e-6]

    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    result = []
    for x, y in p2d:
        d = x if axis_dir == "Y" else y
        if abs(d) > d_max + 1e-6:
            continue
        sq = np.sqrt(max(0.0, R ** 2 - d ** 2))
        z_abs = (zc + sq) if surf_type == "C" else (zc - sq)
        if surf_type == "C" and z_abs < k_cut - 1e-6: continue
        if surf_type == "V" and z_abs > k_cut + 1e-6: continue
        z_rel = z_abs - z0_new
        nx_r = x if axis_dir == "Y" else 0.0
        ny_r = 0.0 if axis_dir == "Y" else y
        nz_r = z_abs - zc
        sign = -1 if surf_type == "V" else 1
        nf = np.hypot(nx_r, np.hypot(ny_r, nz_r))
        if nf < 1e-12: nf = 1.0
        result.append([round(x, 6), round(y, 6), round(z_rel, 6),
                       round(sign * nx_r / nf, 6), round(sign * ny_r / nf, 6), round(sign * nz_r / nf, 6)])
    return result
