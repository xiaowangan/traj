# -*- coding: utf-8 -*-
"""柱面轨迹（凸/凹，轴线沿X或Y，含法向量）"""
import numpy as np
from ._traj_common import generate_surface_raster, generate_spiral_2d


CYLINDRICAL_WALL_THICKNESS_MM = 0.5


def generate_cylindrical(R, zc=0.0, k_cut=0.0, axis_dir="Y", surf_type="C",
                          axis_min=0.0, axis_max=100.0,
                          proj_shape="R", proj_R=0.0,
                          traj_type="G", direction="X",
                          step_len=1.0, line_spacing=5.0,
                          pitch=5.0, arc_step=1.0,
                          cover_type=None,
                          rect_xmin=0.0, rect_xmax=0.0,
                          rect_ymin=0.0, rect_ymax=0.0,
                          circ_R=0.0, circ_xc=0.0, circ_yc=0.0,
                          wall_thickness=CYLINDRICAL_WALL_THICKNESS_MM):
    if R <= 0:
        raise ValueError("圆柱半径R必须为正数")
    if axis_min >= axis_max:
        raise ValueError("轴线范围参数无效")
    delta_z = k_cut - zc
    if abs(delta_z) > R:
        raise ValueError("切割平面与柱面无交线，请调整 k 或 R")
    work_R = float(R)
    if surf_type == "V":
        if wall_thickness <= 0 or wall_thickness >= R:
            raise ValueError("凹柱面固定厚度必须满足 0 < t < R")
        work_R = float(R - wall_thickness)
        if abs(delta_z) >= work_R - 1e-12:
            raise ValueError(
                f"凹柱面厚度 t={wall_thickness:.4f} mm 时，切割平面必须与内加工柱面相交")
    d_max = np.sqrt(work_R ** 2 - delta_z ** 2)
    z0_new = k_cut if surf_type == "C" else (zc - R)

    if axis_dir == "Y":
        y_min_p, y_max_p = axis_min, axis_max
        x_min_p, x_max_p = -d_max, d_max
    else:
        x_min_p, x_max_p = axis_min, axis_max
        y_min_p, y_max_p = -d_max, d_max

    # ``proj_shape``/``proj_R`` are retained for callers using the old API.
    # New callers use the same 1/2/3 coverage convention as the other surfaces.
    if cover_type is None:
        if proj_shape == "C":
            cover_type = 3
            circ_R, circ_xc, circ_yc = float(proj_R), 0.0, 0.0
        else:
            cover_type = 1
    if cover_type not in (1, 2, 3):
        raise ValueError("cover_type 必须为 1/2/3")
    if cover_type == 2:
        if rect_xmin >= rect_xmax or rect_ymin >= rect_ymax:
            raise ValueError("矩形覆盖范围参数无效")
        if (rect_xmin > x_max_p or rect_xmax < x_min_p or
                rect_ymin > y_max_p or rect_ymax < y_min_p):
            raise ValueError("矩形覆盖区域与柱面投影区域无交集")
    elif cover_type == 3:
        if circ_R <= 0:
            raise ValueError("圆形覆盖半径必须为正数")
        nearest_x = min(max(circ_xc, x_min_p), x_max_p)
        nearest_y = min(max(circ_yc, y_min_p), y_max_p)
        if np.hypot(circ_xc - nearest_x, circ_yc - nearest_y) > circ_R + 1e-6:
            raise ValueError("圆形覆盖区域与柱面投影区域无交集")

    def z_at(x, y):
        d = x if axis_dir == "Y" else y
        sq = np.sqrt(max(0.0, work_R ** 2 - d ** 2))
        return (zc + sq) if surf_type == "C" else (zc - sq)

    if cover_type == 1:
        in_local = lambda x, y: True
    elif cover_type == 2:
        in_local = lambda x, y: (
            rect_xmin - 1e-6 <= x <= rect_xmax + 1e-6 and
            rect_ymin - 1e-6 <= y <= rect_ymax + 1e-6)
    else:
        in_local = lambda x, y: (
            (x - circ_xc) ** 2 + (y - circ_yc) ** 2 <= circ_R ** 2 + 1e-6)
    if traj_type == "G":
        p2d = generate_surface_raster(
            x_min_p, x_max_p, y_min_p, y_max_p, direction,
            step_len, line_spacing, z_at, keep=in_local)
    else:
        R_sp = np.hypot(max(abs(x_min_p), abs(x_max_p)), max(abs(y_min_p), abs(y_max_p)))
        raw = generate_spiral_2d(pitch, arc_step, R_sp, 0.0, 0.0)
        p2d = [[x, y] for x, y in raw
               if x_min_p <= x <= x_max_p and y_min_p <= y <= y_max_p and in_local(x, y)]
    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    result = []
    for x, y in p2d:
        d = x if axis_dir == "Y" else y
        if abs(d) > d_max + 1e-6:
            continue
        z_abs = z_at(x, y)
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
