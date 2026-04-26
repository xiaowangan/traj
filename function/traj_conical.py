# -*- coding: utf-8 -*-
"""锥面轨迹（凸锥/凹锥，全覆盖/矩形/圆形，含法向量）"""
import numpy as np
from ._traj_common import generate_raster_rect, generate_spiral_2d


def generate_conical(cone_type=1, alpha_deg=30.0, H=50.0,
                     cover_type=1,
                     rect_xmin=0.0, rect_xmax=0.0, rect_ymin=0.0, rect_ymax=0.0,
                     circ_R=0.0, circ_xc=0.0, circ_yc=0.0,
                     traj_type="G", direction="X",
                     step_len=1.0, line_spacing=5.0,
                     pitch=5.0, arc_step=1.0):
    if alpha_deg <= 0 or alpha_deg >= 90:
        raise ValueError("半顶角必须在 (0, 90) 度之间")
    if H <= 0:
        raise ValueError("高度H必须为正数")
    alpha = alpha_deg * np.pi / 180.0
    tan_a = np.tan(alpha)
    R_base = H * tan_a

    def z_cone(r):
        return (H - r / tan_a) if cone_type == 1 else (r / tan_a)

    def in_bound(x, y):
        r = np.hypot(x, y)
        if r > R_base + 1e-6: return False
        if cover_type == 2:
            return rect_xmin - 1e-6 <= x <= rect_xmax + 1e-6 and rect_ymin - 1e-6 <= y <= rect_ymax + 1e-6
        if cover_type == 3:
            return (x - circ_xc) ** 2 + (y - circ_yc) ** 2 <= circ_R ** 2 + 1e-6
        return True

    if cover_type == 1:
        xmn, xmx, ymn, ymx = -R_base, R_base, -R_base, R_base
        xc_sp, yc_sp, R_sp = 0.0, 0.0, R_base
    elif cover_type == 2:
        if rect_xmin >= rect_xmax or rect_ymin >= rect_ymax:
            raise ValueError("矩形覆盖范围参数无效")
        xmn, xmx, ymn, ymx = rect_xmin, rect_xmax, rect_ymin, rect_ymax
        xc_sp = (rect_xmin + rect_xmax) / 2; yc_sp = (rect_ymin + rect_ymax) / 2
        R_sp = 0.5 * np.hypot(rect_xmax - rect_xmin, rect_ymax - rect_ymin)
    else:
        if circ_R <= 0: raise ValueError("圆形覆盖半径必须为正数")
        xmn = circ_xc - circ_R; xmx = circ_xc + circ_R
        ymn = circ_yc - circ_R; ymx = circ_yc + circ_R
        xc_sp, yc_sp, R_sp = circ_xc, circ_yc, circ_R

    if traj_type == "G":
        p2d = generate_raster_rect(xmn, xmx, ymn, ymx, direction, step_len, line_spacing)
    else:
        p2d = generate_spiral_2d(pitch, arc_step, R_sp, xc_sp, yc_sp)

    p2d = [[x, y] for x, y in p2d if in_bound(x, y)]
    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    result = []
    for x, y in p2d:
        r = np.hypot(x, y)
        z = float(z_cone(r))
        if r < 1e-12:
            nx, ny, nz = 0.0, 0.0, 1.0
        elif cone_type == 1:
            raw = np.array([x / (r * tan_a), y / (r * tan_a), 1.0])
            raw /= np.linalg.norm(raw); nx, ny, nz = raw
        else:
            raw = np.array([-x / (r * tan_a), -y / (r * tan_a), 1.0])
            raw /= np.linalg.norm(raw); nx, ny, nz = raw
        result.append([round(x, 6), round(y, 6), round(z, 6),
                       round(float(nx), 6), round(float(ny), 6), round(float(nz), 6)])
    return result
