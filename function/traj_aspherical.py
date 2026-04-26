# -*- coding: utf-8 -*-
"""
非球面轨迹（栅形/螺旋线，支持离轴，含法向量）
输出：每点 [X, Y, Z, Nx, Ny, Nz]
"""
import numpy as np
from ._traj_common import (generate_raster_rect, generate_raster_circle,
                            generate_spiral_2d)


def generate_aspherical(R, k=0.0,
                        A4=0.0, A6=0.0, A8=0.0, A10=0.0, A12=0.0, A14=0.0,
                        offcenter=0.0,
                        traj_type="G", direction="X",
                        step_len=1.0, line_spacing=5.0,
                        pitch=5.0, arc_step=1.0,
                        bound_type=1,
                        full_width=0.0, full_length=0.0,
                        rect_xmin=0.0, rect_xmax=0.0,
                        rect_ymin=0.0, rect_ymax=0.0,
                        circ_R=0.0, circ_xc=0.0, circ_yc=0.0):
    if R == 0:
        raise ValueError("曲率半径 R 不能为 0")
    C = -1.0 / R

    if bound_type == 1:
        if full_width <= 0 or full_length <= 0:
            raise ValueError("X方向宽度和Y方向长度必须为正数")
        xmin, xmax = -full_width / 2, full_width / 2
        ymin, ymax = -full_length / 2, full_length / 2
        if traj_type == "G":
            p2d = generate_raster_rect(xmin, xmax, ymin, ymax, direction, step_len, line_spacing)
        else:
            R_max = 0.5 * np.hypot(full_width, full_length)
            p2d = generate_spiral_2d(pitch, arc_step, R_max, 0.0, 0.0)
            p2d = [[x, y] for x, y in p2d if xmin - 1e-6 <= x <= xmax + 1e-6 and ymin - 1e-6 <= y <= ymax + 1e-6]
    elif bound_type == 2:
        if rect_xmin >= rect_xmax or rect_ymin >= rect_ymax:
            raise ValueError("矩形边界参数无效")
        if traj_type == "G":
            p2d = generate_raster_rect(rect_xmin, rect_xmax, rect_ymin, rect_ymax, direction, step_len, line_spacing)
        else:
            cx = (rect_xmin + rect_xmax) / 2; cy = (rect_ymin + rect_ymax) / 2
            R_max = 0.5 * np.hypot(rect_xmax - rect_xmin, rect_ymax - rect_ymin)
            p2d = generate_spiral_2d(pitch, arc_step, R_max, cx, cy)
            p2d = [[x, y] for x, y in p2d if rect_xmin - 1e-6 <= x <= rect_xmax + 1e-6 and rect_ymin - 1e-6 <= y <= rect_ymax + 1e-6]
    else:
        if circ_R <= 0:
            raise ValueError("圆形边界半径必须为正数")
        if traj_type == "G":
            p2d = generate_raster_circle(circ_xc, circ_yc, circ_R, direction, step_len, line_spacing)
        else:
            p2d = generate_spiral_2d(pitch, arc_step, circ_R, circ_xc, circ_yc)
            p2d = [[x, y] for x, y in p2d if (x - circ_xc) ** 2 + (y - circ_yc) ** 2 <= circ_R ** 2 + 1e-6]

    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    result = []
    for x, y in p2d:
        ys = y - offcenter
        r_sq = x * x + ys * ys
        under = 1.0 - (1.0 + k) * C * C * r_sq
        sq = np.sqrt(max(0.0, under))
        denom = 1.0 + sq
        Z = (C * r_sq) / denom + (A4 * r_sq ** 2 + A6 * r_sq ** 3 + A8 * r_sq ** 4 +
                                    A10 * r_sq ** 5 + A12 * r_sq ** 6 + A14 * r_sq ** 7)
        r = np.sqrt(r_sq) if r_sq > 1e-24 else 1e-12
        dz_dr = 0.0
        if under >= 0:
            dz_dr = (2 * C * r * denom - C * r_sq * (-(1 + k) * C * C * r / max(sq, 1e-30))) / denom ** 2
        dz_dr += (4 * A4 * r ** 3 + 6 * A6 * r ** 5 + 8 * A8 * r ** 7 +
                  10 * A10 * r ** 9 + 12 * A12 * r ** 11 + 14 * A14 * r ** 13)
        dZdX = dz_dr * (x / r); dZdY = dz_dr * (ys / r)
        nf = np.sqrt(dZdX ** 2 + dZdY ** 2 + 1.0)
        result.append([round(x, 6), round(y, 6), round(Z, 6),
                       round(-dZdX / nf, 6), round(-dZdY / nf, 6), round(1.0 / nf, 6)])
    return result
