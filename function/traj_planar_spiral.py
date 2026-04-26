# -*- coding: utf-8 -*-
"""
平面螺旋线轨迹（等弧长阿基米德螺旋线）
输出：每点 [X, Y, Z=0, Nx=0, Ny=0, Nz=1]
"""
import numpy as np


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"参数「{name}」必须为正数，当前值：{value}")


def _inside_shape(x, y, shape, xmin_s, xmax_s, ymin_s, ymax_s, cx, cy, r):
    if shape == "R":
        return xmin_s - 1e-9 <= x <= xmax_s + 1e-9 and ymin_s - 1e-9 <= y <= ymax_s + 1e-9
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 + 1e-9


def _pt(x, y):
    return [round(float(x), 6), round(float(y), 6), 0.0, 0.0, 0.0, 1.0]


def generate_planar_spiral(shape,
                            rect_A=0.0, rect_B=0.0, circle_R=0.0,
                            pitch=5.0, arc_step=1.0,
                            spiral_cover_type=1, spiral_R_max=0.0,
                            spiral_xmin=0.0, spiral_xmax=0.0,
                            spiral_ymin=0.0, spiral_ymax=0.0):
    """平面螺旋线轨迹，Z=0，法向量(0,0,1)"""
    _check_positive(pitch, "螺距"); _check_positive(arc_step, "弧长步长")
    if shape == "R":
        _check_positive(rect_A, "矩形长A"); _check_positive(rect_B, "矩形宽B")
        xmin_s, xmax_s = -rect_A / 2, rect_A / 2
        ymin_s, ymax_s = -rect_B / 2, rect_B / 2
        cx_s = cy_s = r_s = 0.0
    else:
        _check_positive(circle_R, "圆形半径R")
        xmin_s, xmax_s = -circle_R, circle_R
        ymin_s, ymax_s = -circle_R, circle_R
        cx_s, cy_s, r_s = 0.0, 0.0, circle_R

    if spiral_cover_type == 1:
        _check_positive(spiral_R_max, "最大半径R_max")
        R_max = spiral_R_max; cx_sp = cy_sp = 0.0
    else:
        if spiral_xmin >= spiral_xmax or spiral_ymin >= spiral_ymax:
            raise ValueError("矩形覆盖范围参数无效")
        cx_sp = (spiral_xmin + spiral_xmax) / 2
        cy_sp = (spiral_ymin + spiral_ymax) / 2
        R_max = 0.5 * np.sqrt((spiral_xmax - spiral_xmin) ** 2 + (spiral_ymax - spiral_ymin) ** 2)

    b = pitch / (2 * np.pi)
    theta_fine = np.arange(0, R_max / b + 0.002, 0.002)
    r_fine = b * theta_fine
    x_fine = cx_sp + r_fine * np.cos(theta_fine)
    y_fine = cy_sp + r_fine * np.sin(theta_fine)
    cum_len = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_fine) ** 2 + np.diff(y_fine) ** 2))])
    desired = np.arange(0, cum_len[-1] + 1e-9, arc_step)
    if len(desired) == 0 or desired[-1] < cum_len[-1] - arc_step * 0.01:
        desired = np.append(desired, cum_len[-1])
    theta_d = np.interp(desired, cum_len, theta_fine)
    r_d = b * theta_d
    x_d = cx_sp + r_d * np.cos(theta_d)
    y_d = cy_sp + r_d * np.sin(theta_d)

    points = []
    for x, y in zip(x_d, y_d):
        if not _inside_shape(x, y, shape, xmin_s, xmax_s, ymin_s, ymax_s, cx_s, cy_s, r_s):
            continue
        if spiral_cover_type == 1:
            if (x - cx_sp) ** 2 + (y - cy_sp) ** 2 > (R_max + 1e-9) ** 2:
                continue
        else:
            if not (spiral_xmin - 1e-9 <= x <= spiral_xmax + 1e-9 and
                    spiral_ymin - 1e-9 <= y <= spiral_ymax + 1e-9):
                continue
        points.append(_pt(x, y))
    return points
