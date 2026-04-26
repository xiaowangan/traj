# -*- coding: utf-8 -*-
"""
平面栅形轨迹
输出：每点 [X, Y, Z=0, Nx=0, Ny=0, Nz=1]
"""
import numpy as np


def _check_positive(value, name):
    if value <= 0:
        raise ValueError(f"参数「{name}」必须为正数，当前值：{value}")


def _linspace_inclusive(start, stop, step):
    arr = np.arange(start, stop + 1e-9, step)
    if len(arr) == 0:
        return np.array([start])
    if arr[-1] < stop - step * 0.01:
        arr = np.append(arr, stop)
    return arr


def _inside_shape(x, y, shape, xmin_s, xmax_s, ymin_s, ymax_s, cx, cy, r):
    if shape == "R":
        return xmin_s - 1e-9 <= x <= xmax_s + 1e-9 and ymin_s - 1e-9 <= y <= ymax_s + 1e-9
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 + 1e-9


def _pt(x, y):
    return [round(float(x), 6), round(float(y), 6), 0.0, 0.0, 0.0, 1.0]


def generate_planar_raster(shape,
                            rect_A=0.0, rect_B=0.0, circle_R=0.0,
                            direction="X", step_len=1.0, line_spacing=5.0,
                            cover_type=1,
                            sub_x0=0.0, sub_y0=0.0, sub_C=0.0, sub_D=0.0):
    """平面栅形轨迹，Z=0，法向量(0,0,1)"""
    _check_positive(step_len, "步长")
    _check_positive(line_spacing, "线间距")
    if shape == "R":
        _check_positive(rect_A, "矩形长A"); _check_positive(rect_B, "矩形宽B")
        xmin_s, xmax_s = -rect_A / 2, rect_A / 2
        ymin_s, ymax_s = -rect_B / 2, rect_B / 2
        cx = cy = r = 0.0
    else:
        _check_positive(circle_R, "圆形半径R")
        xmin_s, xmax_s = -circle_R, circle_R
        ymin_s, ymax_s = -circle_R, circle_R
        cx, cy, r = 0.0, 0.0, circle_R

    if cover_type == 1:
        xmin, xmax, ymin, ymax = xmin_s, xmax_s, ymin_s, ymax_s
    else:
        _check_positive(sub_C, "子区域长C"); _check_positive(sub_D, "子区域宽D")
        xmin = max(xmin_s, sub_x0);  xmax = min(xmax_s, sub_x0 + sub_C)
        ymin = max(ymin_s, sub_y0);  ymax = min(ymax_s, sub_y0 + sub_D)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("子区域与形状无交集，请检查坐标参数")

    points = []
    if direction == "X":
        for i, y in enumerate(_linspace_inclusive(ymin, ymax, line_spacing)):
            row = _linspace_inclusive(xmin, xmax, step_len)
            if i % 2: row = row[::-1]
            for x in row:
                if _inside_shape(x, y, shape, xmin_s, xmax_s, ymin_s, ymax_s, cx, cy, r):
                    points.append(_pt(x, y))
    else:
        for i, x in enumerate(_linspace_inclusive(xmin, xmax, line_spacing)):
            col = _linspace_inclusive(ymin, ymax, step_len)
            if i % 2: col = col[::-1]
            for y in col:
                if _inside_shape(x, y, shape, xmin_s, xmax_s, ymin_s, ymax_s, cx, cy, r):
                    points.append(_pt(x, y))
    return points
