# -*- coding: utf-8 -*-
"""各曲面/平面轨迹模块共用的二维投影点生成工具"""
import numpy as np


def _linspace_inclusive(start, stop, step):
    arr = np.arange(start, stop + 1e-9, step)
    if len(arr) == 0:
        return np.array([start])
    if arr[-1] < stop - step * 0.01:
        arr = np.append(arr, stop)
    return arr


def generate_raster_rect(x_min, x_max, y_min, y_max, direction, step_len, line_spacing):
    """矩形区域内的栅形二维投影点（逐行生成，避免大参数时内存问题）"""
    points = []
    if direction == "X":
        for i, y in enumerate(_linspace_inclusive(y_min, y_max, line_spacing)):
            xs = _linspace_inclusive(x_min, x_max, step_len)
            if i % 2:
                xs = xs[::-1]
            y_f = float(y)
            for x in xs:
                points.append([float(x), y_f])
    else:
        for i, x in enumerate(_linspace_inclusive(x_min, x_max, line_spacing)):
            ys = _linspace_inclusive(y_min, y_max, step_len)
            if i % 2:
                ys = ys[::-1]
            x_f = float(x)
            for y in ys:
                points.append([x_f, float(y)])
    return points


def generate_raster_circle(xc, yc, R, direction, step_len, line_spacing):
    """圆形区域内的栅形二维投影点（逐行裁剪，不生成完整矩形再过滤）"""
    R2 = R * R + 1e-6
    points = []
    if direction == "X":
        for i, y in enumerate(_linspace_inclusive(yc - R, yc + R, line_spacing)):
            dy2 = (float(y) - yc) ** 2
            if dy2 > R2:
                continue
            half = np.sqrt(R2 - dy2)
            xs = _linspace_inclusive(xc - half, xc + half, step_len)
            if i % 2:
                xs = xs[::-1]
            y_f = float(y)
            for x in xs:
                if (float(x) - xc) ** 2 + dy2 <= R2:
                    points.append([float(x), y_f])
    else:
        for i, x in enumerate(_linspace_inclusive(xc - R, xc + R, line_spacing)):
            dx2 = (float(x) - xc) ** 2
            if dx2 > R2:
                continue
            half = np.sqrt(R2 - dx2)
            ys = _linspace_inclusive(yc - half, yc + half, step_len)
            if i % 2:
                ys = ys[::-1]
            x_f = float(x)
            for y in ys:
                if dx2 + (float(y) - yc) ** 2 <= R2:
                    points.append([x_f, float(y)])
    return points


def generate_spiral_2d(pitch, arc_step, R_max, xc, yc):
    """等弧长阿基米德螺旋线二维点
    细分步长固定为 0.002 弧度（工程精度，不自适应）。
    """
    DTHETA = 0.05          # 固定细分步长（弧度）
    b = pitch / (2 * np.pi)
    if b <= 0:
        raise ValueError("螺距 pitch 必须为正数")
    theta_max = R_max / b
    theta_fine = np.arange(0, theta_max + DTHETA, DTHETA)
    r_fine = b * theta_fine
    x_fine = xc + r_fine * np.cos(theta_fine)
    y_fine = yc + r_fine * np.sin(theta_fine)
    cum_len = np.concatenate([[0], np.cumsum(
        np.sqrt(np.diff(x_fine) ** 2 + np.diff(y_fine) ** 2))])
    desired = np.arange(0, cum_len[-1] + 1e-9, arc_step)
    if len(desired) == 0:
        desired = np.array([0.0])
    if desired[-1] < cum_len[-1] - arc_step * 0.01:
        desired = np.append(desired, cum_len[-1])
    theta_d = np.interp(desired, cum_len, theta_fine)
    r_d = b * theta_d
    return [[float(xc + r_d[i] * np.cos(theta_d[i])),
             float(yc + r_d[i] * np.sin(theta_d[i]))] for i in range(len(theta_d))]
