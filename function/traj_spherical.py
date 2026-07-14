# -*- coding: utf-8 -*-
"""球面轨迹（凸/凹，栅形/螺旋线，含法向量）

栅形：使用球面弧长参数化（行间距/步长 = 真实球面弧长），避免投影栅格在球冠
边缘出现的 Z 突变；同时支持全部覆盖 / 局部矩形 / 局部圆形三种覆盖区域。
螺旋线：沿用阿基米德螺旋（投影平面等弧长）+ 投影裁剪，与柱/锥/非球面一致。
"""
import numpy as np
from ._traj_common import generate_spiral_2d


SPHERICAL_WALL_THICKNESS_MM = 0.5


def _angle_values(start, stop, step):
    if stop < start:
        return np.array([])
    vals = np.arange(start, stop + 1e-12, step)
    if len(vals) == 0:
        return np.array([start, stop], dtype=float)
    if vals[-1] < stop - step * 0.01:
        vals = np.append(vals, stop)
    return vals


def _interval_intersect(a_lo, a_hi, b_lo, b_hi):
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    if hi < lo - 1e-9:
        return None
    return lo, hi


def _row_x_intervals(y, r_proj, cover_type,
                     rect_xmin, rect_xmax, rect_ymin, rect_ymax,
                     circ_R, circ_xc, circ_yc):
    """给定行 y，返回该行内 x 的允许区间列表（按 cover 与球冠投影圆求交）。"""
    cap_term = r_proj * r_proj - y * y
    if cap_term <= 1e-12:
        return []
    cap_lim = float(np.sqrt(cap_term))
    cap_lo, cap_hi = -cap_lim, cap_lim

    if cover_type == 1:
        return [(cap_lo, cap_hi)]
    if cover_type == 2:
        if y < rect_ymin - 1e-9 or y > rect_ymax + 1e-9:
            return []
        seg = _interval_intersect(cap_lo, cap_hi, rect_xmin, rect_xmax)
        return [seg] if seg else []
    # cover_type == 3：圆形覆盖区
    dy = y - circ_yc
    circ_term = circ_R * circ_R - dy * dy
    if circ_term <= 1e-12:
        return []
    circ_half = float(np.sqrt(circ_term))
    seg = _interval_intersect(cap_lo, cap_hi, circ_xc - circ_half, circ_xc + circ_half)
    return [seg] if seg else []


def _row_y_intervals(x, r_proj, cover_type,
                     rect_xmin, rect_xmax, rect_ymin, rect_ymax,
                     circ_R, circ_xc, circ_yc):
    """对称：给定列 x，返回该列内 y 的允许区间列表。"""
    cap_term = r_proj * r_proj - x * x
    if cap_term <= 1e-12:
        return []
    cap_lim = float(np.sqrt(cap_term))
    cap_lo, cap_hi = -cap_lim, cap_lim

    if cover_type == 1:
        return [(cap_lo, cap_hi)]
    if cover_type == 2:
        if x < rect_xmin - 1e-9 or x > rect_xmax + 1e-9:
            return []
        seg = _interval_intersect(cap_lo, cap_hi, rect_ymin, rect_ymax)
        return [seg] if seg else []
    dx = x - circ_xc
    circ_term = circ_R * circ_R - dx * dx
    if circ_term <= 1e-12:
        return []
    circ_half = float(np.sqrt(circ_term))
    seg = _interval_intersect(cap_lo, cap_hi, circ_yc - circ_half, circ_yc + circ_half)
    return [seg] if seg else []


def _generate_spherical_raster(R, r_proj, direction, step_len, line_spacing,
                               cover_type,
                               rect_xmin, rect_xmax, rect_ymin, rect_ymax,
                               circ_R, circ_xc, circ_yc):
    """按球面弧长生成栅形投影点（行/列在球冠上等弧长分布）。

    行方向：fixed 轴上以 R*Δu 为间距（球面真实弧长 = line_spacing）。
    行内方向：moving 轴上以 R*cos(u)*Δβ 为间距（球面真实弧长 = step_len）。
    """
    # 求"行"轴（direction=X 时行平行 X 轴 → 行轴 = Y）允许的整体范围
    original_cover_type = cover_type
    cover_type = 1
    if direction == "X":
        # 行 y 范围由球冠 (|y| ≤ r_proj) 与 cover 在 y 维上的范围相交决定
        if cover_type == 1:
            fixed_lo, fixed_hi = -r_proj, r_proj
        elif cover_type == 2:
            inter = _interval_intersect(-r_proj, r_proj, rect_ymin, rect_ymax)
            if not inter:
                return []
            fixed_lo, fixed_hi = inter
        else:
            inter = _interval_intersect(-r_proj, r_proj,
                                        circ_yc - circ_R, circ_yc + circ_R)
            if not inter:
                return []
            fixed_lo, fixed_hi = inter
    else:  # direction == "Y"
        if cover_type == 1:
            fixed_lo, fixed_hi = -r_proj, r_proj
        elif cover_type == 2:
            inter = _interval_intersect(-r_proj, r_proj, rect_xmin, rect_xmax)
            if not inter:
                return []
            fixed_lo, fixed_hi = inter
        else:
            inter = _interval_intersect(-r_proj, r_proj,
                                        circ_xc - circ_R, circ_xc + circ_R)
            if not inter:
                return []
            fixed_lo, fixed_hi = inter

    # 把 fixed 范围转成球面角 u（fixed = R sin u）
    u_lo = np.arcsin(np.clip(fixed_lo / R, -1.0, 1.0))
    u_hi = np.arcsin(np.clip(fixed_hi / R, -1.0, 1.0))
    du = line_spacing / R
    rows = _angle_values(u_lo, u_hi, du)
    if len(rows) == 0:
        return []

    points = []
    for idx, u in enumerate(rows):
        fixed = R * np.sin(u)
        rho = max(0.0, R * np.cos(u))  # 该行所在小圆半径
        if direction == "X":
            intervals = _row_x_intervals(fixed, r_proj, cover_type,
                                         rect_xmin, rect_xmax, rect_ymin, rect_ymax,
                                         circ_R, circ_xc, circ_yc)
        else:
            intervals = _row_y_intervals(fixed, r_proj, cover_type,
                                         rect_xmin, rect_xmax, rect_ymin, rect_ymax,
                                         circ_R, circ_xc, circ_yc)
        if not intervals:
            continue

        # 把投影坐标区间转换成球面角 β（moving = rho sin β）
        moving_vals_row = []
        for seg_lo, seg_hi in intervals:
            if rho < 1e-9:
                moving_vals_row.append(np.array([0.0]))
                continue
            b_lo = np.arcsin(np.clip(seg_lo / rho, -1.0, 1.0))
            b_hi = np.arcsin(np.clip(seg_hi / rho, -1.0, 1.0))
            db = step_len / rho
            beta_vals = _angle_values(b_lo, b_hi, db)
            if len(beta_vals) == 0:
                continue
            moving_vals_row.append(rho * np.sin(beta_vals))
        if not moving_vals_row:
            continue
        moving_vals = np.concatenate(moving_vals_row)
        if idx % 2:
            moving_vals = moving_vals[::-1]

        for moving in moving_vals:
            if direction == "X":
                points.append([float(moving), float(fixed)])
            else:
                points.append([float(fixed), float(moving)])
    if original_cover_type == 1:
        return points
    if original_cover_type == 2:
        return [[x, y] for x, y in points
                if rect_xmin - 1e-6 <= x <= rect_xmax + 1e-6 and
                rect_ymin - 1e-6 <= y <= rect_ymax + 1e-6]
    return [[x, y] for x, y in points
            if (x - circ_xc) ** 2 + (y - circ_yc) ** 2 <= circ_R ** 2 + 1e-6]


def generate_spherical(R, zc=0.0, surf_type="convex", h=None,
                       traj_type="G", direction="X",
                       step_len=1.0, line_spacing=5.0,
                       pitch=5.0, arc_step=1.0,
                       cover_type=1,
                       rect_xmin=0.0, rect_xmax=0.0,
                       rect_ymin=0.0, rect_ymax=0.0,
                       circ_R=0.0, circ_xc=0.0, circ_yc=0.0,
                       wall_thickness=SPHERICAL_WALL_THICKNESS_MM):
    if R <= 0:
        raise ValueError("球体半径R必须为正数")
    if h is None or h <= 0 or h > 2 * R:
        raise ValueError(f"球冠高度 h 必须在 (0, 2R={2*R:.4f}] 范围内")

    work_R = float(R)
    if surf_type == "convex":
        z_cut = zc + R - h
        r_proj = float(np.sqrt(max(0.0, R ** 2 - (z_cut - zc) ** 2)))
        z_min_region, z_max_region = z_cut, zc + R
    else:
        if wall_thickness <= 0 or wall_thickness >= R:
            raise ValueError("凹球面固定厚度必须满足 0 < t < R")
        work_R = float(R - wall_thickness)
        z_cut = zc - R
        z_top = z_cut + h
        if abs(z_top - zc) >= work_R - 1e-12:
            raise ValueError(
                f"凹球面厚度 t={wall_thickness:.4f} mm 时，球冠高度 h 必须在 "
                f"({wall_thickness:.4f}, {2 * R - wall_thickness:.4f}) 内")
        r_proj = float(np.sqrt(max(0.0, work_R ** 2 - (z_top - zc) ** 2)))
        z_min_region, z_max_region = zc - work_R, z_top

    if r_proj < 1e-9:
        raise ValueError("投影圆半径为零，请调整 h 值")

    # ---- 覆盖区参数校验 ----
    if cover_type not in (1, 2, 3):
        raise ValueError("cover_type 必须为 1/2/3")
    if cover_type == 2:
        if rect_xmin >= rect_xmax or rect_ymin >= rect_ymax:
            raise ValueError("矩形覆盖范围参数无效")
        # 与投影圆是否有交集
        if (rect_xmin > r_proj or rect_xmax < -r_proj or
                rect_ymin > r_proj or rect_ymax < -r_proj):
            raise ValueError("矩形覆盖区域与球冠投影圆无交集")
    elif cover_type == 3:
        if circ_R <= 0:
            raise ValueError("圆形覆盖半径必须为正数")
        # 圆心到原点距离 + 圆半径必须能与投影圆产生交集
        if np.hypot(circ_xc, circ_yc) - circ_R > r_proj + 1e-6:
            raise ValueError("圆形覆盖区域与球冠投影圆无交集")

    # ---- 生成 2D 投影点 ----
    if traj_type == "G":
        p2d = _generate_spherical_raster(
            work_R, r_proj, direction, step_len, line_spacing,
            cover_type,
            rect_xmin, rect_xmax, rect_ymin, rect_ymax,
            circ_R, circ_xc, circ_yc)
    else:
        # 螺旋线：根据覆盖区域选圆心 / R_max
        if cover_type == 1:
            xc_sp, yc_sp, R_sp = 0.0, 0.0, r_proj
        elif cover_type == 2:
            xc_sp = 0.5 * (rect_xmin + rect_xmax)
            yc_sp = 0.5 * (rect_ymin + rect_ymax)
            R_sp = 0.5 * float(np.hypot(rect_xmax - rect_xmin, rect_ymax - rect_ymin))
        else:
            xc_sp, yc_sp = float(circ_xc), float(circ_yc)
            R_sp = float(circ_R)
        if R_sp < 1e-9:
            raise ValueError("螺旋线最大半径为零，请检查覆盖范围")
        # Local spiral coverage is filtered from the full-cap spiral as well.
        R_sp, xc_sp, yc_sp = r_proj, 0.0, 0.0
        raw = generate_spiral_2d(pitch, arc_step, R_sp, xc_sp, yc_sp)
        p2d = []
        r_proj2 = r_proj * r_proj + 1e-6
        for x, y in raw:
            if x * x + y * y > r_proj2:
                continue
            if cover_type == 2:
                if not (rect_xmin - 1e-6 <= x <= rect_xmax + 1e-6 and
                        rect_ymin - 1e-6 <= y <= rect_ymax + 1e-6):
                    continue
            elif cover_type == 3:
                if (x - circ_xc) ** 2 + (y - circ_yc) ** 2 > circ_R * circ_R + 1e-6:
                    continue
            p2d.append([x, y])

    if not p2d:
        raise ValueError("未生成任何轨迹点，请检查参数设置")

    # ---- 投影点 -> 球面 ----
    result = []
    for x, y in p2d:
        r2 = x * x + y * y
        sq = np.sqrt(max(0.0, work_R ** 2 - r2))
        z_abs = (zc + sq) if surf_type == "convex" else (zc - sq)
        if not (z_min_region - 1e-6 <= z_abs <= z_max_region + 1e-6):
            continue
        z_rel = z_abs - z_cut
        if surf_type == "convex":
            nx, ny, nz = x / work_R, y / work_R, sq / work_R
        else:
            # 内球面的加工法向朝向球心（腔体），即球半径方向的反向。
            nx, ny, nz = -x / work_R, -y / work_R, sq / work_R
        result.append([round(x, 6), round(y, 6), round(z_rel, 6),
                       round(nx, 6), round(ny, 6), round(nz, 6)])
    return result
