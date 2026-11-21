"""期望曲面口径与实测面形的对齐检查，以及各类区域的面积账目（纯数值，不依赖 Qt）。

软件把“实际 = 期望 + 面形误差”当作前提，但两张网格是各自独立生成的。本模块在同一物理
坐标（mm，以各自网格中心为原点）下比较它们，把不一致显式报告出来，而不是默默继续计算。
"""
from __future__ import annotations
import numpy as np

ORIENTATION_NOTE = "镜像、旋转 90° 无法由口径形状判断，请用已知特征点核对测量坐标方向。"


def _centered_coordinates(indices, count, pitch):
    return (np.asarray(indices, dtype=float) - (count - 1) / 2.0) * pitch


def check_alignment(model, surface, dx_mm, dy_mm):
    """比较期望曲面口径与实测有效区；返回可保存的字典（不含大数组）。"""
    mask = np.asarray(model["mask"], dtype=bool)
    surface = np.asarray(surface, dtype=float)
    valid = np.isfinite(surface)
    if not mask.any() or not valid.any():
        raise ValueError("期望曲面口径或实测有效区为空，无法检查对齐")
    mdx, mdy = float(model["dx"]), float(model["dy"])
    ny_m, nx_m = mask.shape
    ny_s, nx_s = valid.shape

    my, mx = np.nonzero(mask)
    model_x = _centered_coordinates(mx, nx_m, mdx)
    model_y = _centered_coordinates(my, ny_m, mdy)
    sy, sx = np.nonzero(valid)
    meas_x = _centered_coordinates(sx, nx_s, dx_mm)
    meas_y = _centered_coordinates(sy, ny_s, dy_mm)

    col = np.rint(meas_x / mdx + (nx_m - 1) / 2.0).astype(int)
    row = np.rint(meas_y / mdy + (ny_m - 1) / 2.0).astype(int)
    inside = (col >= 0) & (col < nx_m) & (row >= 0) & (row < ny_m)
    in_aperture = np.zeros(col.shape, dtype=bool)
    in_aperture[inside] = mask[row[inside], col[inside]]
    covered = np.zeros_like(mask)
    covered[row[in_aperture], col[in_aperture]] = True

    overlap_fraction = float(np.count_nonzero(in_aperture) / in_aperture.size)
    coverage = float(np.count_nonzero(covered) / np.count_nonzero(mask))
    model_extent = (float(np.ptp(model_x)) + mdx, float(np.ptp(model_y)) + mdy)
    meas_extent = (float(np.ptp(meas_x)) + dx_mm, float(np.ptp(meas_y)) + dy_mm)
    offset = (float(np.mean(meas_x) - np.mean(model_x)), float(np.mean(meas_y) - np.mean(model_y)))
    offset_norm = float(np.hypot(*offset))
    pixel_area = float(dx_mm) * float(dy_mm)
    model_area = float(np.count_nonzero(mask)) * mdx * mdy
    meas_area = float(np.count_nonzero(valid)) * pixel_area

    warnings = []
    tolerance = max(2.0 * max(dx_mm, dy_mm), 0.01 * min(model_extent))
    if offset_norm > tolerance:
        warnings.append(
            f"实测有效区中心相对期望口径偏移 {offset_norm:.3g} mm（X {offset[0]:+.3g}，Y {offset[1]:+.3g}）；"
            "请检查装夹坐标、离轴量或测量原点。")
    if overlap_fraction < 0.98:
        warnings.append(f"{1 - overlap_fraction:.1%} 的实测点落在期望口径之外。")
    if coverage < 0.90:
        warnings.append(f"实测数据只覆盖期望口径面积的 {coverage:.1%}。")
    for axis, model_size, meas_size in (("X", model_extent[0], meas_extent[0]),
                                        ("Y", model_extent[1], meas_extent[1])):
        ratio = meas_size / model_size
        if not 0.9 <= ratio <= 1.1:
            warnings.append(f"{axis} 向实测范围 {meas_size:.4g} mm 与期望口径 {model_size:.4g} mm 相差较大（比值 {ratio:.2f}）。")
    return {
        "ok": not warnings,
        "warnings": warnings,
        "note": ORIENTATION_NOTE,
        "center_offset_mm": [offset[0], offset[1]],
        "overlap_fraction": overlap_fraction,
        "aperture_coverage": coverage,
        "model_extent_mm": list(model_extent),
        "measured_extent_mm": list(meas_extent),
        "areas_mm2": {"physical_aperture": model_area, "measured_valid": meas_area,
                      "overlap": float(np.count_nonzero(covered)) * mdx * mdy},
    }


def region_areas(solution, dx_mm, dy_mm):
    """驻留求解用到的几类区域各自的面积，避免把“求解区”“轨迹区”“检测区”混为一谈。"""
    pixel_area = float(dx_mm) * float(dy_mm)
    return {name: float(np.count_nonzero(solution[key])) * pixel_area
            for name, key in (("solver_surface", "surface_mask"),
                              ("trajectory", "trajectory_mask"),
                              ("evaluation", "evaluation_mask"))}
