"""Migrated numerical service: ContactMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np

class ContactMixin:
    @staticmethod
    def disk_fit(radius, conic, coefficients, work_diameter, disk_diameter):
        """MATLAB ToolConButton meridian/equivalent-radius estimate (mm).

        Evaluate only the visible physical radius D/2, use radius magnitude,
        and retain precision until plotting. Rate is spatial |Δgap/Δx|.
        """
        values = [radius, conic, work_diameter, disk_diameter, *coefficients]
        if not np.isfinite(values).all():
            raise ValueError("贴合校验参数必须是有限数值")
        if work_diameter <= 0 or disk_diameter <= 0:
            raise ValueError("工件与抛光盘口径必须大于 0")
        if disk_diameter > work_diameter:
            raise ValueError("抛光盘口径不能大于工件最大口径")
        x = np.arange(0.0, work_diameter / 2.0, 1.0)
        x = np.append(x, work_diameter / 2.0)
        if len(x) > 100000:
            raise ValueError("工件口径过大，超出校验采样范围")
        c = 0.0 if radius == 0 else 1.0 / radius
        under = 1.0 - (1.0 + conic) * c * c * x * x
        if np.any(under <= 0):
            raise ValueError("工件口径超出非球面有效定义域，或边缘斜率无穷大")
        slope = c * x / np.sqrt(under)
        for order, coefficient in enumerate(coefficients, 2):
            slope -= 2 * order * coefficient * x ** (2 * order - 1)
        inverse_radius = np.zeros_like(x)
        inverse_radius[0] = abs(c)
        inverse_radius[1:] = np.abs(slope[1:] / np.hypot(1.0, slope[1:]) / x[1:])
        ratio = disk_diameter / 2.0 * inverse_radius
        if not np.isfinite(ratio).all() or np.any(ratio > 1):
            raise ValueError("抛光盘半径超过局部等效曲率半径，请减小抛光盘口径")
        # Rationalized sag avoids cancellation at large equivalent radii.
        gap = (disk_diameter / 2.0) ** 2 * inverse_radius / (
            1.0 + np.sqrt(1.0 - ratio * ratio))
        rate = np.abs(np.diff(gap) / np.diff(x))
        return {"x": x, "gap": gap, "rate_x": (x[:-1] + x[1:]) / 2,
                "rate": rate, "max_gap": float(np.max(gap)),
                "max_rate": float(np.max(rate)),
                "passed": bool(np.max(gap) <= 0.05 and np.max(rate) <= 5e-5)}
