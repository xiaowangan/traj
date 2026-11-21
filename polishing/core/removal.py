"""Migrated numerical service: RemovalMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np
from polishing.core.sampling import _odd_axis

class RemovalMixin:
    @classmethod
    def build_spot(cls, raw, pixel, dx, dy, duration, x_offset=0.0, y_offset=0.0):
        if duration <= 0:
            raise ValueError("采斑时长必须大于 0")
        source = -np.asarray(raw, dtype=float)
        source -= np.nanmin(source)
        source /= duration
        source[~np.isfinite(source)] = 0.0
        rows, cols = source.shape
        old_dx = float(pixel if pixel and pixel > 0 else dx)
        old_dy = old_dx
        old_x = (np.arange(cols) - (cols - 1) / 2.0) * old_dx + x_offset
        old_y = (np.arange(rows) - (rows - 1) / 2.0) * old_dy + y_offset
        if cols > 1 and rows > 1:
            # 椭圆掩膜按广播方式合成，避免 (1,cols) 原地 += (rows,1) 触发形状错误
            ellipse_x = ((old_x - x_offset) / max(abs(old_x[-1] - x_offset), 1e-12)) ** 2
            ellipse_y = ((old_y - y_offset) / max(abs(old_y[-1] - y_offset), 1e-12)) ** 2
            source[ellipse_y[:, None] + ellipse_x[None, :] >= 1.0] = 0.0
        half_x = max(abs(old_x[0]), abs(old_x[-1])); half_y = max(abs(old_y[0]), abs(old_y[-1]))
        new_x = _odd_axis(half_x, dx); new_y = _odd_axis(half_y, dy)
        horizontal = cls._interp_axis(source, old_x, new_x, axis=1)
        spot = cls._interp_axis(horizontal, old_y, new_y, axis=0)
        spot[~np.isfinite(spot)] = 0.0
        nonzero = np.argwhere(np.abs(spot) > np.max(np.abs(spot)) * 1e-12)
        if nonzero.size == 0:
            raise ValueError("抛光斑去除函数全为零")
        cy = spot.shape[0] // 2; cx = spot.shape[1] // 2
        hy = int(max(cy - nonzero[:, 0].min(), nonzero[:, 0].max() - cy))
        hx = int(max(cx - nonzero[:, 1].min(), nonzero[:, 1].max() - cx))
        spot = spot[cy - hy:cy + hy + 1, cx - hx:cx + hx + 1]
        impulse = float(np.sum(spot))
        return {"kernel": spot, "impulse": impulse, "dx": dx, "dy": dy,
                "volume_efficiency": impulse * dx * dy * 632.8e-6 * 60.0}
