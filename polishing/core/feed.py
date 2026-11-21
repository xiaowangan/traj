"""Migrated numerical service: DwellTrajectoryMixin; legacy numerical conventions retained."""
from __future__ import annotations
import numpy as np

class DwellTrajectoryMixin:
    @staticmethod
    def raster_from_dwell(solution, step_x, step_y, direction="X", start="左下"):
        dwell = solution["dwell"]; x = solution["x"]; y = solution["y"]
        sx = max(1, int(round(step_x / max(abs(x[1] - x[0]), 1e-12)))) if len(x) > 1 else 1
        sy = max(1, int(round(step_y / max(abs(y[1] - y[0]), 1e-12)))) if len(y) > 1 else 1
        rows = list(range(0, len(y), sy)); cols = list(range(0, len(x), sx))
        if "上" in start: rows.reverse()
        if "右" in start: cols.reverse()
        samples = []
        if direction == "X":
            for ri, iy in enumerate(rows):
                current = cols if ri % 2 == 0 else list(reversed(cols))
                for ix in current:
                    if np.isfinite(dwell[iy, ix]): samples.append((x[ix], y[iy], dwell[iy, ix]))
        else:
            for ci, ix in enumerate(cols):
                current = rows if ci % 2 == 0 else list(reversed(rows))
                for iy in current:
                    if np.isfinite(dwell[iy, ix]): samples.append((x[ix], y[iy], dwell[iy, ix]))
        if not samples:
            raise ValueError("驻留时间区域内没有可用轨迹点")
        return np.asarray(samples, dtype=float)
