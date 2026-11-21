"""Migrated numerical service: ProcessingMixin; legacy numerical conventions retained."""
from __future__ import annotations
from collections import deque
import numpy as np
from polishing.core.sampling import _odd_axis

class ProcessingMixin:
    @classmethod
    def circular_kernel(cls, radius, dx, dy):
        if radius <= 0:
            return np.ones((1, 1), dtype=float)
        xs = _odd_axis(radius, dx)
        ys = _odd_axis(radius, dy)
        x2d, y2d = np.meshgrid(xs, ys)
        return ((x2d * x2d + y2d * y2d) <= radius * radius + 1e-12).astype(float)

    @classmethod
    def erode_mask(cls, mask, radius, dx, dy):
        mask = np.asarray(mask, dtype=bool)
        if radius <= 0:
            return mask.copy()
        kernel = cls.circular_kernel(radius, dx, dy)
        counts = cls.fft_convolve_same(mask.astype(float), kernel)
        return mask & (counts >= np.sum(kernel) - 1e-6)

    @classmethod
    def trim_surface(cls, data, radius, dx, dy):
        result = np.asarray(data, dtype=float).copy()
        keep = cls.erode_mask(np.isfinite(result), radius, dx, dy)
        result[~keep] = np.nan
        return result

    @classmethod
    def mean_filter(cls, data, diameter, dx, dy):
        result = np.asarray(data, dtype=float)
        kernel = cls.circular_kernel(max(diameter, 0.0) / 2.0, dx, dy)
        valid = np.isfinite(result)
        sums = cls.fft_convolve_same(np.where(valid, result, 0.0), kernel)
        counts = cls.fft_convolve_same(valid.astype(float), kernel)
        filtered = np.full_like(result, np.nan)
        good = valid & (counts > 1e-9)
        filtered[good] = sums[good] / counts[good]
        return filtered

    @staticmethod
    def rms_filter(data, multiplier):
        result = np.asarray(data, dtype=float).copy()
        valid = np.isfinite(result)
        clean = result[valid]
        if clean.size == 0:
            return result
        mean = float(np.mean(clean))
        rms = float(np.sqrt(np.mean((clean - mean) ** 2)))
        limit = max(float(multiplier), 1.0) * rms
        result[valid] = np.clip(result[valid] - mean, -limit, limit) + mean
        return result

    @staticmethod
    def nearest_fill(data):
        """Fill NaNs by nearest Manhattan neighbour without SciPy."""
        values = np.asarray(data, dtype=float).copy()
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError("没有可用于延拓的有效面形点")
        owner_y = np.full(values.shape, -1, dtype=np.int32)
        owner_x = np.full(values.shape, -1, dtype=np.int32)
        queue = deque()
        for y, x in np.argwhere(valid):
            owner_y[y, x] = y; owner_x[y, x] = x; queue.append((int(y), int(x)))
        rows, cols = values.shape
        while queue:
            y, x = queue.popleft()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < rows and 0 <= xx < cols and owner_y[yy, xx] < 0:
                    owner_y[yy, xx] = owner_y[y, x]
                    owner_x[yy, xx] = owner_x[y, x]
                    queue.append((yy, xx))
        missing = ~valid
        values[missing] = values[owner_y[missing], owner_x[missing]]
        return values

    @classmethod
    def extend_surface(cls, data, distance, dx, dy):
        px = max(0, int(round(distance / dx)))
        py = max(0, int(round(distance / dy)))
        padded = np.pad(np.asarray(data, dtype=float), ((py, py), (px, px)),
                        constant_values=np.nan)
        return cls.nearest_fill(padded)

    @staticmethod
    def _interp_axis(data, old_axis, new_axis, axis):
        source = np.asarray(data, dtype=float)
        moved = np.moveaxis(source, axis, -1)
        output = np.full(moved.shape[:-1] + (len(new_axis),), np.nan)
        for index in np.ndindex(moved.shape[:-1]):
            row = moved[index]
            good = np.isfinite(row)
            if np.count_nonzero(good) >= 2:
                output[index] = np.interp(
                    new_axis, old_axis[good], row[good], left=np.nan, right=np.nan)
            elif np.count_nonzero(good) == 1:
                output[index][np.argmin(np.abs(new_axis - old_axis[good][0]))] = row[good][0]
        return np.moveaxis(output, -1, axis)

    @classmethod
    def resample_grid(cls, data, old_dx, old_dy, new_dx, new_dy):
        rows, cols = data.shape
        old_x = (np.arange(cols) - (cols - 1) / 2.0) * old_dx
        old_y = (np.arange(rows) - (rows - 1) / 2.0) * old_dy
        new_x = _odd_axis(max(abs(old_x[0]), abs(old_x[-1])), new_dx)
        new_y = _odd_axis(max(abs(old_y[0]), abs(old_y[-1])), new_dy)
        horizontal = cls._interp_axis(data, old_x, new_x, axis=1)
        return cls._interp_axis(horizontal, old_y, new_y, axis=0)
