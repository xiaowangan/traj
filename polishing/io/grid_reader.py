"""Migrated numerical service: GridReaderMixin; legacy numerical conventions retained."""
from __future__ import annotations
import os
import numpy as np

class GridReaderMixin:
    @classmethod
    def read_grid(cls, path):
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".dat":
            data, pixel = cls.read_zygo_dat(path)
            return np.flipud(data), pixel
        if suffix == ".npy":
            data = np.load(path)
        else:
            try:
                data = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
            except ValueError:
                data = np.genfromtxt(path, delimiter="," if suffix == ".csv" else None)
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("数据必须是二维矩阵")
        if data.shape[1] == 3 and data.shape[0] > 3:
            xs = np.unique(data[:, 0]); ys = np.unique(data[:, 1])
            if len(xs) * len(ys) != len(data):
                raise ValueError("三列点云必须位于规则 X/Y 网格上")
            grid = np.full((len(ys), len(xs)), np.nan)
            xi = np.searchsorted(xs, data[:, 0])
            yi = np.searchsorted(ys, data[:, 1])
            grid[yi, xi] = data[:, 2]
            pixel = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
            return grid, pixel
        return data, None
