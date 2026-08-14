# -*- coding: utf-8 -*-
"""Dwell-time feature UI and numerical pipeline ported from the MATLAB app."""
from __future__ import annotations

import json
import math
import os
import struct
from collections import deque

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)


def lineedit_input(label_text, default_value=""):
    label = QLabel(label_text)
    edit = QLineEdit(str(default_value))
    row = QHBoxLayout()
    row.addWidget(label)
    row.addWidget(edit)
    return edit, row


def combox_input(layout, label_text, widget):
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    row.addWidget(widget)
    layout.addLayout(row)


def _odd_axis(half_size, spacing):
    count = max(1, int(math.ceil(max(half_size, 0.0) / spacing)))
    return np.arange(-count, count + 1, dtype=float) * spacing


class DwellTimeEngine:
    """Pure NumPy implementation of the six-stage dwell-time workflow."""

    @staticmethod
    def rotate_vectors(vectors, axis, cosine, sine):
        """Rodrigues rotation without invoking a BLAS matrix product."""
        vectors = np.asarray(vectors, dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return (vectors * cosine + np.cross(axis, vectors) * sine +
                np.sum(vectors * axis, axis=-1, keepdims=True) * axis * (1.0 - cosine))

    @staticmethod
    def read_zygo_dat(path):
        """Read the big-endian Zygo DAT layout used by the source MLAPP."""
        with open(path, "rb") as stream:
            header = stream.read(4096)
            if len(header) < 238:
                raise ValueError("Zygo DAT 文件头不完整")

            def unpack(fmt, offset):
                return struct.unpack_from(">" + fmt, header, offset)[0]

            header_format = unpack("h", 4)
            intensity_bytes = unpack("i", 60)
            origin_x = unpack("h", 64)
            origin_y = unpack("h", 66)
            width = unpack("h", 68)
            height = unpack("h", 70)
            scale = unpack("f", 164)
            obliquity = unpack("f", 176)
            pixel_mm = unpack("f", 184) * 1000.0
            phase_code = unpack("h", 218)
            phase_resolution = 4096.0 if phase_code == 0 else 32768.0
            camera_width = unpack("h", 234)
            camera_height = unpack("h", 236)
            if min(width, height, camera_width, camera_height) <= 0:
                raise ValueError("Zygo DAT 尺寸字段无效")
            data_offset = intensity_bytes + (4096 if header_format == 3 else 834)
            stream.seek(data_offset)
            raw = stream.read(width * height * 4)
        if len(raw) != width * height * 4:
            raise ValueError("Zygo DAT 相位数据不完整")
        phase = np.frombuffer(raw, dtype=">i4").astype(float).reshape(height, width)
        phase[phase >= 2147483640] = np.nan
        phase *= scale * obliquity / phase_resolution
        full = np.full((camera_height, camera_width), np.nan, dtype=float)
        y1 = min(camera_height, origin_y + height)
        x1 = min(camera_width, origin_x + width)
        full[max(origin_y, 0):y1, max(origin_x, 0):x1] = phase[
            max(-origin_y, 0):height - max(origin_y + height - camera_height, 0),
            max(-origin_x, 0):width - max(origin_x + width - camera_width, 0),
        ]
        return full, float(pixel_mm)

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

    @staticmethod
    def stats(data):
        clean = np.asarray(data, dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return {"rms": float("nan"), "pv": float("nan"), "count": 0}
        centered = clean - np.mean(clean)
        return {
            "rms": float(np.sqrt(np.mean(centered * centered))),
            "pv": float(np.max(centered) - np.min(centered)),
            "count": int(clean.size),
        }

    @staticmethod
    def fft_convolve_same(data, kernel):
        data = np.asarray(data, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        shape = (data.shape[0] + kernel.shape[0] - 1,
                 data.shape[1] + kernel.shape[1] - 1)
        spectrum = np.fft.rfftn(data, shape) * np.fft.rfftn(kernel, shape)
        full = np.fft.irfftn(spectrum, shape)
        y0 = kernel.shape[0] // 2
        x0 = kernel.shape[1] // 2
        return full[y0:y0 + data.shape[0], x0:x0 + data.shape[1]]

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

    @staticmethod
    def asphere(x, y, radius, conic, coefficients=()):
        rp2 = x * x + y * y
        if abs(radius) < 1e-12:
            z = np.zeros_like(x, dtype=float)
        else:
            c = 1.0 / radius
            root = np.sqrt(np.maximum(0.0, 1.0 - (conic + 1.0) * c * c * rp2))
            z = -(c * rp2) / (1.0 + root)
        for index, value in enumerate(coefficients, 1):
            if np.isfinite(value) and value != 0:
                z -= value * rp2 ** (index + 1)
        return z

    @staticmethod
    def biconic(x, y, rx, ry, kx, ky, ax=(), ay=()):
        cx = 0.0 if abs(rx) < 1e-12 else 1.0 / rx
        cy = 0.0 if abs(ry) < 1e-12 else 1.0 / ry
        root = np.sqrt(np.maximum(
            0.0, 1.0 - (kx + 1.0) * cx * cx * x * x -
            (ky + 1.0) * cy * cy * y * y))
        z = -(cx * x * x + cy * y * y) / (1.0 + root)
        for index, value in enumerate(ax, 1):
            if np.isfinite(value) and value != 0:
                z -= value * x ** (2 * (index + 1))
        for index, value in enumerate(ay, 1):
            if np.isfinite(value) and value != 0:
                z -= value * y ** (2 * (index + 1))
        return z

    @classmethod
    def freeform(cls, x, y, rx, ry, kx, ky, axy=()):
        """MLAPP freeform basis: all x^(n-j)y^j terms for orders 1..9."""
        z = cls.biconic(x, y, rx, ry, kx, ky)
        coefficients = list(axy)[:54]
        index = 0
        for order in range(1, 10):
            for y_power in range(order + 1):
                value = coefficients[index] if index < len(coefficients) else 0.0
                if np.isfinite(value) and value != 0:
                    z -= value * x ** (order - y_power) * y ** y_power
                index += 1
        return z

    @classmethod
    def generate_model(cls, config):
        dx = float(config["dx"]); dy = float(config["dy"])
        aperture = config.get("aperture", "圆口径")
        diameter = float(config.get("diameter", 0.0))
        lx = float(config.get("lx", diameter)); ly = float(config.get("ly", diameter))
        span_x = diameter if aperture != "矩形" else lx
        span_y = diameter if aperture != "矩形" else ly
        if span_x <= 0 or span_y <= 0:
            raise ValueError("请设置有效的口径尺寸")
        x = _odd_axis(span_x / 2.0, dx)
        y = _odd_axis(span_y / 2.0, dy)
        x_local, y_local = np.meshgrid(x, y)
        bx = float(config.get("x_offset", 0.0)); by = float(config.get("y_offset", 0.0))
        xm = x_local + bx; ym = y_local + by
        kind = config.get("type", "非球面")
        if kind == "平面":
            # 平面即曲率半径无穷大：强制 R=0/K=0/无高次项
            config = dict(config, r=0.0, k=0.0, a=())
        if kind in ("平面", "球面", "非球面"):
            z = cls.asphere(xm, ym, float(config.get("r", 0.0)),
                            float(config.get("k", 0.0)), config.get("a", ()))
            z0 = cls.asphere(np.array(bx), np.array(by), float(config.get("r", 0.0)),
                             float(config.get("k", 0.0)), config.get("a", ()))
        elif kind == "自由曲面":
            z = cls.freeform(xm, ym, float(config.get("rx", 0.0)),
                             float(config.get("ry", 0.0)), float(config.get("kx", 0.0)),
                             float(config.get("ky", 0.0)), config.get("axy", ()))
            z0 = cls.freeform(np.array(bx), np.array(by), float(config.get("rx", 0.0)),
                              float(config.get("ry", 0.0)), float(config.get("kx", 0.0)),
                              float(config.get("ky", 0.0)), config.get("axy", ()))
        else:
            z = cls.biconic(xm, ym, float(config.get("rx", 0.0)),
                            float(config.get("ry", 0.0)), float(config.get("kx", 0.0)),
                            float(config.get("ky", 0.0)), config.get("ax", ()), config.get("ay", ()))
            z0 = cls.biconic(np.array(bx), np.array(by), float(config.get("rx", 0.0)),
                             float(config.get("ry", 0.0)), float(config.get("kx", 0.0)),
                             float(config.get("ky", 0.0)), config.get("ax", ()), config.get("ay", ()))
        gy, gx = np.gradient(z, dy, dx)
        normals = np.stack((-gx, -gy, np.ones_like(z)), axis=-1)
        points = np.stack((x_local, y_local, z - float(z0)), axis=-1)
        tilt = math.radians(float(config.get("tilt", 0.0)))
        gamma = math.atan2(by, bx) if abs(bx) + abs(by) > 0 else 0.0
        if abs(tilt) > 1e-12:
            axis = np.array([-math.sin(gamma), math.cos(gamma), 0.0])
            points = cls.rotate_vectors(points, axis, math.cos(tilt), math.sin(tilt))
            normals = cls.rotate_vectors(normals, axis, math.cos(tilt), math.sin(tilt))
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals /= np.maximum(norm, 1e-12)
        xx = points[..., 0]; yy = points[..., 1]
        if aperture == "圆口径":
            inner = max(float(config.get("inner_diameter", 0.0)), 0.0)
            mask = (xx * xx + yy * yy <= (diameter / 2.0) ** 2 + 1e-12)
            if inner > 0:
                mask &= xx * xx + yy * yy >= (inner / 2.0) ** 2
        elif aperture == "矩形":
            corner = max(float(config.get("corner", 0.0)), 0.0)
            qx = np.maximum(np.abs(xx) - (lx / 2.0 - corner), 0.0)
            qy = np.maximum(np.abs(yy) - (ly / 2.0 - corner), 0.0)
            mask = ((np.abs(xx) <= lx / 2.0) & (np.abs(yy) <= ly / 2.0) &
                    (qx * qx + qy * qy <= corner * corner + 1e-12))
        else:
            sides = max(3, int(config.get("sides", 6)))
            angle = math.radians(float(config.get("first_angle", 0.0)))
            theta = np.arctan2(yy, xx) - angle
            radial_limit = (diameter / 2.0) * math.cos(math.pi / sides) / np.maximum(
                np.cos((theta + math.pi / sides) % (2 * math.pi / sides) - math.pi / sides), 1e-12)
            mask = np.hypot(xx, yy) <= radial_limit
        points[~mask] = np.nan; normals[~mask] = np.nan
        return {"points": points, "normals": normals, "mask": mask,
                "dx": dx, "dy": dy, "config": dict(config)}

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
            ellipse = ((old_x[None, :] - x_offset) / max(abs(old_x[-1] - x_offset), 1e-12)) ** 2
            ellipse += ((old_y[:, None] - y_offset) / max(abs(old_y[-1] - y_offset), 1e-12)) ** 2
            source[ellipse >= 1.0] = 0.0
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

    @classmethod
    def solve_bounded_least_squares(cls, surface, kernel, dx, dy,
                                    uniform=0.0, min_dwell=0.0, max_dwell=np.inf,
                                    trajectory_trim=0.0, evaluation_trim=0.0,
                                    iterations=300, tolerance=1e-6):
        surface = np.asarray(surface, dtype=float)
        kernel = np.asarray(kernel, dtype=float)
        if surface.ndim != 2 or kernel.ndim != 2:
            raise ValueError("面形和抛光斑必须是二维矩阵")
        if not np.isfinite(surface).any() or not np.any(kernel > 0):
            raise ValueError("面形或抛光斑没有有效数据")
        hy = kernel.shape[0] // 2; hx = kernel.shape[1] // 2
        work = np.pad(surface, ((hy, hy), (hx, hx)), constant_values=np.nan)
        surface_mask = np.isfinite(work)
        trajectory_mask = cls.erode_mask(surface_mask, max(trajectory_trim, 0.0), dx, dy)
        evaluation_mask = cls.erode_mask(surface_mask, max(evaluation_trim, 0.0), dx, dy)
        if not trajectory_mask.any() or not evaluation_mask.any():
            raise ValueError("轨迹区或评价区为空，请减小裁边量")
        target = np.zeros_like(work)
        base = float(np.nanmin(work))
        target[surface_mask] = work[surface_mask] - base + float(uniform)
        lower = max(float(min_dwell), 0.0)
        upper = float(max_dwell)
        if upper < lower:
            raise ValueError("最大驻留时间不能小于最小驻留时间")
        dwell = np.zeros_like(work)
        dwell[trajectory_mask] = lower
        estimate = dwell.copy()
        momentum = 1.0
        lipschitz = max(float(np.sum(np.abs(kernel))) ** 2, 1e-12)
        objective = []
        for _ in range(max(1, int(iterations))):
            removal_est = cls.fft_convolve_same(estimate, kernel)
            error = np.zeros_like(work)
            error[evaluation_mask] = removal_est[evaluation_mask] - target[evaluation_mask]
            gradient = cls.fft_convolve_same(error, np.flip(kernel, axis=(0, 1)))
            candidate = estimate - gradient / lipschitz
            candidate[trajectory_mask] = np.clip(candidate[trajectory_mask], lower, upper)
            candidate[~trajectory_mask] = 0.0
            new_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
            accelerated = candidate + ((momentum - 1.0) / new_momentum) * (candidate - dwell)
            accelerated[~trajectory_mask] = 0.0
            delta = np.linalg.norm(candidate - dwell)
            scale = max(np.linalg.norm(dwell), 1.0)
            dwell = candidate; estimate = accelerated; momentum = new_momentum
            if len(objective) < 20 or len(objective) % 10 == 0:
                residual_now = removal_est[evaluation_mask] - target[evaluation_mask]
                objective.append(float(np.dot(residual_now, residual_now)))
            if delta / scale < tolerance:
                break
        removal = cls.fft_convolve_same(dwell, kernel)
        residual = np.full_like(work, np.nan)
        residual[surface_mask] = work[surface_mask] - removal[surface_mask]
        before = work[evaluation_mask]
        after = residual[evaluation_mask]
        before_rms = float(np.sqrt(np.mean((before - np.mean(before)) ** 2)))
        after_rms = float(np.sqrt(np.mean((after - np.mean(after)) ** 2)))
        display_dwell = dwell.copy(); display_dwell[~trajectory_mask] = np.nan
        x = (np.arange(work.shape[1]) - (work.shape[1] - 1) / 2.0) * dx
        y = (np.arange(work.shape[0]) - (work.shape[0] - 1) / 2.0) * dy
        return {
            "dwell": display_dwell, "removal": removal, "residual": residual,
            "target": target, "surface_mask": surface_mask,
            "trajectory_mask": trajectory_mask, "evaluation_mask": evaluation_mask,
            "x": x, "y": y, "before_rms": before_rms, "after_rms": after_rms,
            "iterations": len(objective), "objective": objective,
        }

    @staticmethod
    def sample_nearest(model, xs, ys):
        points = model["points"].reshape(-1, 3)
        normals = model["normals"].reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        points = points[valid]; normals = normals[valid]
        query = np.column_stack((xs, ys))
        out_p = np.full((len(query), 3), np.nan); out_n = np.full((len(query), 3), np.nan)
        for start in range(0, len(query), 256):
            q = query[start:start + 256]
            dist = (q[:, None, 0] - points[None, :, 0]) ** 2 + (
                q[:, None, 1] - points[None, :, 1]) ** 2
            idx = np.argmin(dist, axis=1)
            out_p[start:start + len(q)] = points[idx]
            out_n[start:start + len(q)] = normals[idx]
        return out_p, out_n

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

    @staticmethod
    def generate_cnc(trajectory, model_points, model_normals, config, grid_dx, grid_dy):
        xyz = np.asarray(model_points, dtype=float)
        normals = np.asarray(model_normals, dtype=float)
        dwell = np.asarray(trajectory[:, 2], dtype=float)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & np.isfinite(dwell)
        xyz = xyz[valid]; normals = normals[valid]; dwell = dwell[valid]
        if len(xyz) < 2:
            raise ValueError("生成 CNC 至少需要两个有效轨迹点")
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        alpha = math.radians(float(config.get("alpha", 0.0)))
        beta = math.radians(float(config.get("beta", 0.0)))
        tool_flat = np.array([
            math.sin(beta) * math.cos(alpha),
            -math.sin(alpha),
            math.cos(beta) * math.cos(alpha),
        ])
        tool_vectors = np.empty_like(normals)
        z_axis = np.array([0.0, 0.0, 1.0])
        for index, normal in enumerate(normals):
            axis = np.cross(z_axis, normal); axis_norm = np.linalg.norm(axis)
            cosine = float(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
            if axis_norm < 1e-12:
                tool_vectors[index] = tool_flat if cosine >= 0 else np.array(
                    [tool_flat[0], -tool_flat[1], -tool_flat[2]])
            else:
                tool_vectors[index] = DwellTimeEngine.rotate_vectors(
                    tool_flat, axis / axis_norm, cosine,
                    math.sqrt(max(0.0, 1.0 - cosine * cosine)))
        tool_vectors /= np.maximum(np.linalg.norm(tool_vectors, axis=1, keepdims=True), 1e-12)
        angle_x = np.degrees(np.arcsin(np.clip(-tool_vectors[:, 1], -1.0, 1.0)))
        angle_y = np.degrees(np.arcsin(np.clip(
            tool_vectors[:, 0] / np.maximum(np.hypot(tool_vectors[:, 0], tool_vectors[:, 2]), 1e-12), -1.0, 1.0)))
        radius = float(config.get("tool_radius", 0.0)); depth = float(config.get("depth", 0.0))
        offset = np.array([float(config.get("x_offset", 0.0)),
                           float(config.get("y_offset", 0.0)),
                           float(config.get("z_offset", 0.0))])
        path = xyz + radius * normals - depth * tool_vectors + offset
        path[:, 2] -= radius
        step_x = float(config.get("step_x", grid_dx)); step_y = float(config.get("step_y", grid_dy))
        point_dwell = dwell / max(grid_dx * grid_dy, 1e-12) * step_x * step_y
        point_dwell /= np.maximum(np.abs(normals[:, 2]), 1e-9)
        delta = np.diff(path, axis=0); distance = np.linalg.norm(delta, axis=1)
        feed = np.empty(len(path), dtype=float)
        feed[1:] = 2.0 * distance / np.maximum(point_dwell[1:] + point_dwell[:-1], 1e-12)
        feed[0] = feed[1]
        ideal_feed = feed.copy()
        feed = np.clip(feed, float(config.get("min_speed", 0.01)), float(config.get("max_speed", 50.0)))
        total_time = float(np.sum(distance / np.maximum(feed[1:], 1e-12)))
        body = np.column_stack((path, angle_x, angle_y, feed))
        fast = float(config.get("max_speed", 50.0)) / 2.0
        end_lift = np.array([path[-1, 0], path[-1, 1], path[-1, 2] + 2 * depth + 20,
                             angle_x[-1], angle_y[-1], fast])
        header = np.array([
            [path[0, 0], path[0, 1], path[0, 2] + 100, angle_x[0], angle_y[0], fast],
            [path[0, 0], path[0, 1], path[0, 2], angle_x[0], angle_y[0], fast],
            end_lift,
            [path[-1, 0], path[-1, 1], path[-1, 2] + 100, angle_x[-1], angle_y[-1], fast],
        ])
        return {"data": np.vstack((header, body, end_lift)), "path": path,
                "feed": feed, "ideal_feed": ideal_feed, "total_time": total_time}


class DwellTimeMixin:
    """UI/controller mixin for the top-level dwell-time feature."""

    def _init_dwell_state(self):
        self._dwell_files = {}
        self._dwell_state = {
            "dx": 1.0, "dy": 1.0, "model": None, "surface_raw": None,
            "surface": None, "surface_pixel": None, "spot_raw": None,
            "spot_pixel": None, "spot": None, "solution": None,
            "trajectory": None, "cnc": None,
        }

    @staticmethod
    def _dwell_float(edit, name, minimum=None, allow_equal=True):
        try:
            value = float(edit.text())
        except ValueError as exc:
            raise ValueError(f"参数“{name}”输入无效") from exc
        if minimum is not None:
            invalid = value < minimum if allow_equal else value <= minimum
            if invalid:
                relation = "不小于" if allow_equal else "大于"
                raise ValueError(f"参数“{name}”必须{relation} {minimum}")
        return value

    @staticmethod
    def _dwell_coefficients(edit, name, limit=None):
        text = edit.text().strip()
        if not text:
            return []
        try:
            values = [float(item) for item in text.replace(";", ",").replace(" ", ",").split(",")
                      if item.strip()]
        except ValueError as exc:
            raise ValueError(f"参数“{name}”必须是用逗号分隔的数字") from exc
        if limit is not None and len(values) > limit:
            raise ValueError(f"参数“{name}”最多允许 {limit} 项")
        return values

    def _dwell_scroll_page(self, title, subtitle):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(6); layout.setContentsMargins(8, 8, 8, 8)
        # 功能介绍不再放在右侧控制台，而是仿照轨迹规划 "Trajectory | ..." 覆盖条，
        # 作为独立横条显示在左侧预览区最上方；横幅文本随页面携带，切换页面时同步。
        scroll.dwell_banner_text = f" {title} | {subtitle}"
        return scroll, layout

    def _dwell_add_fields(self, layout, fields, readonly=()):
        for attr, label, default in fields:
            edit, row = lineedit_input(label, default)
            if attr in readonly:
                edit.setReadOnly(True)
                edit.setStyleSheet("background:#e8f0fa; color:#10243f;")
            setattr(self, attr, edit); layout.addLayout(row)

    def _dwell_report(self, message, label=None, error=False):
        if label is not None:
            label.setText(("✘ " if error else "✔ ") + message)
            label.setStyleSheet(
                f"color:{'#c0392b' if error else '#1a7a3c'}; font-size:11px;")
        self._main.statusbar.showMessage(message)
        self._main.set_status(message)
        self._main.terminal_output.appendPlainText("[驻留时间] " + message)

    def _dwell_error(self, title, error, label=None):
        self._dwell_report(str(error), label, error=True)
        QMessageBox.warning(self._main, title, str(error))

    def _build_dwell_initial_page(self):
        scroll, layout = self._dwell_scroll_page(
            "初始设置", "设置抛光工具、计算网格和工作路径；网格参数会同步到后续五个页面。")

        # ── 抛光工具 ──
        tool_group = QGroupBox("抛光工具"); tool_form = QVBoxLayout(tool_group)
        self.dw_tool = QComboBox(); self.dw_tool.addItems(["小工具", "气囊"])
        combox_input(tool_form, "抛光工具：", self.dw_tool)
        self.dw_stage = QComboBox(); self.dw_stage.addItems(["粗抛", "平滑"])
        combox_input(tool_form, "抛光阶段：", self.dw_stage)
        layout.addWidget(tool_group)

        # ── 计算网格（仅此处可编辑，同步到其余页面）──
        grid_group = QGroupBox("计算网格"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_dx", "X 间距 (mm)：", "1"), ("dw_dy", "Y 间距 (mm)：", "1"),
        ])
        layout.addWidget(grid_group)

        # ── 工作路径 ──
        work_group = QGroupBox("工作路径"); work_form = QVBoxLayout(work_group)
        self._dwell_add_fields(work_form, [
            ("dw_work_dir", "工作路径：", ""),
        ])
        choose = QPushButton("选择工作目录")
        choose.clicked.connect(self._dwell_choose_work_dir); work_form.addWidget(choose)
        layout.addWidget(work_group)

        # ── 抛光盘参数 ──
        disk_group = QGroupBox("抛光盘参数"); disk_form = QVBoxLayout(disk_group)
        self._dwell_add_fields(disk_form, [
            ("dw_disk_material", "抛光盘材料：", ""),
            ("dw_disk_diameter", "抛光盘口径 (mm)：", ""),
            ("dw_disk_speed", "抛光盘转速 (rpm)：", ""),
            ("dw_pressure", "抛光压力：", ""),
            ("dw_feed", "进给速度 (mm/min)：", ""),
        ])
        layout.addWidget(disk_group)

        apply_button = QPushButton("输入初始设置")
        apply_button.clicked.connect(self._dwell_apply_initial); layout.addWidget(apply_button)
        self.dw_initial_info = QLabel(""); self.dw_initial_info.setWordWrap(True)
        layout.addWidget(self.dw_initial_info); layout.addStretch()
        return scroll

    def _dwell_choose_work_dir(self):
        path = QFileDialog.getExistingDirectory(
            self._main, "选择驻留时间工作目录", self.dw_work_dir.text().strip())
        if path: self.dw_work_dir.setText(path)

    def _dwell_apply_initial(self):
        try:
            dx = self._dwell_float(self.dw_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_dy, "Y间距", 0.0, False)
            work_dir = self.dw_work_dir.text().strip()
            if work_dir and not os.path.isdir(work_dir):
                raise ValueError("工作路径不存在")
            self._dwell_state.update(dx=dx, dy=dy, tool=self.dw_tool.currentText(),
                                     stage=self.dw_stage.currentText(), work_dir=work_dir)
            for attr, value in (("dw_surface_dx", dx), ("dw_surface_dy", dy),
                                ("dw_spot_dx", dx), ("dw_spot_dy", dy),
                                ("dw_solve_dx", dx), ("dw_solve_dy", dy),
                                ("dw_model_dx", dx), ("dw_model_dy", dy)):
                if hasattr(self, attr): getattr(self, attr).setText(f"{value:g}")
            self._dwell_report(f"初始设置已生效：网格 {dx:g} × {dy:g} mm。", self.dw_initial_info)
        except ValueError as exc:
            self._dwell_error("初始设置", exc, self.dw_initial_info)

    def _build_dwell_model_page(self):
        scroll, layout = self._dwell_scroll_page(
            "建模", "根据面形参数在扩展区域内生成完整曲面，再按离轴量与口径尺寸裁出模型。")

        # ── 镜面口径选择：口径决定镜子的轮廓与尺寸，选类型后只显示对应参数 ──
        aperture_group = QGroupBox("镜面口径选择"); aperture_form = QVBoxLayout(aperture_group)
        self.dw_aperture = QComboBox()
        self.dw_aperture.addItems(["—— 请选择口径类型 ——", "圆口径", "矩形", "正多边形"])
        combox_input(aperture_form, "口径类型：", self.dw_aperture)
        self.dw_box_diameter = self._dwell_field_box(
            aperture_form, "dw_model_diameter", "外径/外接直径 (mm)：", "100")
        self.dw_box_inner = self._dwell_field_box(
            aperture_form, "dw_model_inner", "内径 (mm)：", "0")
        self.dw_box_lx = self._dwell_field_box(aperture_form, "dw_model_lx", "LX (mm)：", "100")
        self.dw_box_ly = self._dwell_field_box(aperture_form, "dw_model_ly", "LY (mm)：", "100")
        self.dw_box_corner = self._dwell_field_box(
            aperture_form, "dw_model_corner", "圆角半径 (mm)：", "0")
        self.dw_box_sides = self._dwell_field_box(aperture_form, "dw_model_sides", "边数：", "6")
        self.dw_box_first_angle = self._dwell_field_box(
            aperture_form, "dw_model_first_angle", "首顶点角度 (°)：", "0")
        layout.addWidget(aperture_group)

        # ── 面形参数：面形类型决定母镜表面形状，选类型后只显示对应参数 ──
        surface_group = QGroupBox("面形参数"); form = QVBoxLayout(surface_group)
        self.dw_model_load = QComboBox(); self.dw_model_load.addItems(["否", "是"])
        combox_input(form, "读取面形参数：", self.dw_model_load)
        self.dw_model_load.currentIndexChanged.connect(self._dwell_load_combo_changed)
        self.dw_model_type = QComboBox()
        self.dw_model_type.addItems(
            ["—— 请选择面形类型 ——", "平面", "球面", "非球面", "柱面/双曲率", "自由曲面"])
        combox_input(form, "面形类型：", self.dw_model_type)
        self.dw_box_plane_hint = QWidget()
        hint_layout = QVBoxLayout(self.dw_box_plane_hint)
        hint_layout.setContentsMargins(0, 0, 0, 0); hint_layout.setSpacing(0)
        hint = QLabel("平面无需面形参数,其半径R为无穷。")
        hint.setStyleSheet("color:#50647a; font-size:11px;")
        hint_layout.addWidget(hint)
        form.addWidget(self.dw_box_plane_hint)
        self.dw_box_r = self._dwell_field_box(form, "dw_model_r", "曲率半径 R (mm)：", "100")
        self.dw_box_k = self._dwell_field_box(form, "dw_model_k", "圆锥常数 K：", "0")
        self.dw_box_a4 = self._dwell_field_box(form, "dw_model_a4", "A4：", "0")
        self.dw_box_a6 = self._dwell_field_box(form, "dw_model_a6", "A6：", "0")
        self.dw_box_a8 = self._dwell_field_box(form, "dw_model_a8", "A8：", "0")
        self.dw_box_rx = self._dwell_field_box(form, "dw_model_rx", "X曲率半径 Rx (mm)：", "100")
        self.dw_box_kx = self._dwell_field_box(form, "dw_model_kx", "X圆锥常数 Kx：", "0")
        self.dw_box_ry = self._dwell_field_box(form, "dw_model_ry", "Y曲率半径 Ry (mm)：", "100")
        self.dw_box_ky = self._dwell_field_box(form, "dw_model_ky", "Y圆锥常数 Ky：", "0")
        self.dw_box_ax = self._dwell_field_box(
            form, "dw_model_ax", "双曲率 Ax 高次项（逗号分隔）：", "")
        self.dw_box_ay = self._dwell_field_box(
            form, "dw_model_ay", "双曲率 Ay 高次项（逗号分隔）：", "")
        self.dw_box_axy = self._dwell_field_box(
            form, "dw_model_axy", "自由曲面 Axy 1～9阶（最多54项）：", "")
        layout.addWidget(surface_group)

        # ── 离轴参数：单独成框，决定从母镜上切取离轴段的位置与姿态 ──
        offaxis_group = QGroupBox("离轴参数"); offaxis_form = QVBoxLayout(offaxis_group)
        self._dwell_add_fields(offaxis_form, [
            ("dw_model_xoff", "X 离轴量 (mm)：", "0"),
            ("dw_model_yoff", "Y 离轴量 (mm)：", "0"),
            ("dw_model_tilt", "倾斜角 (°)：", "0"),
        ])
        # 远轴角位 γ = atan2(By, Bx)：由离轴量自动算出，只读
        self.dw_model_gamma = QLineEdit("0")
        self.dw_model_gamma.setReadOnly(True)
        self.dw_model_gamma.setStyleSheet("background:#e8f0fa; color:#10243f;")
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("远轴角位 (°)："))
        gamma_row.addWidget(self.dw_model_gamma)
        offaxis_form.addLayout(gamma_row)
        self.dw_model_xoff.textChanged.connect(self._dwell_sync_gamma)
        self.dw_model_yoff.textChanged.connect(self._dwell_sync_gamma)
        self._dwell_sync_gamma()
        layout.addWidget(offaxis_group)

        # ── 计算网格：只读镜像初始设置的全局 DX/DY，不参与建模计算 ──
        grid_group = QGroupBox("计算网格"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_model_dx", "X 间距 (mm)：", "1"),
            ("dw_model_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_model_dx", "dw_model_dy"))
        layout.addWidget(grid_group)

        self._model_box_attrs = (
            "dw_box_diameter", "dw_box_inner", "dw_box_lx", "dw_box_ly", "dw_box_corner",
            "dw_box_sides", "dw_box_first_angle", "dw_box_plane_hint",
            "dw_box_r", "dw_box_k", "dw_box_a4", "dw_box_a6", "dw_box_a8",
            "dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky",
            "dw_box_ax", "dw_box_ay", "dw_box_axy",
        )
        self.dw_aperture.currentIndexChanged.connect(self._dwell_model_sync_visibility)
        self.dw_model_type.currentIndexChanged.connect(self._dwell_model_sync_visibility)
        self._dwell_model_sync_visibility()

        row = QHBoxLayout()
        generate = QPushButton("生成曲面模型"); save = QPushButton("保存面形参数")
        clear = QPushButton("清空")
        for button in (generate, save, clear): row.addWidget(button)
        layout.addLayout(row)
        self.dw_model_info = QLabel(""); self.dw_model_info.setWordWrap(True)
        layout.addWidget(self.dw_model_info); layout.addStretch()
        generate.clicked.connect(self._dwell_generate_model)
        save.clicked.connect(self._dwell_save_model_config)
        clear.clicked.connect(self._clear_dwell_model_fields)
        return scroll

    def _dwell_field_box(self, layout, attr, label, default):
        edit, row = lineedit_input(label, default)
        row.setContentsMargins(0, 0, 0, 0)
        setattr(self, attr, edit)
        box = QWidget(); box.setLayout(row)
        layout.addWidget(box)
        return box

    def _dwell_model_sync_visibility(self, *_args):
        """按口径/面形下拉框的当前选择，只显示该类型需要填写的参数行。"""
        aperture_visible = {
            "圆口径": ("dw_box_diameter", "dw_box_inner"),
            "矩形": ("dw_box_lx", "dw_box_ly", "dw_box_corner"),
            "正多边形": ("dw_box_diameter", "dw_box_sides", "dw_box_first_angle"),
        }
        surface_visible = {
            "平面": ("dw_box_plane_hint",),
            "球面": ("dw_box_r", "dw_box_k"),
            "非球面": ("dw_box_r", "dw_box_k", "dw_box_a4", "dw_box_a6", "dw_box_a8"),
            "柱面/双曲率": ("dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky",
                          "dw_box_ax", "dw_box_ay"),
            "自由曲面": ("dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky", "dw_box_axy"),
        }
        shown = set(aperture_visible.get(self.dw_aperture.currentText(), ()))
        shown |= set(surface_visible.get(self.dw_model_type.currentText(), ()))
        for attr in self._model_box_attrs:
            getattr(self, attr).setVisible(attr in shown)

    def _dwell_sync_gamma(self, *_args):
        """远轴角位 γ = atan2(By, Bx)，由离轴量自动算出（只读显示）。"""
        try:
            bx = float(self.dw_model_xoff.text())
        except ValueError:
            bx = 0.0
        try:
            by = float(self.dw_model_yoff.text())
        except ValueError:
            by = 0.0
        gamma = math.degrees(math.atan2(by, bx)) if abs(bx) + abs(by) > 0 else 0.0
        self.dw_model_gamma.setText(f"{gamma:.3f}")

    def _dwell_load_combo_changed(self, index):
        """面形参数组顶部的“是否读取”开关：选“是”时读取参数存档并复位。"""
        if self.dw_model_load.currentText() != "是":
            return
        self.dw_model_load.blockSignals(True)
        try:
            self._dwell_load_model_config()
        finally:
            self.dw_model_load.setCurrentIndex(0)
            self.dw_model_load.blockSignals(False)

    def _dwell_model_config(self):
        aperture = self.dw_aperture.currentText()
        if aperture not in ("圆口径", "矩形", "正多边形"):
            raise ValueError("请先选择口径类型")
        kind = self.dw_model_type.currentText()
        if kind not in ("平面", "球面", "非球面", "柱面/双曲率", "自由曲面"):
            raise ValueError("请先选择面形类型")
        return {
            "type": kind, "aperture": aperture,
            "r": self._dwell_float(self.dw_model_r, "R"), "k": self._dwell_float(self.dw_model_k, "K"),
            "rx": self._dwell_float(self.dw_model_rx, "Rx"), "kx": self._dwell_float(self.dw_model_kx, "Kx"),
            "ry": self._dwell_float(self.dw_model_ry, "Ry"), "ky": self._dwell_float(self.dw_model_ky, "Ky"),
            "a": [self._dwell_float(self.dw_model_a4, "A4"),
                  self._dwell_float(self.dw_model_a6, "A6"), self._dwell_float(self.dw_model_a8, "A8")],
            "ax": self._dwell_coefficients(self.dw_model_ax, "Ax"),
            "ay": self._dwell_coefficients(self.dw_model_ay, "Ay"),
            "axy": self._dwell_coefficients(self.dw_model_axy, "Axy", 54),
            "x_offset": self._dwell_float(self.dw_model_xoff, "X离轴量"),
            "y_offset": self._dwell_float(self.dw_model_yoff, "Y离轴量"),
            "tilt": self._dwell_float(self.dw_model_tilt, "倾斜角"),
            "dx": float(self._dwell_state.get("dx", 1.0)),
            "dy": float(self._dwell_state.get("dy", 1.0)),
            "diameter": self._dwell_float(self.dw_model_diameter, "外径", 0.0, False),
            "inner_diameter": self._dwell_float(self.dw_model_inner, "内径", 0.0),
            "lx": self._dwell_float(self.dw_model_lx, "LX", 0.0, False),
            "ly": self._dwell_float(self.dw_model_ly, "LY", 0.0, False),
            "corner": self._dwell_float(self.dw_model_corner, "圆角半径", 0.0),
            "sides": int(self._dwell_float(self.dw_model_sides, "边数", 3.0)),
            "first_angle": self._dwell_float(self.dw_model_first_angle, "首顶点角度"),
        }

    def _dwell_generate_model(self):
        try:
            config = self._dwell_model_config()
            model = DwellTimeEngine.generate_model(config)
            self._dwell_state["model"] = model
            valid = int(np.count_nonzero(model["mask"]))
            self._dwell_report(f"曲面模型生成完成：{valid} 个有效节点。", self.dw_model_info)
            try:
                self._main.preview.plot_dwell_model(
                    model, f"{config['type']} | 口径类型 {config['aperture']}")
            except Exception as exc:
                self._dwell_report(f"三维预览失败：{exc}", self.dw_model_info, error=True)
        except Exception as exc:
            self._dwell_error("建模失败", exc, self.dw_model_info)

    def _dwell_save_model_config(self):
        try: config = self._dwell_model_config()
        except ValueError as exc:
            self._dwell_error("保存失败", exc, self.dw_model_info); return
        path, _ = QFileDialog.getSaveFileName(self._main, "保存面形参数", "dwell_model.json", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as stream: json.dump(config, stream, ensure_ascii=False, indent=2)
            self._dwell_report(f"面形参数已保存：{os.path.basename(path)}", self.dw_model_info)
        except OSError as exc: self._dwell_error("保存失败", exc, self.dw_model_info)

    def _dwell_load_model_config(self):
        path, _ = QFileDialog.getOpenFileName(self._main, "读取面形参数", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as stream: config = json.load(stream)
            mapping = {
                "r":"dw_model_r", "k":"dw_model_k", "rx":"dw_model_rx", "kx":"dw_model_kx",
                "ry":"dw_model_ry", "ky":"dw_model_ky", "x_offset":"dw_model_xoff",
                "y_offset":"dw_model_yoff", "tilt":"dw_model_tilt",
                "diameter":"dw_model_diameter", "inner_diameter":"dw_model_inner", "lx":"dw_model_lx",
                "ly":"dw_model_ly", "corner":"dw_model_corner", "sides":"dw_model_sides",
                "first_angle":"dw_model_first_angle",
            }
            for key, attr in mapping.items():
                if key in config: getattr(self, attr).setText(str(config[key]))
            if "type" in config: self.dw_model_type.setCurrentText(config["type"])
            if "aperture" in config: self.dw_aperture.setCurrentText(config["aperture"])
            values = list(config.get("a", [])) + [0, 0, 0]
            for attr, value in zip(("dw_model_a4", "dw_model_a6", "dw_model_a8"), values): getattr(self, attr).setText(str(value))
            for key, attr in (("ax", "dw_model_ax"), ("ay", "dw_model_ay"), ("axy", "dw_model_axy")):
                getattr(self, attr).setText(",".join(str(value) for value in config.get(key, [])))
            self._dwell_report(f"面形参数已读取：{os.path.basename(path)}", self.dw_model_info)
        except Exception as exc: self._dwell_error("读取失败", exc, self.dw_model_info)

    def _clear_dwell_model_fields(self):
        for attr in ("dw_model_r", "dw_model_k", "dw_model_rx", "dw_model_kx", "dw_model_ry",
                     "dw_model_ky", "dw_model_a4", "dw_model_a6", "dw_model_a8", "dw_model_ax",
                     "dw_model_ay", "dw_model_axy", "dw_model_xoff",
                     "dw_model_yoff", "dw_model_tilt", "dw_model_diameter", "dw_model_inner",
                     "dw_model_lx", "dw_model_ly", "dw_model_corner",
                     "dw_model_sides", "dw_model_first_angle"):
            getattr(self, attr).clear()
        self.dw_aperture.setCurrentIndex(0)
        self.dw_model_type.setCurrentIndex(0)
        self._dwell_state["model"] = None
        self._dwell_report("建模参数和模型缓存已清空。", self.dw_model_info)

    def _build_dwell_surface_page(self):
        scroll, layout = self._dwell_scroll_page(
            "面形数据", "读取 Zygo DAT、NPY、CSV 或规则文本网格，并执行裁边、均值滤波、RMS 滤波和延拓。")

        # ── 数据导入：选择文件后读入面形矩阵 ──
        import_group = QGroupBox("数据导入"); import_form = QVBoxLayout(import_group)
        self._dwell_add_fields(import_form, [
            ("dw_surface_file", "文件路径：", ""),
        ])
        select = QPushButton("选择面形数据文件"); select.clicked.connect(self._dwell_select_surface)
        import_form.addWidget(select)
        layout.addWidget(import_group)

        # ── 处理参数：裁边、延拓与滤波 ──
        process_group = QGroupBox("处理参数"); process_form = QVBoxLayout(process_group)
        self._dwell_add_fields(process_form, [
            ("dw_surface_trim", "裁边尺寸 (mm)：", "0"),
            ("dw_surface_extend", "延拓尺寸 (mm)：", "0"),
            ("dw_surface_filter", "均值滤波窗尺寸 (mm)：", "2"),
            ("dw_surface_rms", "RMS 倍数：", "3"),
        ])
        layout.addWidget(process_group)

        # ── 网格间距（只读，来自初始设置）──
        grid_group = QGroupBox("网格间距"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_surface_dx", "X 间距 (mm)：", "1"),
            ("dw_surface_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_surface_dx", "dw_surface_dy"))
        layout.addWidget(grid_group)

        row = QHBoxLayout()
        actions = (("裁边", self._dwell_trim_surface), ("延拓/填充", self._dwell_extend_surface),
                   ("均值滤波", self._dwell_mean_surface), ("RMS 滤波", self._dwell_rms_surface))
        for caption, callback in actions:
            button = QPushButton(caption); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        self.dw_surface_info = QLabel(""); self.dw_surface_info.setWordWrap(True)
        layout.addWidget(self.dw_surface_info); layout.addStretch()
        return scroll

    def _dwell_select_surface(self):
        path, _ = QFileDialog.getOpenFileName(
            self._main, "选择面形数据", self.dw_surface_file.text().strip(),
            "数据文件 (*.dat *.npy *.csv *.txt);;所有文件 (*.*)")
        if not path: return
        try:
            raw, pixel = DwellTimeEngine.read_grid(path)
            dx = self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)
            data = raw if pixel is None else DwellTimeEngine.resample_grid(raw, pixel, pixel, dx, dy)
            self.dw_surface_file.setText(path)
            self._dwell_state.update(surface_raw=raw, surface=data, surface_pixel=pixel, dx=dx, dy=dy)
            stat = DwellTimeEngine.stats(data)
            self._dwell_report(
                f"面形已读取：{data.shape[1]}×{data.shape[0]}，PV={stat['pv']:.6g}，RMS={stat['rms']:.6g}。",
                self.dw_surface_info)
            self._dwell_refresh_surface_preview()
        except Exception as exc: self._dwell_error("面形读取失败", exc, self.dw_surface_info)

    def _require_surface(self):
        data = self._dwell_state.get("surface")
        if data is None: raise ValueError("请先选择面形数据文件")
        return data

    def _dwell_refresh_surface_preview(self):
        """按建模页同款逻辑，把面形数据同步到左侧二维预览（左=导入，右=处理后）。"""
        raw = self._dwell_state.get("surface_raw")
        processed = self._dwell_state.get("surface")
        if raw is None or processed is None:
            return
        try:
            self._main.preview.plot_dwell_surface(raw, processed)
        except Exception as exc:
            self._dwell_report(f"二维预览失败：{exc}", self.dw_surface_info, error=True)

    def _surface_operation(self, name, operation):
        try:
            data = operation(self._require_surface())
            self._dwell_state["surface"] = data
            stat = DwellTimeEngine.stats(data)
            self._dwell_report(f"{name}完成：有效点 {stat['count']}，PV={stat['pv']:.6g}，RMS={stat['rms']:.6g}。", self.dw_surface_info)
            self._dwell_refresh_surface_preview()
        except Exception as exc: self._dwell_error(name + "失败", exc, self.dw_surface_info)

    def _dwell_trim_surface(self):
        self._surface_operation("裁边", lambda data: DwellTimeEngine.trim_surface(
            data, self._dwell_float(self.dw_surface_trim, "裁边尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))

    def _dwell_mean_surface(self):
        self._surface_operation("均值滤波", lambda data: DwellTimeEngine.mean_filter(
            data, self._dwell_float(self.dw_surface_filter, "滤波窗尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))

    def _dwell_rms_surface(self):
        self._surface_operation("RMS滤波", lambda data: DwellTimeEngine.rms_filter(
            data, self._dwell_float(self.dw_surface_rms, "RMS倍数", 1.0)))

    def _dwell_extend_surface(self):
        self._surface_operation("延拓/填充", lambda data: DwellTimeEngine.extend_surface(
            data, self._dwell_float(self.dw_surface_extend, "延拓尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))

    def _build_dwell_spot_page(self):
        scroll, layout = self._dwell_scroll_page(
            "抛光斑", "读取采斑数据，按采斑时长、偏置和计算网格生成单位时间去除函数。")

        # ── 数据导入 ──
        import_group = QGroupBox("数据导入"); import_form = QVBoxLayout(import_group)
        self._dwell_add_fields(import_form, [
            ("dw_spot_file", "文件路径：", ""),
        ])
        select = QPushButton("选择抛光斑文件"); select.clicked.connect(self._dwell_select_spot)
        import_form.addWidget(select)
        layout.addWidget(import_group)

        # ── 采样参数 ──
        sample_group = QGroupBox("采样参数"); sample_form = QVBoxLayout(sample_group)
        self._dwell_add_fields(sample_form, [
            ("dw_spot_duration", "采斑时长 (s)：", "1"),
            ("dw_spot_xoffset", "X 偏置 (mm)：", "0"),
            ("dw_spot_yoffset", "Y 偏置 (mm)：", "0"),
        ])
        layout.addWidget(sample_group)

        # ── 计算网格（只读，来自初始设置）──
        grid_group = QGroupBox("计算网格"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_spot_dx", "X 间距 (mm)：", "1"), ("dw_spot_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_spot_dx", "dw_spot_dy"))
        layout.addWidget(grid_group)

        calculate = QPushButton("计算去除函数"); calculate.clicked.connect(self._dwell_calculate_spot)
        layout.addWidget(calculate)
        self.dw_spot_info = QLabel(""); self.dw_spot_info.setWordWrap(True)
        layout.addWidget(self.dw_spot_info); layout.addStretch()
        return scroll

    def _dwell_select_spot(self):
        path, _ = QFileDialog.getOpenFileName(
            self._main, "选择抛光斑数据", self.dw_spot_file.text().strip(),
            "数据文件 (*.dat *.npy *.csv *.txt);;所有文件 (*.*)")
        if not path: return
        try:
            raw, pixel = DwellTimeEngine.read_grid(path)
            self.dw_spot_file.setText(path)
            self._dwell_state.update(spot_raw=raw, spot_pixel=pixel, spot=None)
            self._dwell_report(f"抛光斑原始数据已读取：{raw.shape[1]}×{raw.shape[0]}。", self.dw_spot_info)
        except Exception as exc: self._dwell_error("抛光斑读取失败", exc, self.dw_spot_info)

    def _dwell_calculate_spot(self):
        try:
            raw = self._dwell_state.get("spot_raw")
            if raw is None: raise ValueError("请先选择抛光斑数据文件")
            spot = DwellTimeEngine.build_spot(
                raw, self._dwell_state.get("spot_pixel"),
                self._dwell_float(self.dw_spot_dx, "X间距", 0.0, False),
                self._dwell_float(self.dw_spot_dy, "Y间距", 0.0, False),
                self._dwell_float(self.dw_spot_duration, "采斑时长", 0.0, False),
                self._dwell_float(self.dw_spot_xoffset, "X偏置"),
                self._dwell_float(self.dw_spot_yoffset, "Y偏置"))
            self._dwell_state["spot"] = spot
            self._dwell_report(
                f"去除函数计算完成：{spot['kernel'].shape[1]}×{spot['kernel'].shape[0]}，"
                f"脉冲={spot['impulse']:.6g}，体去除效率={spot['volume_efficiency']:.6g} mm³/min。",
                self.dw_spot_info)
        except Exception as exc: self._dwell_error("去除函数计算失败", exc, self.dw_spot_info)

    def _build_dwell_solve_page(self):
        scroll, layout = self._dwell_scroll_page(
            "驻留时间求解", "仅保留带上下限约束的最小二乘法；脉冲迭代法和均抛求解路径均已删除。")

        # ── 求解方法 ──
        method_group = QGroupBox("求解方法"); method_layout = QVBoxLayout(method_group)
        self.dw_solver_method = QLineEdit("最小二乘法"); self.dw_solver_method.setReadOnly(True)
        self.dw_solver_method.setStyleSheet("background:#e8f0fa; color:#10243f;")
        row = QHBoxLayout(); row.addWidget(QLabel("方法：")); row.addWidget(self.dw_solver_method)
        method_layout.addLayout(row); layout.addWidget(method_group)

        # ── 驻留约束 ──
        constraint_group = QGroupBox("驻留约束"); constraint_form = QVBoxLayout(constraint_group)
        self._dwell_add_fields(constraint_form, [
            ("dw_uniform", "均抛厚度：", "0.5"),
            ("dw_max_dwell", "最大驻留时间 (s)：", "20"),
            ("dw_min_dwell", "最小驻留时间 (s)：", "0.02"),
        ])
        layout.addWidget(constraint_group)

        # ── 区域裁边 ──
        trim_group = QGroupBox("区域裁边"); trim_form = QVBoxLayout(trim_group)
        self._dwell_add_fields(trim_form, [
            ("dw_traj_margin", "轨迹区裁边 (mm)：", "0"),
            ("dw_eval_trim", "评价区裁边 (mm)：", "0"),
        ])
        layout.addWidget(trim_group)

        # ── 迭代与网格（X/Y 间距只读，来自初始设置）──
        iter_group = QGroupBox("迭代与网格"); iter_form = QVBoxLayout(iter_group)
        self._dwell_add_fields(iter_form, [
            ("dw_solve_iterations", "最大迭代次数：", "300"),
            ("dw_solve_tolerance", "收敛阈值：", "1e-6"),
            ("dw_solve_dx", "X 间距 (mm)：", "1"), ("dw_solve_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_solve_dx", "dw_solve_dy"))
        layout.addWidget(iter_group)

        # ── 边缘修饰 ──
        modify_group = QGroupBox("边缘修饰"); modify_form = QVBoxLayout(modify_group)
        self._dwell_add_fields(modify_form, [
            ("dw_modify_width", "边缘修饰宽度 (mm)：", "0"),
            ("dw_edge_adjust", "边缘驻留调整量：", "0"),
        ])
        layout.addWidget(modify_group)

        row = QHBoxLayout()
        solve = QPushButton("最小二乘法求解驻留时间")
        modify = QPushButton("驻留时间修饰")
        row.addWidget(solve); row.addWidget(modify); layout.addLayout(row)
        self.dw_solve_info = QLabel(""); self.dw_solve_info.setWordWrap(True)
        layout.addWidget(self.dw_solve_info); layout.addStretch()
        solve.clicked.connect(self._do_dwell_least_squares)
        modify.clicked.connect(self._do_dwell_modify)
        return scroll

    def _do_dwell_least_squares(self):
        try:
            surface = self._dwell_state.get("surface"); spot = self._dwell_state.get("spot")
            if surface is None: raise ValueError("请先完成面形数据导入与处理")
            if spot is None: raise ValueError("请先计算抛光斑去除函数")
            dx = self._dwell_float(self.dw_solve_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_solve_dy, "Y间距", 0.0, False)
            solution = DwellTimeEngine.solve_bounded_least_squares(
                surface, spot["kernel"], dx, dy,
                uniform=self._dwell_float(self.dw_uniform, "均抛厚度", 0.0),
                min_dwell=self._dwell_float(self.dw_min_dwell, "最小驻留时间", 0.0),
                max_dwell=self._dwell_float(self.dw_max_dwell, "最大驻留时间", 0.0),
                trajectory_trim=self._dwell_float(self.dw_traj_margin, "轨迹区裁边", 0.0),
                evaluation_trim=self._dwell_float(self.dw_eval_trim, "评价区裁边", 0.0),
                iterations=int(self._dwell_float(self.dw_solve_iterations, "最大迭代次数", 1.0)),
                tolerance=self._dwell_float(self.dw_solve_tolerance, "收敛阈值", 0.0, False))
            self._dwell_state["solution"] = solution
            self._dwell_report(
                f"最小二乘求解完成：RMS {solution['before_rms']:.6g} → {solution['after_rms']:.6g}，"
                f"记录 {len(solution['objective'])} 个收敛检查点。", self.dw_solve_info)
        except Exception as exc: self._dwell_error("驻留时间求解失败", exc, self.dw_solve_info)

    def _do_dwell_modify(self):
        try:
            solution = self._dwell_state.get("solution")
            if solution is None: raise ValueError("请先完成最小二乘法求解")
            width = self._dwell_float(self.dw_modify_width, "边缘修饰宽度", 0.0)
            adjustment = self._dwell_float(self.dw_edge_adjust, "边缘驻留调整量")
            if width <= 0 or adjustment == 0:
                self._dwell_report("修饰参数为零，驻留时间保持不变。", self.dw_solve_info); return
            dx = self._dwell_float(self.dw_solve_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_solve_dy, "Y间距", 0.0, False)
            mask = solution["trajectory_mask"]
            inner = DwellTimeEngine.erode_mask(mask, width, dx, dy)
            edge = mask & ~inner
            dwell = solution["dwell"].copy()
            dwell[edge] += adjustment * dx * dy
            minimum = self._dwell_float(self.dw_min_dwell, "最小驻留时间", 0.0)
            maximum = self._dwell_float(self.dw_max_dwell, "最大驻留时间", 0.0)
            dwell[mask] = np.clip(dwell[mask], minimum, maximum)
            solution["dwell"] = dwell
            self._dwell_report(f"边缘驻留时间修饰完成：调整 {np.count_nonzero(edge)} 个节点。", self.dw_solve_info)
        except Exception as exc: self._dwell_error("驻留时间修饰失败", exc, self.dw_solve_info)

    def _build_dwell_cnc_page(self):
        scroll, layout = self._dwell_scroll_page(
            "CNC程序生成", "依据驻留时间生成栅线轨迹、工具姿态、受限进给速度和六列 PATH 文件。")

        # ── 进给速度 ──
        speed_group = QGroupBox("进给速度"); speed_form = QVBoxLayout(speed_group)
        self._dwell_add_fields(speed_form, [
            ("dw_cnc_max_speed", "最高进给速度：", "50"),
            ("dw_cnc_min_speed", "最低进给速度：", "0.01"),
        ])
        layout.addWidget(speed_group)

        # ── 工具参数 ──
        tool_group = QGroupBox("工具参数"); tool_form = QVBoxLayout(tool_group)
        self._dwell_add_fields(tool_form, [
            ("dw_cnc_tool_radius", "工具球半径 (mm)：", "0"),
            ("dw_cnc_cylinder", "气缸压入量 (mm)：", "8"),
        ])
        layout.addWidget(tool_group)

        # ── 偏置与姿态 ──
        offset_group = QGroupBox("偏置与姿态"); offset_form = QVBoxLayout(offset_group)
        self._dwell_add_fields(offset_form, [
            ("dw_cnc_xoffset", "X 偏置 (mm)：", "0"), ("dw_cnc_yoffset", "Y 偏置 (mm)：", "0"),
            ("dw_cnc_zoffset", "Z 偏置 (mm)：", "0"), ("dw_cnc_rx", "绕 X 角 (°)：", "0"),
            ("dw_cnc_ry", "绕 Y 角 (°)：", "0"),
        ])
        layout.addWidget(offset_group)

        # ── 轨迹间距 ──
        spacing_group = QGroupBox("轨迹间距"); spacing_form = QVBoxLayout(spacing_group)
        self._dwell_add_fields(spacing_form, [
            ("dw_cnc_dx", "轨迹间距 X (mm)：", "1"),
            ("dw_cnc_dy", "轨迹间距 Y (mm)：", "1"),
        ])
        self.dw_cnc_line = QComboBox(); self.dw_cnc_line.addItems(["X", "Y"])
        combox_input(spacing_form, "线方向：", self.dw_cnc_line)
        self.dw_cnc_start = QComboBox(); self.dw_cnc_start.addItems(["左下", "右下", "右上", "左上"])
        combox_input(spacing_form, "起始点位：", self.dw_cnc_start)
        layout.addWidget(spacing_group)

        row = QHBoxLayout()
        traj = QPushButton("表面轨迹生成"); cnc = QPushButton("CNC 程序生成"); save = QPushButton("保存 CNC 代码")
        for button in (traj, cnc, save): row.addWidget(button)
        layout.addLayout(row)
        self.dw_cnc_info = QLabel(""); self.dw_cnc_info.setWordWrap(True)
        layout.addWidget(self.dw_cnc_info); layout.addStretch()
        traj.clicked.connect(self._dwell_generate_trajectory)
        cnc.clicked.connect(self._dwell_generate_cnc)
        save.clicked.connect(self._dwell_save_cnc)
        return scroll

    def _dwell_generate_trajectory(self):
        try:
            solution = self._dwell_state.get("solution")
            if solution is None: raise ValueError("请先完成驻留时间求解")
            step_x = self._dwell_float(self.dw_cnc_dx, "轨迹间距X", 0.0, False)
            step_y = self._dwell_float(self.dw_cnc_dy, "轨迹间距Y", 0.0, False)
            samples = DwellTimeEngine.raster_from_dwell(
                solution, step_x, step_y, self.dw_cnc_line.currentText(), self.dw_cnc_start.currentText())
            model = self._dwell_state.get("model")
            if model is None:
                xyz = np.column_stack((samples[:, :2], np.zeros(len(samples))))
                normals = np.tile((0.0, 0.0, 1.0), (len(samples), 1))
            else:
                xyz, normals = DwellTimeEngine.sample_nearest(model, samples[:, 0], samples[:, 1])
            valid = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1)
            samples = samples[valid]; xyz = xyz[valid]; normals = normals[valid]
            if len(samples) < 2: raise ValueError("模型口径内的有效轨迹点不足")
            trajectory = {"samples": samples, "xyz": xyz, "normals": normals,
                          "step_x": step_x, "step_y": step_y}
            self._dwell_state["trajectory"] = trajectory
            points = np.column_stack((xyz, normals)).tolist()
            self._main.preview.plot_surface(points, {
                "surface_name": "驻留时间轨迹", "traj_name": "栅形轨迹",
                "traj_type": "G", "direction": self.dw_cnc_line.currentText(), "geom": None})
            self._dwell_report(f"表面轨迹生成完成：{len(samples)} 个点。", self.dw_cnc_info)
        except Exception as exc: self._dwell_error("轨迹生成失败", exc, self.dw_cnc_info)

    def _dwell_generate_cnc(self):
        try:
            trajectory = self._dwell_state.get("trajectory")
            if trajectory is None: raise ValueError("请先生成表面轨迹")
            config = {
                "max_speed": self._dwell_float(self.dw_cnc_max_speed, "最高进给速度", 0.0, False),
                "min_speed": self._dwell_float(self.dw_cnc_min_speed, "最低进给速度", 0.0, False),
                "tool_radius": self._dwell_float(self.dw_cnc_tool_radius, "工具球半径", 0.0),
                "depth": self._dwell_float(self.dw_cnc_cylinder, "气缸压入量", 0.0),
                "x_offset": self._dwell_float(self.dw_cnc_xoffset, "X偏置"),
                "y_offset": self._dwell_float(self.dw_cnc_yoffset, "Y偏置"),
                "z_offset": self._dwell_float(self.dw_cnc_zoffset, "Z偏置"),
                "alpha": self._dwell_float(self.dw_cnc_rx, "绕X角"),
                "beta": self._dwell_float(self.dw_cnc_ry, "绕Y角"),
                "step_x": trajectory["step_x"], "step_y": trajectory["step_y"],
            }
            if config["min_speed"] > config["max_speed"]:
                raise ValueError("最低进给速度不能大于最高进给速度")
            cnc = DwellTimeEngine.generate_cnc(
                trajectory["samples"], trajectory["xyz"], trajectory["normals"], config,
                self._dwell_float(self.dw_solve_dx, "求解X间距", 0.0, False),
                self._dwell_float(self.dw_solve_dy, "求解Y间距", 0.0, False))
            self._dwell_state["cnc"] = cnc
            self._dwell_report(
                f"CNC 程序生成完成：{len(cnc['data'])} 行，总加工时间 {cnc['total_time']:.6g}。",
                self.dw_cnc_info)
        except Exception as exc: self._dwell_error("CNC 程序生成失败", exc, self.dw_cnc_info)

    def _dwell_save_cnc(self):
        cnc = self._dwell_state.get("cnc")
        if cnc is None:
            self._dwell_error("保存失败", ValueError("请先生成 CNC 程序"), self.dw_cnc_info); return
        path, _ = QFileDialog.getSaveFileName(self._main, "保存 CNC 代码", "dwell_time.path", "PATH (*.path);;TXT (*.txt)")
        if not path: return
        try:
            np.savetxt(path, cnc["data"], fmt="%.6f")
            self._dwell_report(f"CNC 代码已保存：{os.path.basename(path)}", self.dw_cnc_info)
        except OSError as exc: self._dwell_error("保存失败", exc, self.dw_cnc_info)
