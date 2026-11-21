"""Migrated numerical service: SurfaceMixin; legacy numerical conventions retained."""
from __future__ import annotations
import math
import numpy as np
from polishing.core.sampling import _odd_axis

class SurfaceMixin:
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
