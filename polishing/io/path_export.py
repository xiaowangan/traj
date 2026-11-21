"""Migrated numerical service: MachinePathMixin; legacy numerical conventions retained."""
from __future__ import annotations
import math
import numpy as np
from polishing.core.transforms import TransformMixin

class MachinePathMixin:
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
                tool_vectors[index] = TransformMixin.rotate_vectors(
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
