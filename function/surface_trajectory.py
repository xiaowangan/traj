# -*- coding: utf-8 -*-
"""
曲面轨迹统一入口
从各独立轨迹模块导入，统一对外暴露接口
输出格式：每点 [X, Y, Z, Nx, Ny, Nz]
"""
from .traj_aspherical  import generate_aspherical
from .traj_spherical   import generate_spherical
from .traj_cylindrical import generate_cylindrical
from .traj_conical     import generate_conical

__all__ = [
    "generate_aspherical",
    "generate_spherical",
    "generate_cylindrical",
    "generate_conical",
    "save_surface_trajectory_txt",
]


def save_surface_trajectory_txt(points, filepath, traj_name="", surface_name=""):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 曲面轨迹点文件\n")
        f.write(f"# 曲面类型：{surface_name}  轨迹类型：{traj_name}\n")
        f.write(f"# 总点数：{len(points)}\n")
        f.write(f"# 列顺序：X  Y  Z  Nx  Ny  Nz\n")
        f.write(f"# {'─'*60}\n")
        for p in points:
            f.write(f"{p[0]:>14.6f}  {p[1]:>14.6f}  {p[2]:>14.6f}  "
                    f"{p[3]:>10.6f}  {p[4]:>10.6f}  {p[5]:>10.6f}\n")
