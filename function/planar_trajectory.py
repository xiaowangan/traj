# -*- coding: utf-8 -*-
"""
平面轨迹统一入口（保持向后兼容的别名）
"""
from .traj_planar_raster import generate_planar_raster
from .traj_planar_spiral import generate_planar_spiral

# 向后兼容别名（旧代码可能直接调用 generate_raster / generate_spiral）
generate_raster = generate_planar_raster
generate_spiral = generate_planar_spiral

__all__ = [
    "generate_planar_raster",
    "generate_planar_spiral",
    "generate_raster",
    "generate_spiral",
    "save_trajectory_txt",
]


def save_trajectory_txt(points, filepath, traj_type="", shape_desc=""):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 平面轨迹点文件\n")
        f.write(f"# 轨迹类型：{traj_type}  形状：{shape_desc}\n")
        f.write(f"# 总点数：{len(points)}\n")
        f.write(f"# 列顺序：X  Y  Z  Nx  Ny  Nz\n")
        f.write(f"# {'─'*60}\n")
        for p in points:
            f.write(f"{p[0]:>14.6f}  {p[1]:>14.6f}  {p[2]:>6.1f}  "
                    f"{p[3]:>6.1f}  {p[4]:>6.1f}  {p[5]:>6.1f}\n")

def save_trajectory_xlsx(points, filepath, traj_type="", desc=""):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "轨迹点"
    ws.append(["X", "Y", "Z", "Nx", "Ny", "Nz"])
    ws.append([f"# 轨迹类型：{traj_type}  描述：{desc}  总点数：{len(points)}"])
    for p in points:
        ws.append([round(v, 6) for v in p])
    wb.save(filepath)