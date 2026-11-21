"""TXT/XLSX trajectory exports; XLSX dependency is loaded only when used."""
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
