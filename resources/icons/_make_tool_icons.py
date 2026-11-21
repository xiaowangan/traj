"""生成与 workflow_*.png 同风格（深蓝描边 + 橙色强调）的工具栏图标。"""
import sys
from PIL import Image, ImageDraw
OUT = sys.argv[1]
S = 8                      # 超采样倍数
N = 64
NAVY, ORANGE, GREEN = (33, 79, 120, 255), (229, 139, 50, 255), (62, 140, 90, 255)
W = 4 * S                  # 线宽

def new():
    im = Image.new("RGBA", (N * S, N * S), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)

def P(*pts):
    return [(x * S, y * S) for x, y in pts]

def rr(d, box, r, color=NAVY, w=W, fill=None):
    x0, y0, x1, y1 = [v * S for v in box]
    d.rounded_rectangle((x0, y0, x1, y1), r * S, outline=color, width=w, fill=fill)

def line(d, pts, color=NAVY, w=W):
    d.line(P(*pts), fill=color, width=w, joint="curve")
    for x, y in P(pts[0], pts[-1]):          # 圆头
        d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=color)

def save(name, im):
    im.resize((N, N), Image.LANCZOS).save(f"{OUT}/{name}.png")

# 保存：软盘
im, d = new()
rr(d, (9, 9, 55, 55), 6)
rr(d, (19, 9, 45, 25), 2)
rr(d, (18, 37, 46, 55), 2, ORANGE)
line(d, [(24, 44), (40, 44)], ORANGE, 3 * S)
save("tool_save", im)

# 打开：文件夹 + 箭头
im, d = new()
d.line(P((8, 20), (8, 52), (46, 52), (54, 30), (22, 30), (16, 52)), fill=NAVY, width=W, joint="curve")
d.line(P((8, 20), (8, 12), (24, 12), (29, 19), (46, 19), (46, 30)), fill=NAVY, width=W, joint="curve")
line(d, [(40, 12), (54, 12)], ORANGE)
line(d, [(48, 6), (54, 12), (48, 18)], ORANGE)
save("tool_open", im)

# 导入 STL：立方体
im, d = new()
hexa = P((32, 8), (54, 20), (54, 44), (32, 56), (10, 44), (10, 20), (32, 8))
d.line(hexa, fill=NAVY, width=W, joint="curve")
line(d, [(10, 20), (32, 32), (54, 20)], ORANGE)
line(d, [(32, 32), (32, 56)], ORANGE)
save("tool_stl", im)

# 授权：证书 + 印章
im, d = new()
rr(d, (8, 8, 50, 46), 5)
line(d, [(16, 19), (42, 19)])
line(d, [(16, 29), (34, 29)])
d.ellipse(P((34, 34), (56, 56)) [0] + P((34, 34), (56, 56))[1], fill=(223, 233, 245, 255), outline=ORANGE, width=W)
line(d, [(41, 45), (44, 49), (50, 41)], GREEN, 3 * S)
save("tool_license", im)

# 退出：门 + 箭头
im, d = new()
d.line(P((30, 10), (10, 10), (10, 54), (30, 54)), fill=NAVY, width=W, joint="curve")
line(d, [(30, 10), (30, 18)]); line(d, [(30, 46), (30, 54)])
line(d, [(24, 32), (54, 32)], ORANGE)
line(d, [(44, 22), (54, 32), (44, 42)], ORANGE)
save("tool_exit", im)

# 资源与标定：三条滑杆
im, d = new()
for y, cx in ((16, 22), (32, 42), (48, 28)):
    line(d, [(8, y), (56, y)])
    d.ellipse((( cx - 6) * S, (y - 6) * S, (cx + 6) * S, (y + 6) * S), fill=(223, 233, 245, 255), outline=ORANGE, width=W)
save("tool_resource", im)

# 新建案例：文档 + 加号
im, d = new()
d.line(P((40, 8), (14, 8), (14, 56), (50, 56), (50, 18), (40, 8)), fill=NAVY, width=W, joint="curve")
line(d, [(40, 8), (40, 18), (50, 18)])
line(d, [(32, 28), (32, 46)], ORANGE)
line(d, [(23, 37), (41, 37)], ORANGE)
save("tool_new", im)

# 抛光斑/TIF：同心圆斑点 + 剖面曲线
im, d = new()
d.ellipse(P((8, 8), (56, 56))[0] + P((8, 8), (56, 56))[1], outline=NAVY, width=W)
d.ellipse(P((20, 20), (44, 44))[0] + P((20, 20), (44, 44))[1], outline=ORANGE, width=W)
d.ellipse(P((29, 29), (35, 35))[0] + P((29, 29), (35, 35))[1], fill=ORANGE)
save("tool_tif", im)

# 覆盖轨迹：蛇形栅线
im, d = new()
d.line(P((10, 12), (54, 12), (54, 25), (10, 25), (10, 38), (54, 38), (54, 51), (10, 51)),
       fill=NAVY, width=W, joint="curve")
for x, y in ((10, 12), (10, 51)):
    d.ellipse(((x - 4) * S, (y - 4) * S, (x + 4) * S, (y + 4) * S), fill=ORANGE)
save("tool_path", im)
