# -*- coding: utf-8 -*-
"""
平面轨迹生成软件
界面完全仿照师兄的离散曲面抛光轨迹规划软件：
  - Ribbon 工具栏（复用相同 GUI/ 框架和 stylesheets/）
  - 右侧 QDockWidget + QStackedWidget 参数控制台
  - 左侧中央区域显示轨迹预览图
  - 底部状态栏
  - 背景色 #dfe9f5，字体微软雅黑
"""

import sys
import os

# ── 确保 cwd 是软件根目录（stylesheets/ icons/ 的相对路径依赖此）──
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QStackedWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QScrollArea, QPlainTextEdit, QStyleFactory,
    QAction, QToolBar
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

# ── 复用师兄的 Ribbon 框架 ──────────────────────────────────────────
from GUI.RibbonWidget import RibbonWidget
from GUI.RibbonButton import RibbonButton
from GUI.Icons import get_icon
from GUI.StyleSheets import get_stylesheet

# ── 本模块 ──────────────────────────────────────────────────────────
from function.planar_trajectory import (generate_planar_raster, generate_planar_spiral,
    generate_raster, generate_spiral, save_trajectory_txt)
from function.surface_trajectory import (
    generate_aspherical, generate_spherical,
    generate_cylindrical, generate_conical,
    save_surface_trajectory_txt
)
from function.license_manager   import get_hardware_id, activate, verify_license

plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']


# ════════════════════════════════════════════════════════════════════
# 工具函数（完全仿 Control_page.py 的写法）
# ════════════════════════════════════════════════════════════════════
def lineedit_input(label_text, default_value=""):
    label     = QLabel(label_text)
    line_edit = QLineEdit(default_value)
    layout    = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(line_edit)
    return line_edit, layout


def combox_input(layout, label_text, widget):
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    row.addWidget(widget)
    layout.addLayout(row)


def divider():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


# ════════════════════════════════════════════════════════════════════
# 预览画布（左侧中央区域）
# ════════════════════════════════════════════════════════════════════
class PreviewCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #dfe9f5;")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # ── 两个独立画布，用 QStackedWidget 切换（平面2D / 曲面双3D） ──
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        # ── 画布 0：平面轨迹（2D 单图） ──
        self._fig2d, self._ax2d = plt.subplots(figsize=(7, 6), dpi=96)
        self._fig2d.patch.set_facecolor("#dfe9f5")
        self._ax2d.set_facecolor("#f0f5fc")
        canvas2d = FigureCanvas(self._fig2d)
        canvas2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w2d = QWidget(); l2d = QVBoxLayout(w2d); l2d.setContentsMargins(0,0,0,0)
        l2d.addWidget(canvas2d)
        self._stack.addWidget(w2d)   # idx 0
        self._canvas2d = canvas2d

        # ── 画布 1：曲面轨迹（左=曲面形状，右=轨迹，并排 3D） ──
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self._fig3d = plt.figure(figsize=(14, 6), dpi=96)
        self._fig3d.patch.set_facecolor("#dfe9f5")
        # 左子图：曲面形状（密集散点云）
        self._ax3d_surf = self._fig3d.add_subplot(121, projection="3d")
        # 右子图：轨迹路径
        self._ax3d_traj = self._fig3d.add_subplot(122, projection="3d")
        self._cb3d  = None   # colorbar 句柄（绑在右图），每次重建前先删
        self._cb3d_surf = None  # colorbar 句柄（绑在左图）
        canvas3d = FigureCanvas(self._fig3d)
        canvas3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        w3d = QWidget(); l3d = QVBoxLayout(w3d); l3d.setContentsMargins(0,0,0,0)
        l3d.addWidget(canvas3d)
        self._stack.addWidget(w3d)   # idx 1
        self._canvas3d = canvas3d

        # 初始显示 2D 空白提示
        self._ax2d.set_title("（尚未生成轨迹）", fontsize=11, color="#888888")
        self._ax2d.axis("off")
        self._canvas2d.draw()
        self._stack.setCurrentIndex(0)

    # ── 平面轨迹：2D 画布 ──────────────────────────────────────────
    def plot(self, points, params):
        self._stack.setCurrentIndex(0)
        ax = self._ax2d
        ax.clear()
        ax.set_facecolor("#f0f5fc")
        ax.grid(True, color="#c8d8ee", linewidth=0.6, linestyle="--")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)

        if not points:
            ax.set_title("未生成任何轨迹点", fontsize=11, color="#c0392b")
            self._canvas2d.draw()
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        shape = params.get("shape", "R")

        if shape == "R":
            A, B = params.get("rect_A", 0), params.get("rect_B", 0)
            ax.add_patch(mpatches.Rectangle((-A/2, -B/2), A, B,
                lw=1.5, edgecolor="#1a3f6f", facecolor="none",
                linestyle="--", label="形状边界"))
        else:
            R = params.get("circle_R", 0)
            ax.add_patch(mpatches.Circle((0, 0), R,
                lw=1.5, edgecolor="#1a3f6f", facecolor="none",
                linestyle="--", label="形状边界"))

        if params.get("traj_type") == "G" and params.get("cover_type") == 2:
            x0, y0 = params.get("sub_x0", 0), params.get("sub_y0", 0)
            C,  D  = params.get("sub_C",  0), params.get("sub_D",  0)
            ax.add_patch(mpatches.Rectangle((x0, y0), C, D,
                lw=1, edgecolor="#e67e22", facecolor="none",
                linestyle=":", label="子区域"))

        ax.plot(xs, ys, "-", color="#2563b0", lw=0.8, alpha=0.75, label="轨迹路径")
        ax.scatter(xs, ys, s=7, color="#1a3f6f", zorder=3, label=f"轨迹点({len(points)})")
        ax.scatter([xs[0]],  [ys[0]],  s=60, color="#27ae60",
                   zorder=5, marker="^", label="起点")
        ax.scatter([xs[-1]], [ys[-1]], s=60, color="#c0392b",
                   zorder=5, marker="s", label="终点")

        tname = "栅形" if params.get("traj_type") == "G" else "螺旋线"
        sname = "矩形" if shape == "R" else "圆形"
        ax.set_title(f"{sname} · {tname}轨迹  共 {len(points)} 个轨迹点",
                     fontsize=11, color="#1a3f6f", pad=8)
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        self._fig2d.tight_layout()
        self._canvas2d.draw()

    # ── 曲面轨迹：左=曲面形状，右=轨迹路径（并排 3D 双视图） ───────
    def plot_surface(self, points, params):
        self._stack.setCurrentIndex(1)

        if self._cb3d is not None:
            try: self._cb3d.remove()
            except: pass
            self._cb3d = None
        if self._cb3d_surf is not None:
            try: self._cb3d_surf.remove()
            except: pass
            self._cb3d_surf = None
        self._fig3d.clear()
        self._ax3d_surf = self._fig3d.add_subplot(121, projection="3d")
        self._ax3d_traj = self._fig3d.add_subplot(122, projection="3d")

        if not points:
            self._ax3d_traj.set_title("未生成任何轨迹点", fontsize=11, color="#c0392b")
            self._canvas3d.draw()
            return

        import numpy as _np
        xs = _np.array([p[0] for p in points], dtype=float)
        ys = _np.array([p[1] for p in points], dtype=float)
        zs = _np.array([p[2] for p in points], dtype=float)

        surface_name = params.get("surface_name", "曲面")
        traj_name    = params.get("traj_name", "")

        def _set_aspect(ax, xs_, ys_, zs_):
            x_rng = float(xs_.max() - xs_.min())
            y_rng = float(ys_.max() - ys_.min())
            z_rng = float(zs_.max() - zs_.min())
            eps = max(x_rng, y_rng, z_rng) * 1e-3 + 1e-6
            ax.set_box_aspect((max(x_rng, eps), max(y_rng, eps), max(z_rng, eps)))

        def _break_jumps(xs_, ys_, zs_):
            if len(xs_) < 3:
                return xs_.copy(), ys_.copy(), zs_.copy()
            seg = _np.sqrt(_np.diff(xs_)**2 + _np.diff(ys_)**2 + _np.diff(zs_)**2)
            med = _np.median(seg[seg > 0]) if _np.any(seg > 0) else 0.0
            xp, yp, zp = xs_.astype(float), ys_.astype(float), zs_.astype(float)
            if med > 0:
                for i in _np.where(seg > 1.5 * med)[0][::-1]:
                    xp = _np.insert(xp, i+1, _np.nan)
                    yp = _np.insert(yp, i+1, _np.nan)
                    zp = _np.insert(zp, i+1, _np.nan)
            return xp, yp, zp

        # ── 左图：曲面形状（密集散点云）─────────────────────────────
        ax_s = self._ax3d_surf
        ax_s.set_facecolor("#f0f5fc")
        idx_s = _np.random.choice(len(points), min(10000, len(points)), replace=False)
        idx_s.sort()
        xs_s = xs[idx_s];
        ys_s = ys[idx_s];
        zs_s = zs[idx_s]
        sc_s = ax_s.scatter(xs_s, ys_s, zs_s, c=zs_s, cmap="jet",
                            s=6, zorder=3, depthshade=True, alpha=0.85)
        self._cb3d_surf = self._fig3d.colorbar(sc_s, ax=ax_s, shrink=0.5, pad=0.12)
        self._cb3d_surf.set_label("Z (mm)", fontsize=8)
        ax_s.set_title(f"{surface_name} 形状", fontsize=10, color="#1a3f6f", pad=8)
        ax_s.set_xlabel("X (mm)", fontsize=8, labelpad=4)
        ax_s.set_ylabel("Y (mm)", fontsize=8, labelpad=4)
        ax_s.set_zlabel("Z (mm)", fontsize=8, labelpad=4)
        _set_aspect(ax_s, xs, ys, zs)

        # ── 右图：轨迹路径（密集散点 + 起终点，不画连线避免菱形混乱）──
        ax_t = self._ax3d_traj
        ax_t.set_facecolor("#f0f5fc")
        idx_t = _np.random.choice(len(points), min(10000, len(points)), replace=False)
        idx_t.sort()
        xs_t = xs[idx_t];
        ys_t = ys[idx_t];
        zs_t = zs[idx_t]
        sc_t = ax_t.scatter(xs_t, ys_t, zs_t, c=zs_t, cmap="jet",
                            s=6, zorder=3, depthshade=True, alpha=0.85)
        ax_t.scatter([xs[0]], [ys[0]], [zs[0]],
                     s=60, color="#27ae60", zorder=5, marker="^", label="起点")
        ax_t.scatter([xs[-1]], [ys[-1]], [zs[-1]],
                     s=60, color="#c0392b", zorder=5, marker="s", label="终点")
        self._cb3d = self._fig3d.colorbar(sc_t, ax=ax_t, shrink=0.5, pad=0.12)
        self._cb3d.set_label("Z (mm)  ←蓝低  红高→", fontsize=8)
        ax_t.set_title(f"{traj_name}  共 {len(points)} 个轨迹点",
                       fontsize=10, color="#1a3f6f", pad=8)
        ax_t.set_xlabel("X (mm)", fontsize=8, labelpad=4)
        ax_t.set_ylabel("Y (mm)", fontsize=8, labelpad=4)
        ax_t.set_zlabel("Z (mm)", fontsize=8, labelpad=4)
        ax_t.legend(loc="upper left", fontsize=8, framealpha=0.85)
        _set_aspect(ax_t, xs, ys, zs)

        self._fig3d.tight_layout()
        self._canvas3d.draw()
# ════════════════════════════════════════════════════════════════════
class ControlPanel(QStackedWidget):
    """
    仿照师兄软件：QDockWidget 里放 QStackedWidget，
    每个功能对应一个 page，Ribbon 按钮切换页面。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main = parent          # MainWindow 引用，用于访问 preview/statusbar

        # 记录各页面索引
        self.idx_blank   = self.count()
        self.addWidget(QWidget())    # page 0：空白（初始状态）

        self.idx_license = self.count()
        self.addWidget(self._build_license_page())

        # 曲面轨迹：统一入口页（下拉选择 + 子页面）
        self.idx_surface = self.count()
        self.addWidget(self._build_surface_selector_page())

        # 当前缓存的轨迹点和参数
        self._points = []
        self._params = {}

    # ── 共用：保存 TXT ──────────────────────────────────────────────
    def _do_save(self, traj_name, fname_hint, is_surface=False):
        if not self._points:
            QMessageBox.warning(self._main, "提示", "请先生成轨迹")
            return
        default = (fname_hint.strip() or "trajectory") + ".txt"
        path, _ = QFileDialog.getSaveFileName(
            self._main, "保存轨迹文件", default, "文本文件 (*.txt)")
        if not path:
            return
        try:
            if is_surface:
                surface_name = self._params.get("surface_name", "")
                save_surface_trajectory_txt(self._points, path, traj_name, surface_name)
            else:
                shape_name = "矩形" if self._params.get("shape") == "R" else "圆形"
                save_trajectory_txt(self._points, path, traj_name, shape_name)
            QMessageBox.information(self._main, "保存成功",
                f"轨迹文件已保存：\n{path}\n共 {len(self._points)} 个点")
            self._main.statusbar.showMessage(f"已保存至 {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self._main, "保存失败", str(e))

    def _finish(self, points, params, save_btn, info_lbl, tname, is_surface=False):
        self._points = points
        self._params = params
        if is_surface:
            self._main.preview.plot_surface(points, params)
        else:
            self._main.preview.plot(points, params)
        save_btn.setEnabled(True)
        if is_surface:
            info_lbl.setText(
                f"✔ 生成完成 | {tname} | {len(points)} 个轨迹点\n"
                f"  输出含 X Y Z Nx Ny Nz")
        else:
            sname = "矩形" if params.get("shape") == "R" else "圆形"
            info_lbl.setText(
                f"✔ 生成完成 | {sname}{tname} | {len(points)} 个轨迹点\n"
                f"  Z=0，法向量 (0,0,1)")
        info_lbl.setStyleSheet("color:#1a7a3c; font-size:11px;")
        self._main.statusbar.showMessage(
            f"{tname}生成完成，共 {len(points)} 个轨迹点")
        # 输出到结果终端
        xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
        self._main.terminal_output.appendPlainText(
            f"[轨迹生成] {tname}，共 {len(points)} 个点\n"
            f"  X∈[{min(xs):.3f}, {max(xs):.3f}]  "
            f"Y∈[{min(ys):.3f}, {max(ys):.3f}]  "
            f"Z∈[{min(zs):.3f}, {max(zs):.3f}]")

    # ────────────────────────────────────────────────────────────────
    # 授权管理页面
    # ────────────────────────────────────────────────────────────────
    def _build_license_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("软件授权管理")
        g   = QVBoxLayout(grp)

        # 机器码
        hwid_row = QHBoxLayout()
        hwid_row.addWidget(QLabel("本机机器码："))
        self.lic_edt_hwid = QLineEdit(get_hardware_id())
        self.lic_edt_hwid.setReadOnly(True)
        self.lic_edt_hwid.setStyleSheet(
            "font-family:Consolas; color:#1a3f6f; background:#e8f0fa;")
        hwid_row.addWidget(self.lic_edt_hwid)
        btn_copy = QPushButton("复制")
        btn_copy.setFixedWidth(46)
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.lic_edt_hwid.text()))
        hwid_row.addWidget(btn_copy)
        g.addLayout(hwid_row)
        g.addWidget(QLabel("  将机器码发送给供应商以获取激活码"))

        self.lic_edt_days, row_d = lineedit_input("授权天数：", "365")
        g.addLayout(row_d)

        self.lic_edt_code, row_c = lineedit_input("激活码：")
        self.lic_edt_code.setPlaceholderText("XXXXXXXX-XXXXXXXX-XXXXXXXX-XXXXXXXX")
        self.lic_edt_code.setStyleSheet("font-family:Consolas;")
        g.addLayout(row_c)

        self.lic_btn_act = QPushButton("立即激活")
        g.addWidget(self.lic_btn_act)
        g.addWidget(divider())

        self.lic_lbl_status = QLabel("（点击'查询状态'刷新）")
        self.lic_lbl_status.setWordWrap(True)
        g.addWidget(self.lic_lbl_status)
        btn_q = QPushButton("查询当前授权状态")
        g.addWidget(btn_q)

        layout.addWidget(grp)
        layout.addStretch()

        self.lic_btn_act.clicked.connect(self._do_activate)
        btn_q.clicked.connect(self._do_query)
        return page

    def _do_activate(self):
        code = self.lic_edt_code.text().strip()
        if not code:
            QMessageBox.warning(self._main, "提示", "请输入激活码"); return
        try:
            days = int(self.lic_edt_days.text())
            assert days > 0
        except:
            QMessageBox.warning(self._main, "提示", "授权天数必须为正整数"); return
        ok, msg = activate(code, days)
        color = "#1a7a3c" if ok else "#c0392b"
        mark  = "✔" if ok else "✘"
        self.lic_lbl_status.setText(f"{mark} {msg}")
        self.lic_lbl_status.setStyleSheet(f"color:{color}; font-size:12px;")
        if ok:
            QMessageBox.information(self._main, "激活成功", msg)
        else:
            QMessageBox.warning(self._main, "激活失败", msg)

    def _do_query(self):
        ok, msg = verify_license()
        color = "#1a7a3c" if ok else "#c0392b"
        mark  = "✔" if ok else "✘"
        self.lic_lbl_status.setText(f"{mark} {msg}")
        self.lic_lbl_status.setStyleSheet(f"color:{color}; font-size:12px;")

    # ────────────────────────────────────────────────────────────────
    # 共用：轨迹类型子组件（栅形/螺旋线）
    # ────────────────────────────────────────────────────────────────
    def _build_traj_group(self, prefix):
        """返回 (grp, cmb_type, cmb_dir, edt_step, edt_spacing, edt_pitch, edt_arc)
        无论栅形还是螺旋线，均只显示「间距」和「步长」两个输入框。
        默认值：间距=0.8 mm，步长=0.25 mm（固定）。
        """
        grp = QGroupBox("轨迹参数")
        g   = QVBoxLayout(grp)

        cmb_type = QComboBox()
        cmb_type.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g, "轨迹类型：", cmb_type)

        # 隐藏的栅形方向（保留供读取逻辑使用，不再显示）
        cmb_dir = QComboBox()
        cmb_dir.addItems(["平行于 X 轴（沿 Y 推进）", "平行于 Y 轴（沿 X 推进）"])
        cmb_dir.setVisible(False)

        # 统一显示：间距（上）、步长（下）
        edt_spacing, row_sp  = lineedit_input("间距 (mm)：",  "0.8")
        edt_step,    row_st  = lineedit_input("步长 (mm)：",  "0.25")
        g.addLayout(row_sp)
        g.addLayout(row_st)

        # 保留 pitch/arc 对象供 _read_traj 使用，不显示
        edt_pitch = QLineEdit("0.8");   edt_pitch.setVisible(False)
        edt_arc   = QLineEdit("0.25");  edt_arc.setVisible(False)

        def on_spacing_changed(txt):
            edt_pitch.setText(txt)
        def on_step_changed(txt):
            edt_arc.setText(txt)

        edt_spacing.textChanged.connect(on_spacing_changed)
        edt_step.textChanged.connect(on_step_changed)

        return grp, cmb_type, cmb_dir, edt_step, edt_spacing, edt_pitch, edt_arc

    def _read_traj(self, cmb_type, cmb_dir, edt_step, edt_spacing, edt_pitch, edt_arc):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        traj_type = "G" if cmb_type.currentIndex() == 0 else "S"
        direction = "X" if cmb_dir.currentIndex() == 0 else "Y"
        return dict(
            traj_type=traj_type, direction=direction,
            step_len=f(edt_step, "点间步长"),
            line_spacing=f(edt_spacing, "线间距"),
            pitch=f(edt_pitch, "螺距"),
            arc_step=f(edt_arc, "弧长步长"),
        )

    # ────────────────────────────────────────────────────────────────
    # 非球面轨迹页面
    # ────────────────────────────────────────────────────────────────
    def _build_aspherical_page(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        w = QWidget(); scroll.setWidget(w)
        layout = QVBoxLayout(w); layout.setSpacing(6); layout.setContentsMargins(6,6,6,6)

        # ① 非球面参数
        grp1 = QGroupBox("非球面参数")
        g1   = QVBoxLayout(grp1)
        self.asp_R,       r1  = lineedit_input("曲率半径 R (mm, +凸/-凹)：")
        self.asp_k,       r2  = lineedit_input("圆锥常数 k：",         "0")
        self.asp_A4,      r3  = lineedit_input("A4：",                  "0")
        self.asp_A6,      r4  = lineedit_input("A6：",                  "0")
        self.asp_A8,      r5  = lineedit_input("A8：",                  "0")
        self.asp_A10,     r6  = lineedit_input("A10：",                 "0")
        self.asp_A12,     r7  = lineedit_input("A12：",                 "0")
        self.asp_A14,     r8  = lineedit_input("A14：",                 "0")
        self.asp_off,     r9  = lineedit_input("离轴量 offcenter (mm)：", "0")
        for r in [r1,r2,r3,r4,r5,r6,r7,r8,r9]: g1.addLayout(r)
        layout.addWidget(grp1)

        # ② 边界
        grp2 = QGroupBox("轨迹边界")
        g2   = QVBoxLayout(grp2)
        self.asp_cmb_bound = QComboBox()
        self.asp_cmb_bound.addItems(["全口径矩形", "局部矩形", "局部圆形"])
        combox_input(g2, "边界类型：", self.asp_cmb_bound)

        self.asp_W,  rW  = lineedit_input("X方向宽度 (mm)：")
        self.asp_L,  rL  = lineedit_input("Y方向长度 (mm)：")
        self.asp_x1, rx1 = lineedit_input("矩形 X_min (mm)：", "0")
        self.asp_x2, rx2 = lineedit_input("矩形 X_max (mm)：", "0")
        self.asp_y1, ry1 = lineedit_input("矩形 Y_min (mm)：", "0")
        self.asp_y2, ry2 = lineedit_input("矩形 Y_max (mm)：", "0")
        self.asp_cR, rcR = lineedit_input("圆形半径 (mm)：",   "0")
        self.asp_cx, rcx = lineedit_input("圆心 X (mm)：",     "0")
        self.asp_cy, rcy = lineedit_input("圆心 Y (mm)：",     "0")
        for r in [rW,rL,rx1,rx2,ry1,ry2,rcR,rcx,rcy]: g2.addLayout(r)

        def _asp_bound_changed(idx):
            self.asp_W.setVisible(idx==0); self.asp_L.setVisible(idx==0)
            for w in [self.asp_x1,self.asp_x2,self.asp_y1,self.asp_y2]:
                w.setVisible(idx==1)
            for w in [self.asp_cR,self.asp_cx,self.asp_cy]:
                w.setVisible(idx==2)
        self.asp_cmb_bound.currentIndexChanged.connect(_asp_bound_changed)
        _asp_bound_changed(0)
        layout.addWidget(grp2)

        # ③ 轨迹参数
        grp3, self.asp_t, self.asp_dir, self.asp_st, self.asp_sp, self.asp_pt, self.asp_arc = \
            self._build_traj_group("asp")
        layout.addWidget(grp3)

        # ④ 输出
        grp4 = QGroupBox("输出设置")
        g4   = QVBoxLayout(grp4)
        self.asp_fname, rf = lineedit_input("文件名：", "aspherical_traj")
        g4.addLayout(rf); layout.addWidget(grp4)

        btn_row = QHBoxLayout()
        self.asp_btn_gen  = QPushButton("生成轨迹")
        self.asp_btn_save = QPushButton("保存 TXT"); self.asp_btn_save.setEnabled(False)
        btn_row.addWidget(self.asp_btn_gen); btn_row.addWidget(self.asp_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.asp_info = QLabel(""); self.asp_info.setWordWrap(True)
        layout.addWidget(self.asp_info); layout.addStretch()

        self.asp_btn_gen.clicked.connect(self._do_generate_aspherical)
        self.asp_btn_save.clicked.connect(
            lambda: self._do_save("非球面轨迹", self.asp_fname.text(), is_surface=True))
        return scroll

    def _do_generate_aspherical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            tp = self._read_traj(self.asp_t, self.asp_dir, self.asp_st,
                                  self.asp_sp, self.asp_pt, self.asp_arc)
            bi = self.asp_cmb_bound.currentIndex()
            p = dict(
                R=f(self.asp_R,"曲率半径R"), k=f(self.asp_k,"k"),
                A4=f(self.asp_A4,"A4"), A6=f(self.asp_A6,"A6"),
                A8=f(self.asp_A8,"A8"), A10=f(self.asp_A10,"A10"),
                A12=f(self.asp_A12,"A12"), A14=f(self.asp_A14,"A14"),
                offcenter=f(self.asp_off,"离轴量"),
                bound_type=bi+1,
                full_width=f(self.asp_W,"X宽度"), full_length=f(self.asp_L,"Y长度"),
                rect_xmin=f(self.asp_x1,"X_min"), rect_xmax=f(self.asp_x2,"X_max"),
                rect_ymin=f(self.asp_y1,"Y_min"), rect_ymax=f(self.asp_y2,"Y_max"),
                circ_R=f(self.asp_cR,"圆形半径"),
                circ_xc=f(self.asp_cx,"圆心X"), circ_yc=f(self.asp_cy,"圆心Y"),
                **tp)
            pts = generate_aspherical(**p)
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        meta = {"surface_name":"非球面", "traj_name":("栅形" if tp["traj_type"]=="G" else "螺旋线")+"轨迹"}
        self._finish(pts, meta, self.asp_btn_save, self.asp_info, "非球面轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 球面轨迹页面
    # ────────────────────────────────────────────────────────────────
    def _build_spherical_page(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        w = QWidget(); scroll.setWidget(w)
        layout = QVBoxLayout(w); layout.setSpacing(6); layout.setContentsMargins(6,6,6,6)

        grp1 = QGroupBox("球面参数")
        g1   = QVBoxLayout(grp1)
        self.sph_R,   r1 = lineedit_input("球体半径 R (mm)：")
        self.sph_zc,  r2 = lineedit_input("球心 Z 坐标 zc：",  "0")
        self.sph_h,   r3 = lineedit_input("球冠高度 h (mm)：")
        self.sph_cmb_surf = QComboBox()
        self.sph_cmb_surf.addItems(["凸球 (Convex)", "凹球 (Concave)"])
        for r in [r1,r2,r3]: g1.addLayout(r)
        combox_input(g1, "表面类型：", self.sph_cmb_surf)
        layout.addWidget(grp1)

        grp2, self.sph_t, self.sph_dir, self.sph_st, self.sph_sp, self.sph_pt, self.sph_arc = \
            self._build_traj_group("sph")
        layout.addWidget(grp2)

        grp3 = QGroupBox("输出设置")
        g3   = QVBoxLayout(grp3)
        self.sph_fname, rf = lineedit_input("文件名：", "spherical_traj")
        g3.addLayout(rf); layout.addWidget(grp3)

        btn_row = QHBoxLayout()
        self.sph_btn_gen  = QPushButton("生成轨迹")
        self.sph_btn_save = QPushButton("保存 TXT"); self.sph_btn_save.setEnabled(False)
        btn_row.addWidget(self.sph_btn_gen); btn_row.addWidget(self.sph_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.sph_info = QLabel(""); self.sph_info.setWordWrap(True)
        layout.addWidget(self.sph_info); layout.addStretch()

        self.sph_btn_gen.clicked.connect(self._do_generate_spherical)
        self.sph_btn_save.clicked.connect(
            lambda: self._do_save("球面轨迹", self.sph_fname.text(), is_surface=True))
        return scroll

    def _do_generate_spherical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            tp = self._read_traj(self.sph_t, self.sph_dir, self.sph_st,
                                  self.sph_sp, self.sph_pt, self.sph_arc)
            surf = "convex" if self.sph_cmb_surf.currentIndex() == 0 else "concave"
            pts = generate_spherical(
                R=f(self.sph_R,"球体半径R"), zc=f(self.sph_zc,"球心Z"),
                surf_type=surf, h=f(self.sph_h,"球冠高度h"), **tp)
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        meta = {"surface_name":"球面", "traj_name":("栅形" if tp["traj_type"]=="G" else "螺旋线")+"轨迹"}
        self._finish(pts, meta, self.sph_btn_save, self.sph_info, "球面轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 柱面轨迹页面
    # ────────────────────────────────────────────────────────────────
    def _build_cylindrical_page(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        w = QWidget(); scroll.setWidget(w)
        layout = QVBoxLayout(w); layout.setSpacing(6); layout.setContentsMargins(6,6,6,6)

        grp1 = QGroupBox("柱面参数")
        g1   = QVBoxLayout(grp1)
        self.cyl_cmb_axis = QComboBox()
        self.cyl_cmb_axis.addItems(["轴线沿 Y 方向", "轴线沿 X 方向"])
        combox_input(g1, "轴线方向：", self.cyl_cmb_axis)
        self.cyl_cmb_surf = QComboBox()
        self.cyl_cmb_surf.addItems(["凸柱外表面 (Convex)", "凹柱内表面 (Concave)"])
        combox_input(g1, "表面类型：", self.cyl_cmb_surf)
        self.cyl_R,    r1 = lineedit_input("圆柱半径 R (mm)：")
        self.cyl_zc,   r2 = lineedit_input("圆柱截面圆心 Z (zc)：", "0")
        self.cyl_k,    r3 = lineedit_input("切割平面高度 k：",       "0")
        self.cyl_amin, r4 = lineedit_input("轴向范围起点 (mm)：",    "-50")
        self.cyl_amax, r5 = lineedit_input("轴向范围终点 (mm)：",    "50")
        for r in [r1,r2,r3,r4,r5]: g1.addLayout(r)

        self.cyl_cmb_proj = QComboBox()
        self.cyl_cmb_proj.addItems(["矩形投影区域", "圆形投影区域"])
        combox_input(g1, "投影区域：", self.cyl_cmb_proj)
        self.cyl_pR, rp = lineedit_input("投影圆半径 (mm)：", "0")
        g1.addLayout(rp)

        def _cyl_proj_changed(idx):
            self.cyl_pR.setVisible(idx == 1)
        self.cyl_cmb_proj.currentIndexChanged.connect(_cyl_proj_changed)
        _cyl_proj_changed(0)
        layout.addWidget(grp1)

        grp2, self.cyl_t, self.cyl_dir, self.cyl_st, self.cyl_sp, self.cyl_pt, self.cyl_arc = \
            self._build_traj_group("cyl")
        layout.addWidget(grp2)

        grp3 = QGroupBox("输出设置")
        g3   = QVBoxLayout(grp3)
        self.cyl_fname, rf = lineedit_input("文件名：", "cylindrical_traj")
        g3.addLayout(rf); layout.addWidget(grp3)

        btn_row = QHBoxLayout()
        self.cyl_btn_gen  = QPushButton("生成轨迹")
        self.cyl_btn_save = QPushButton("保存 TXT"); self.cyl_btn_save.setEnabled(False)
        btn_row.addWidget(self.cyl_btn_gen); btn_row.addWidget(self.cyl_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.cyl_info = QLabel(""); self.cyl_info.setWordWrap(True)
        layout.addWidget(self.cyl_info); layout.addStretch()

        self.cyl_btn_gen.clicked.connect(self._do_generate_cylindrical)
        self.cyl_btn_save.clicked.connect(
            lambda: self._do_save("柱面轨迹", self.cyl_fname.text(), is_surface=True))
        return scroll

    def _do_generate_cylindrical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            tp = self._read_traj(self.cyl_t, self.cyl_dir, self.cyl_st,
                                  self.cyl_sp, self.cyl_pt, self.cyl_arc)
            axis_dir = "Y" if self.cyl_cmb_axis.currentIndex() == 0 else "X"
            surf_t   = "C" if self.cyl_cmb_surf.currentIndex() == 0 else "V"
            proj_s   = "R" if self.cyl_cmb_proj.currentIndex() == 0 else "C"
            pts = generate_cylindrical(
                R=f(self.cyl_R,"圆柱半径R"), zc=f(self.cyl_zc,"圆心Z"),
                k_cut=f(self.cyl_k,"切割平面k"),
                axis_dir=axis_dir, surf_type=surf_t,
                axis_min=f(self.cyl_amin,"轴向起点"), axis_max=f(self.cyl_amax,"轴向终点"),
                proj_shape=proj_s, proj_R=f(self.cyl_pR,"投影圆半径"), **tp)
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        meta = {"surface_name":"柱面", "traj_name":("栅形" if tp["traj_type"]=="G" else "螺旋线")+"轨迹"}
        self._finish(pts, meta, self.cyl_btn_save, self.cyl_info, "柱面轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 锥面轨迹页面
    # ────────────────────────────────────────────────────────────────
    def _build_conical_page(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        w = QWidget(); scroll.setWidget(w)
        layout = QVBoxLayout(w); layout.setSpacing(6); layout.setContentsMargins(6,6,6,6)

        grp1 = QGroupBox("锥面参数")
        g1   = QVBoxLayout(grp1)
        self.con_cmb_type = QComboBox()
        self.con_cmb_type.addItems(["凸锥 (Convex)", "凹锥 (Concave)"])
        combox_input(g1, "锥体类型：", self.con_cmb_type)
        self.con_alpha, r1 = lineedit_input("半顶角 α (度)：",  "30")
        self.con_H,     r2 = lineedit_input("高度 H (mm)：",    "50")
        for r in [r1,r2]: g1.addLayout(r)
        layout.addWidget(grp1)

        grp2 = QGroupBox("覆盖范围")
        g2   = QVBoxLayout(grp2)
        self.con_cmb_cover = QComboBox()
        self.con_cmb_cover.addItems(["全覆盖（底面圆）", "局部矩形", "局部圆形"])
        combox_input(g2, "覆盖类型：", self.con_cmb_cover)
        self.con_rx1, rc1 = lineedit_input("矩形 X_min (mm)：", "0")
        self.con_rx2, rc2 = lineedit_input("矩形 X_max (mm)：", "0")
        self.con_ry1, rc3 = lineedit_input("矩形 Y_min (mm)：", "0")
        self.con_ry2, rc4 = lineedit_input("矩形 Y_max (mm)：", "0")
        self.con_cR,  rc5 = lineedit_input("圆形半径 (mm)：",   "0")
        self.con_cx,  rc6 = lineedit_input("圆心 X (mm)：",     "0")
        self.con_cy,  rc7 = lineedit_input("圆心 Y (mm)：",     "0")
        for r in [rc1,rc2,rc3,rc4,rc5,rc6,rc7]: g2.addLayout(r)

        def _con_cover_changed(idx):
            for ww in [self.con_rx1,self.con_rx2,self.con_ry1,self.con_ry2]:
                ww.setVisible(idx==1)
            for ww in [self.con_cR,self.con_cx,self.con_cy]:
                ww.setVisible(idx==2)
        self.con_cmb_cover.currentIndexChanged.connect(_con_cover_changed)
        _con_cover_changed(0)
        layout.addWidget(grp2)

        grp3, self.con_t, self.con_dir, self.con_st, self.con_sp, self.con_pt, self.con_arc = \
            self._build_traj_group("con")
        layout.addWidget(grp3)

        grp4 = QGroupBox("输出设置")
        g4   = QVBoxLayout(grp4)
        self.con_fname, rf = lineedit_input("文件名：", "conical_traj")
        g4.addLayout(rf); layout.addWidget(grp4)

        btn_row = QHBoxLayout()
        self.con_btn_gen  = QPushButton("生成轨迹")
        self.con_btn_save = QPushButton("保存 TXT"); self.con_btn_save.setEnabled(False)
        btn_row.addWidget(self.con_btn_gen); btn_row.addWidget(self.con_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.con_info = QLabel(""); self.con_info.setWordWrap(True)
        layout.addWidget(self.con_info); layout.addStretch()

        self.con_btn_gen.clicked.connect(self._do_generate_conical)
        self.con_btn_save.clicked.connect(
            lambda: self._do_save("锥面轨迹", self.con_fname.text(), is_surface=True))
        return scroll

    def _do_generate_conical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            tp = self._read_traj(self.con_t, self.con_dir, self.con_st,
                                  self.con_sp, self.con_pt, self.con_arc)
            cone_t  = 1 if self.con_cmb_type.currentIndex() == 0 else 2
            cover_t = self.con_cmb_cover.currentIndex() + 1
            pts = generate_conical(
                cone_type=cone_t,
                alpha_deg=f(self.con_alpha,"半顶角α"), H=f(self.con_H,"高度H"),
                cover_type=cover_t,
                rect_xmin=f(self.con_rx1,"X_min"), rect_xmax=f(self.con_rx2,"X_max"),
                rect_ymin=f(self.con_ry1,"Y_min"), rect_ymax=f(self.con_ry2,"Y_max"),
                circ_R=f(self.con_cR,"圆形半径"),
                circ_xc=f(self.con_cx,"圆心X"), circ_yc=f(self.con_cy,"圆心Y"),
                **tp)
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        meta = {"surface_name":"锥面", "traj_name":("栅形" if tp["traj_type"]=="G" else "螺旋线")+"轨迹"}
        self._finish(pts, meta, self.con_btn_save, self.con_info, "锥面轨迹", is_surface=True)


    # ════════════════════════════════════════════════════════════════════
    # 曲面轨迹 —— 统一选择器页面（顶部下拉框 + 子页 QStackedWidget）
    # ════════════════════════════════════════════════════════════════════
    # ────────────────────────────────────────────────────────────────
    # 平面轨迹页面（栅形 + 螺旋线，Z=0，法向量(0,0,1)）
    # ────────────────────────────────────────────────────────────────
    def _build_planar_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # ① 形状参数
        grp1 = QGroupBox("形状参数")
        g1 = QVBoxLayout(grp1)
        self.pl_cmb_shape = QComboBox()
        self.pl_cmb_shape.addItems(["矩形 (Rectangle)", "圆形 (Circle)"])
        combox_input(g1, "形状类型：", self.pl_cmb_shape)
        self.pl_edt_A, rowA = lineedit_input("矩形长 A (mm)：", "100")
        self.pl_edt_B, rowB = lineedit_input("矩形宽 B (mm)：", "100")
        self.pl_edt_R, rowR = lineedit_input("圆形半径 R (mm)：", "50")
        g1.addLayout(rowA); g1.addLayout(rowB); g1.addLayout(rowR)
        layout.addWidget(grp1)

        # ② 轨迹类型
        grp2 = QGroupBox("轨迹类型")
        g2 = QVBoxLayout(grp2)
        self.pl_cmb_traj = QComboBox()
        self.pl_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g2, "轨迹类型：", self.pl_cmb_traj)
        layout.addWidget(grp2)

        # ③ 栅形参数
        grp3 = QGroupBox("栅形参数")
        g3 = QVBoxLayout(grp3)
        self.pl_cmb_dir = QComboBox()
        self.pl_cmb_dir.addItems(["平行于 X 轴（沿 Y 方向推进）",
                                   "平行于 Y 轴（沿 X 方向推进）"])
        combox_input(g3, "扫描方向：", self.pl_cmb_dir)
        self.pl_edt_step,    row_st  = lineedit_input("点间步长 (mm)：", "1.0")
        self.pl_edt_spacing, row_sp  = lineedit_input("线间距 (mm)：",   "5.0")
        g3.addLayout(row_st); g3.addLayout(row_sp)
        self.pl_cmb_cover = QComboBox()
        self.pl_cmb_cover.addItems(["完全覆盖形状_覆盖整个平面", "仅覆盖子区域_设置的区域"])
        combox_input(g3, "覆盖类型：", self.pl_cmb_cover)
        self.pl_lbl_sub = QLabel("── 子区域参数 ──")
        g3.addWidget(self.pl_lbl_sub)
        self.pl_edt_sx0, row_sx0 = lineedit_input("左下角 X₀ (mm)：", "0")
        self.pl_edt_sy0, row_sy0 = lineedit_input("左下角 Y₀ (mm)：", "0")
        self.pl_edt_sC,  row_sC  = lineedit_input("区域长 C (mm)：",  "10")
        self.pl_edt_sD,  row_sD  = lineedit_input("区域宽 D (mm)：",  "10")
        for row in [row_sx0, row_sy0, row_sC, row_sD]:
            g3.addLayout(row)
        layout.addWidget(grp3)

        # ④ 螺旋线参数
        grp4 = QGroupBox("螺旋线参数")
        g4 = QVBoxLayout(grp4)
        self.pl_edt_pitch,   row_pit = lineedit_input("螺距（每圈半径增量，mm）：", "5.0")
        self.pl_edt_arcstep, row_as  = lineedit_input("弧长步长（点间距，mm）：",   "1.0")
        g4.addLayout(row_pit); g4.addLayout(row_as)
        self.pl_cmb_spiral_cover = QComboBox()
        self.pl_cmb_spiral_cover.addItems(["圆形覆盖范围", "矩形覆盖范围"])
        combox_input(g4, "螺旋覆盖类型：", self.pl_cmb_spiral_cover)
        self.pl_edt_Rmax, row_rm = lineedit_input("最大半径 R_max (mm)：", "50")
        g4.addLayout(row_rm)
        self.pl_lbl_srect = QLabel("── 矩形范围参数 ──")
        g4.addWidget(self.pl_lbl_srect)
        self.pl_edt_sxmin, row_sxn = lineedit_input("X_min (mm)：",  "0")
        self.pl_edt_symin, row_syn = lineedit_input("Y_min (mm)：",  "0")
        self.pl_edt_sxmax, row_sxx = lineedit_input("X_max (mm)：", "100")
        self.pl_edt_symax, row_syx = lineedit_input("Y_max (mm)：", "100")
        for row in [row_sxn, row_syn, row_sxx, row_syx]:
            g4.addLayout(row)
        layout.addWidget(grp4)

        # ⑤ 输出
        grp5 = QGroupBox("输出设置")
        g5 = QVBoxLayout(grp5)
        self.pl_edt_fname, row_fn = lineedit_input("文件名：", "planar_traj")
        g5.addLayout(row_fn)
        layout.addWidget(grp5)

        btn_row = QHBoxLayout()
        self.pl_btn_gen  = QPushButton("生成轨迹")
        self.pl_btn_save = QPushButton("保存 TXT")
        self.pl_btn_save.setEnabled(False)
        btn_row.addWidget(self.pl_btn_gen)
        btn_row.addWidget(self.pl_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.pl_info_lbl = QLabel("")
        self.pl_info_lbl.setWordWrap(True)
        layout.addWidget(self.pl_info_lbl)
        layout.addStretch()

        # 信号
        self.pl_cmb_shape.currentIndexChanged.connect(self._pl_shape_changed)
        self.pl_cmb_traj.currentIndexChanged.connect(self._pl_traj_changed)
        self.pl_cmb_cover.currentIndexChanged.connect(self._pl_cover_changed)
        self.pl_cmb_spiral_cover.currentIndexChanged.connect(self._pl_spiral_cover_changed)
        self.pl_btn_gen.clicked.connect(self._do_generate_planar)
        self.pl_btn_save.clicked.connect(
            lambda: self._do_save("平面轨迹", self.pl_edt_fname.text()))
        self._pl_shape_changed()
        self._pl_traj_changed()
        self._pl_cover_changed()
        self._pl_spiral_cover_changed()
        return scroll

    def _pl_shape_changed(self):
        is_r = self.pl_cmb_shape.currentIndex() == 0
        for w in [self.pl_edt_A, self.pl_edt_B]: w.setVisible(is_r)
        self.pl_edt_R.setVisible(not is_r)

    def _pl_traj_changed(self):
        is_raster = (self.pl_cmb_traj.currentIndex() == 0)
        # 栅形控件
        for w in [self.pl_cmb_dir, self.pl_edt_step, self.pl_edt_spacing,
                  self.pl_cmb_cover]:
            w.setVisible(is_raster)
        # 子区域（受 cover 控制，先全部按 raster 来，再由 _pl_cover_changed 精细控制）
        self.pl_lbl_sub.setVisible(is_raster and self.pl_cmb_cover.currentIndex() == 1)
        for w in [self.pl_edt_sx0, self.pl_edt_sy0, self.pl_edt_sC, self.pl_edt_sD]:
            w.setVisible(is_raster and self.pl_cmb_cover.currentIndex() == 1)
        # 螺旋线控件
        for w in [self.pl_edt_pitch, self.pl_edt_arcstep, self.pl_cmb_spiral_cover]:
            w.setVisible(not is_raster)
        self._pl_spiral_cover_changed()

    def _pl_cover_changed(self):
        is_raster = (self.pl_cmb_traj.currentIndex() == 0)
        sub = is_raster and (self.pl_cmb_cover.currentIndex() == 1)
        self.pl_lbl_sub.setVisible(sub)
        for w in [self.pl_edt_sx0, self.pl_edt_sy0, self.pl_edt_sC, self.pl_edt_sD]:
            w.setVisible(sub)

    def _pl_spiral_cover_changed(self):
        is_raster = (self.pl_cmb_traj.currentIndex() == 0)
        is_circ = (self.pl_cmb_spiral_cover.currentIndex() == 0)
        self.pl_edt_Rmax.setVisible(not is_raster and is_circ)
        self.pl_lbl_srect.setVisible(not is_raster and not is_circ)
        for w in [self.pl_edt_sxmin, self.pl_edt_symin,
                  self.pl_edt_sxmax, self.pl_edt_symax]:
            w.setVisible(not is_raster and not is_circ)

    def _do_generate_planar(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            shape = "R" if self.pl_cmb_shape.currentIndex() == 0 else "C"
            traj  = "G" if self.pl_cmb_traj.currentIndex() == 0 else "S"
            p = {"shape": shape}
            if shape == "R":
                p["rect_A"] = f(self.pl_edt_A, "矩形长A")
                p["rect_B"] = f(self.pl_edt_B, "矩形宽B")
            else:
                p["circle_R"] = f(self.pl_edt_R, "圆形半径R")

            if traj == "G":
                p["direction"]    = "X" if self.pl_cmb_dir.currentIndex() == 0 else "Y"
                p["step_len"]     = f(self.pl_edt_step,    "步长")
                p["line_spacing"] = f(self.pl_edt_spacing, "线间距")
                p["cover_type"]   = self.pl_cmb_cover.currentIndex() + 1
                if p["cover_type"] == 2:
                    p["sub_x0"] = f(self.pl_edt_sx0, "左下角X₀")
                    p["sub_y0"] = f(self.pl_edt_sy0, "左下角Y₀")
                    p["sub_C"]  = f(self.pl_edt_sC,  "区域长C")
                    p["sub_D"]  = f(self.pl_edt_sD,  "区域宽D")
                pts = generate_planar_raster(**p)
                tname = "栅形轨迹"
            else:
                p["pitch"]    = f(self.pl_edt_pitch,   "螺距")
                p["arc_step"] = f(self.pl_edt_arcstep, "弧长步长")
                p["spiral_cover_type"] = self.pl_cmb_spiral_cover.currentIndex() + 1
                if p["spiral_cover_type"] == 1:
                    p["spiral_R_max"] = f(self.pl_edt_Rmax, "最大半径R_max")
                else:
                    p["spiral_xmin"] = f(self.pl_edt_sxmin, "X_min")
                    p["spiral_ymin"] = f(self.pl_edt_symin, "Y_min")
                    p["spiral_xmax"] = f(self.pl_edt_sxmax, "X_max")
                    p["spiral_ymax"] = f(self.pl_edt_symax, "Y_max")
                pts = generate_planar_spiral(**p)
                tname = "螺旋线轨迹"
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return

        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点，请检查参数"); return

        sname = "矩形" if shape == "R" else "圆形"
        params = {"shape": shape, "traj_type": traj,
                  "rect_A": p.get("rect_A", 0), "rect_B": p.get("rect_B", 0),
                  "circle_R": p.get("circle_R", 0)}
        self._finish(pts, params, self.pl_btn_save, self.pl_info_lbl,
                     f"{sname}{tname}")

    def _build_surface_selector_page(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setSpacing(6)
        outer_layout.setContentsMargins(6, 6, 6, 6)

        selector_grp = QGroupBox("轨迹类型选择")
        sel_layout   = QVBoxLayout(selector_grp)
        self.surf_cmb = QComboBox()
        self.surf_cmb.addItems([
            "—— 请选择轨迹类型 ——",
            "平面轨迹 (Planar)",
            "非球面 (Aspherical)",
            "球面 (Spherical)",
            "柱面 (Cylindrical)",
            "锥面 (Conical)",
        ])
        self.surf_cmb.setFixedHeight(30)
        sel_layout.addWidget(self.surf_cmb)
        outer_layout.addWidget(selector_grp)

        self.surf_stack = QStackedWidget()

        hint = QWidget()
        h_lay = QVBoxLayout(hint)
        h_lay.addStretch()
        lbl = QLabel("↑  请从上方下拉框选择轨迹类型")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#888888; font-size:12px;")
        h_lay.addWidget(lbl)
        h_lay.addStretch()
        self.surf_stack.addWidget(hint)                           # idx 0

        self.surf_stack.addWidget(self._build_planar_page())      # idx 1
        self.surf_stack.addWidget(self._build_aspherical_page())  # idx 2
        self.surf_stack.addWidget(self._build_spherical_page())   # idx 3
        self.surf_stack.addWidget(self._build_cylindrical_page()) # idx 4
        self.surf_stack.addWidget(self._build_conical_page())     # idx 5

        outer_layout.addWidget(self.surf_stack, 1)

        self.surf_cmb.currentIndexChanged.connect(
            lambda idx: self.surf_stack.setCurrentIndex(idx))

        return outer

    # ────────────────────────────────────────────────────────────────
    # 非球面页面
    # ────────────────────────────────────────────────────────────────
    def _build_aspherical_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        grp1 = QGroupBox("非球面基本参数")
        g1 = QVBoxLayout(grp1)
        self.asp_edt_R,   row_R   = lineedit_input("曲率半径 R (mm，正凸负凹)：", "100")
        self.asp_edt_k,   row_k   = lineedit_input("圆锥常数 k：", "0")
        self.asp_edt_off, row_off = lineedit_input("离轴量 offcenter (mm)：", "0")
        for row in [row_R, row_k, row_off]:
            g1.addLayout(row)
        layout.addWidget(grp1)

        grp2 = QGroupBox("多项式系数（不使用填 0）")
        g2 = QVBoxLayout(grp2)
        self.asp_edt_A4,  row_A4  = lineedit_input("A4：",  "0")
        self.asp_edt_A6,  row_A6  = lineedit_input("A6：",  "0")
        self.asp_edt_A8,  row_A8  = lineedit_input("A8：",  "0")
        self.asp_edt_A10, row_A10 = lineedit_input("A10：", "0")
        self.asp_edt_A12, row_A12 = lineedit_input("A12：", "0")
        self.asp_edt_A14, row_A14 = lineedit_input("A14：", "0")
        for row in [row_A4, row_A6, row_A8, row_A10, row_A12, row_A14]:
            g2.addLayout(row)
        layout.addWidget(grp2)

        grp3 = QGroupBox("非球面口径")
        g3 = QVBoxLayout(grp3)
        self.asp_edt_W, row_W = lineedit_input("X方向总宽度 (mm)：", "50")
        self.asp_edt_L, row_L = lineedit_input("Y方向总长度 (mm)：", "50")
        g3.addLayout(row_W); g3.addLayout(row_L)
        layout.addWidget(grp3)

        grp4 = QGroupBox("轨迹边界")
        g4 = QVBoxLayout(grp4)
        self.asp_cmb_bound = QComboBox()
        self.asp_cmb_bound.addItems([
            "全口径矩形边界",
            "局部矩形边界",
            "局部圆形边界",
        ])
        combox_input(g4, "边界类型：", self.asp_cmb_bound)

        self.asp_lbl_rect = QLabel("── 矩形边界参数 ──")
        g4.addWidget(self.asp_lbl_rect)
        self.asp_edt_xmin, row_xn = lineedit_input("X_min (mm)：", "-25")
        self.asp_edt_xmax, row_xx = lineedit_input("X_max (mm)：",  "25")
        self.asp_edt_ymin, row_yn = lineedit_input("Y_min (mm)：", "-25")
        self.asp_edt_ymax, row_yx = lineedit_input("Y_max (mm)：",  "25")
        for row in [row_xn, row_xx, row_yn, row_yx]:
            g4.addLayout(row)

        self.asp_lbl_circ = QLabel("── 圆形边界参数 ──")
        g4.addWidget(self.asp_lbl_circ)
        self.asp_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "20")
        self.asp_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.asp_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        for row in [row_cR, row_cxc, row_cyc]:
            g4.addLayout(row)
        layout.addWidget(grp4)

        grp5 = QGroupBox("轨迹参数")
        g5 = QVBoxLayout(grp5)
        self.asp_cmb_traj = QComboBox()
        self.asp_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g5, "轨迹类型：", self.asp_cmb_traj)
        self.asp_cmb_dir = QComboBox()
        self.asp_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g5, "栅形方向：", self.asp_cmb_dir)
        self.asp_edt_spacing, row_sp  = lineedit_input("间距 (mm)：",   "0.8")
        self.asp_edt_step,    row_st  = lineedit_input("步长 (mm)：",   "0.25")
        self.asp_edt_pitch,   row_pit = lineedit_input("间距 (mm)：",   "0.8")
        self.asp_edt_arcstep, row_as  = lineedit_input("步长 (mm)：",   "0.25")
        for row in [row_sp, row_st]:
            g5.addLayout(row)
        layout.addWidget(grp5)

        grp6 = QGroupBox("输出设置")
        g6 = QVBoxLayout(grp6)
        self.asp_edt_fname, row_fn = lineedit_input("文件名：", "aspherical_traj")
        g6.addLayout(row_fn)
        layout.addWidget(grp6)

        btn_row = QHBoxLayout()
        self.asp_btn_gen  = QPushButton("生成轨迹")
        self.asp_btn_save = QPushButton("保存 TXT")
        self.asp_btn_save.setEnabled(False)
        btn_row.addWidget(self.asp_btn_gen)
        btn_row.addWidget(self.asp_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.asp_info_lbl = QLabel("")
        self.asp_info_lbl.setWordWrap(True)
        layout.addWidget(self.asp_info_lbl)
        layout.addStretch()

        self.asp_cmb_bound.currentIndexChanged.connect(self._asp_bound_changed)
        self.asp_cmb_traj.currentIndexChanged.connect(self._asp_traj_changed)
        self.asp_btn_gen.clicked.connect(self._do_generate_aspherical)
        self.asp_btn_save.clicked.connect(
            lambda: self._do_save("非球面轨迹", self.asp_edt_fname.text(), is_surface=True))
        self._asp_bound_changed()
        self._asp_traj_changed()
        return scroll

    def _asp_bound_changed(self):
        idx = self.asp_cmb_bound.currentIndex()
        show_rect = (idx == 1)
        show_circ = (idx == 2)
        self.asp_lbl_rect.setVisible(show_rect)
        for w in [self.asp_edt_xmin, self.asp_edt_xmax,
                  self.asp_edt_ymin, self.asp_edt_ymax]:
            w.setVisible(show_rect)
        self.asp_lbl_circ.setVisible(show_circ)
        for w in [self.asp_edt_cR, self.asp_edt_cxc, self.asp_edt_cyc]:
            w.setVisible(show_circ)

    def _asp_traj_changed(self):
        is_raster = (self.asp_cmb_traj.currentIndex() == 0)
        self.asp_cmb_dir.setVisible(is_raster)
        # 间距/步长始终显示，不随轨迹类型切换
        self.asp_edt_spacing.setVisible(True)
        self.asp_edt_step.setVisible(True)
        self.asp_edt_pitch.setVisible(False)
        self.asp_edt_arcstep.setVisible(False)

    def _do_generate_aspherical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            R   = f(self.asp_edt_R,   "曲率半径R")
            k   = f(self.asp_edt_k,   "圆锥常数k")
            off = f(self.asp_edt_off, "离轴量")
            A4  = f(self.asp_edt_A4,  "A4")
            A6  = f(self.asp_edt_A6,  "A6")
            A8  = f(self.asp_edt_A8,  "A8")
            A10 = f(self.asp_edt_A10, "A10")
            A12 = f(self.asp_edt_A12, "A12")
            A14 = f(self.asp_edt_A14, "A14")
            W   = f(self.asp_edt_W,   "X方向宽度")
            L   = f(self.asp_edt_L,   "Y方向长度")
            bound = self.asp_cmb_bound.currentIndex() + 1
            traj  = "G" if self.asp_cmb_traj.currentIndex() == 0 else "S"
            dire  = "X" if self.asp_cmb_dir.currentIndex()  == 0 else "Y"
            step_len     = f(self.asp_edt_step,    "步长")
            line_spacing = f(self.asp_edt_spacing, "间距")
            pitch        = line_spacing
            arc_step     = step_len
            kwargs = dict(R=R, k=k, A4=A4, A6=A6, A8=A8, A10=A10, A12=A12, A14=A14,
                          offcenter=off, traj_type=traj, direction=dire,
                          step_len=step_len, line_spacing=line_spacing,
                          pitch=pitch, arc_step=arc_step,
                          bound_type=bound, full_width=W, full_length=L)
            if bound == 2:
                kwargs.update(rect_xmin=f(self.asp_edt_xmin,"X_min"),
                              rect_xmax=f(self.asp_edt_xmax,"X_max"),
                              rect_ymin=f(self.asp_edt_ymin,"Y_min"),
                              rect_ymax=f(self.asp_edt_ymax,"Y_max"))
            elif bound == 3:
                kwargs.update(circ_R=f(self.asp_edt_cR,"圆形半径"),
                              circ_xc=f(self.asp_edt_cxc,"圆心X"),
                              circ_yc=f(self.asp_edt_cyc,"圆心Y"))
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_aspherical(**kwargs)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        params = {"surface_name": "非球面", "traj_name": tname + "轨迹"}
        self._finish(pts, params, self.asp_btn_save, self.asp_info_lbl,
                     f"非球面{tname}轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 球面页面
    # ────────────────────────────────────────────────────────────────
    def _build_spherical_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        grp1 = QGroupBox("球面参数")
        g1 = QVBoxLayout(grp1)
        self.sph_edt_R,  row_R  = lineedit_input("球体半径 R (正数, mm)：", "50")
        self.sph_edt_zc, row_zc = lineedit_input("球心 Z 坐标 zc：", "0")
        self.sph_edt_h,  row_h  = lineedit_input("球冠高度 h (mm，0 < h ≤ 2R)：", "10")
        g1.addLayout(row_R); g1.addLayout(row_zc); g1.addLayout(row_h)
        self.sph_cmb_type = QComboBox()
        self.sph_cmb_type.addItems(["凸球面 (Convex)", "凹球面 (Concave)"])
        combox_input(g1, "表面类型：", self.sph_cmb_type)
        layout.addWidget(grp1)

        grp2 = QGroupBox("轨迹参数")
        g2 = QVBoxLayout(grp2)
        self.sph_cmb_traj = QComboBox()
        self.sph_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g2, "轨迹类型：", self.sph_cmb_traj)
        self.sph_cmb_dir = QComboBox()
        self.sph_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g2, "栅形方向：", self.sph_cmb_dir)
        self.sph_edt_spacing, row_sp  = lineedit_input("间距 (mm)：",   "0.8")
        self.sph_edt_step,    row_st  = lineedit_input("步长 (mm)：",   "0.25")
        self.sph_edt_pitch,   row_pit = lineedit_input("间距 (mm)：",   "0.8")
        self.sph_edt_arcstep, row_as  = lineedit_input("步长 (mm)：",   "0.25")
        for row in [row_sp, row_st]:
            g2.addLayout(row)
        layout.addWidget(grp2)

        grp3 = QGroupBox("输出设置")
        g3 = QVBoxLayout(grp3)
        self.sph_edt_fname, row_fn = lineedit_input("文件名：", "spherical_traj")
        g3.addLayout(row_fn)
        layout.addWidget(grp3)

        btn_row = QHBoxLayout()
        self.sph_btn_gen  = QPushButton("生成轨迹")
        self.sph_btn_save = QPushButton("保存 TXT")
        self.sph_btn_save.setEnabled(False)
        btn_row.addWidget(self.sph_btn_gen)
        btn_row.addWidget(self.sph_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.sph_info_lbl = QLabel("")
        self.sph_info_lbl.setWordWrap(True)
        layout.addWidget(self.sph_info_lbl)
        layout.addStretch()

        self.sph_cmb_traj.currentIndexChanged.connect(self._sph_traj_changed)
        self.sph_btn_gen.clicked.connect(self._do_generate_spherical)
        self.sph_btn_save.clicked.connect(
            lambda: self._do_save("球面轨迹", self.sph_edt_fname.text(), is_surface=True))
        self._sph_traj_changed()
        return scroll

    def _sph_traj_changed(self):
        is_raster = (self.sph_cmb_traj.currentIndex() == 0)
        self.sph_cmb_dir.setVisible(is_raster)
        self.sph_edt_spacing.setVisible(True)
        self.sph_edt_step.setVisible(True)
        self.sph_edt_pitch.setVisible(False)
        self.sph_edt_arcstep.setVisible(False)

    def _do_generate_spherical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            R    = f(self.sph_edt_R,  "球体半径R")
            zc   = f(self.sph_edt_zc, "球心Z坐标")
            h    = f(self.sph_edt_h,  "球冠高度h")
            surf = "convex" if self.sph_cmb_type.currentIndex() == 0 else "concave"
            traj = "G" if self.sph_cmb_traj.currentIndex() == 0 else "S"
            dire = "X" if self.sph_cmb_dir.currentIndex()  == 0 else "Y"
            step_len     = f(self.sph_edt_step,    "步长")
            line_spacing = f(self.sph_edt_spacing, "间距")
            pitch        = line_spacing
            arc_step     = step_len
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_spherical(R=R, zc=zc, surf_type=surf, h=h,
                                     traj_type=traj, direction=dire,
                                     step_len=step_len, line_spacing=line_spacing,
                                     pitch=pitch, arc_step=arc_step)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        surf_cn = "凸球面" if surf == "convex" else "凹球面"
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹"}
        self._finish(pts, params, self.sph_btn_save, self.sph_info_lbl,
                     f"{surf_cn}{tname}轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 柱面页面
    # ────────────────────────────────────────────────────────────────
    def _build_cylindrical_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        grp1 = QGroupBox("柱面几何参数")
        g1 = QVBoxLayout(grp1)
        self.cyl_cmb_axis = QComboBox()
        self.cyl_cmb_axis.addItems(["轴线平行 Y 轴", "轴线平行 X 轴"])
        combox_input(g1, "轴线方向：", self.cyl_cmb_axis)
        self.cyl_cmb_type = QComboBox()
        self.cyl_cmb_type.addItems(["凸柱外表面 (Convex)", "凹柱内表面 (Concave)"])
        combox_input(g1, "曲面类型：", self.cyl_cmb_type)
        self.cyl_edt_R,    row_R   = lineedit_input("圆柱截面半径 R (mm)：", "50")
        self.cyl_edt_zc,   row_zc  = lineedit_input("圆柱截面圆心 Z：", "0")
        self.cyl_edt_k,    row_k   = lineedit_input("切割平面高度 k：", "0")
        self.cyl_edt_amin, row_an  = lineedit_input("轴线方向起点 (mm)：", "-50")
        self.cyl_edt_amax, row_ax  = lineedit_input("轴线方向终点 (mm)：",  "50")
        for row in [row_R, row_zc, row_k, row_an, row_ax]:
            g1.addLayout(row)
        layout.addWidget(grp1)

        grp2 = QGroupBox("投影区域")
        g2 = QVBoxLayout(grp2)
        self.cyl_cmb_proj = QComboBox()
        self.cyl_cmb_proj.addItems(["矩形投影区域", "圆形投影区域"])
        combox_input(g2, "投影形状：", self.cyl_cmb_proj)
        self.cyl_edt_projR, row_pR = lineedit_input("圆形投影半径 (mm)：", "20")
        g2.addLayout(row_pR)
        layout.addWidget(grp2)

        grp3 = QGroupBox("轨迹参数")
        g3 = QVBoxLayout(grp3)
        self.cyl_cmb_traj = QComboBox()
        self.cyl_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g3, "轨迹类型：", self.cyl_cmb_traj)
        self.cyl_cmb_dir = QComboBox()
        self.cyl_cmb_dir.addItems(["X方向步进", "Y方向步进"])
        combox_input(g3, "栅形方向：", self.cyl_cmb_dir)
        self.cyl_edt_spacing, row_sp  = lineedit_input("间距 (mm)：",   "0.8")
        self.cyl_edt_step,    row_st  = lineedit_input("步长 (mm)：",   "0.25")
        self.cyl_edt_pitch,   row_pit = lineedit_input("间距 (mm)：",   "0.8")
        self.cyl_edt_arcstep, row_as  = lineedit_input("步长 (mm)：",   "0.25")
        for row in [row_sp, row_st]:
            g3.addLayout(row)
        layout.addWidget(grp3)

        grp4 = QGroupBox("输出设置")
        g4 = QVBoxLayout(grp4)
        self.cyl_edt_fname, row_fn = lineedit_input("文件名：", "cylindrical_traj")
        g4.addLayout(row_fn)
        layout.addWidget(grp4)

        btn_row = QHBoxLayout()
        self.cyl_btn_gen  = QPushButton("生成轨迹")
        self.cyl_btn_save = QPushButton("保存 TXT")
        self.cyl_btn_save.setEnabled(False)
        btn_row.addWidget(self.cyl_btn_gen)
        btn_row.addWidget(self.cyl_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.cyl_info_lbl = QLabel("")
        self.cyl_info_lbl.setWordWrap(True)
        layout.addWidget(self.cyl_info_lbl)
        layout.addStretch()

        self.cyl_cmb_traj.currentIndexChanged.connect(self._cyl_traj_changed)
        self.cyl_cmb_proj.currentIndexChanged.connect(self._cyl_proj_changed)
        self.cyl_btn_gen.clicked.connect(self._do_generate_cylindrical)
        self.cyl_btn_save.clicked.connect(
            lambda: self._do_save("柱面轨迹", self.cyl_edt_fname.text(), is_surface=True))
        self._cyl_traj_changed()
        self._cyl_proj_changed()
        return scroll

    def _cyl_traj_changed(self):
        is_raster = (self.cyl_cmb_traj.currentIndex() == 0)
        self.cyl_cmb_dir.setVisible(is_raster)
        self.cyl_edt_spacing.setVisible(True)
        self.cyl_edt_step.setVisible(True)
        self.cyl_edt_pitch.setVisible(False)
        self.cyl_edt_arcstep.setVisible(False)

    def _cyl_proj_changed(self):
        is_circ = (self.cyl_cmb_proj.currentIndex() == 1)
        self.cyl_edt_projR.setVisible(is_circ)

    def _do_generate_cylindrical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            R     = f(self.cyl_edt_R,    "圆柱半径R")
            zc    = f(self.cyl_edt_zc,   "圆柱圆心Z")
            k_cut = f(self.cyl_edt_k,    "切割平面k")
            amin  = f(self.cyl_edt_amin, "轴线起点")
            amax  = f(self.cyl_edt_amax, "轴线终点")
            axis  = "Y" if self.cyl_cmb_axis.currentIndex() == 0 else "X"
            surf  = "C" if self.cyl_cmb_type.currentIndex() == 0 else "V"
            proj  = "R" if self.cyl_cmb_proj.currentIndex() == 0 else "C"
            proj_R = f(self.cyl_edt_projR, "投影圆半径") if proj == "C" else 0.0
            traj  = "G" if self.cyl_cmb_traj.currentIndex() == 0 else "S"
            dire  = "X" if self.cyl_cmb_dir.currentIndex()  == 0 else "Y"
            step_len     = f(self.cyl_edt_step,    "步长")
            line_spacing = f(self.cyl_edt_spacing, "间距")
            pitch        = line_spacing
            arc_step     = step_len
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_cylindrical(R=R, zc=zc, k_cut=k_cut,
                                       axis_dir=axis, surf_type=surf,
                                       axis_min=amin, axis_max=amax,
                                       proj_shape=proj, proj_R=proj_R,
                                       traj_type=traj, direction=dire,
                                       step_len=step_len, line_spacing=line_spacing,
                                       pitch=pitch, arc_step=arc_step)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        surf_cn = "凸柱面" if surf == "C" else "凹柱面"
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹"}
        self._finish(pts, params, self.cyl_btn_save, self.cyl_info_lbl,
                     f"{surf_cn}{tname}轨迹", is_surface=True)

    # ────────────────────────────────────────────────────────────────
    # 锥面页面
    # ────────────────────────────────────────────────────────────────
    def _build_conical_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        grp1 = QGroupBox("锥面参数")
        g1 = QVBoxLayout(grp1)
        self.con_cmb_type = QComboBox()
        self.con_cmb_type.addItems(["凸锥 (Convex Cone)", "凹锥 (Concave Cone)"])
        combox_input(g1, "锥体类型：", self.con_cmb_type)
        self.con_edt_alpha, row_al = lineedit_input("半顶角 α (度)：", "30")
        self.con_edt_H,     row_H  = lineedit_input("高度 H (正数, mm)：", "50")
        g1.addLayout(row_al); g1.addLayout(row_H)
        layout.addWidget(grp1)

        grp2 = QGroupBox("覆盖范围")
        g2 = QVBoxLayout(grp2)
        self.con_cmb_cover = QComboBox()
        self.con_cmb_cover.addItems([
            "全部覆盖（整个底面圆）",
            "局部矩形区域",
            "局部圆形区域",
        ])
        combox_input(g2, "覆盖类型：", self.con_cmb_cover)

        self.con_lbl_rect = QLabel("── 矩形区域参数 ──")
        g2.addWidget(self.con_lbl_rect)
        self.con_edt_rxmin, row_rxn = lineedit_input("X_min (mm)：", "-10")
        self.con_edt_rxmax, row_rxx = lineedit_input("X_max (mm)：",  "10")
        self.con_edt_rymin, row_ryn = lineedit_input("Y_min (mm)：", "-10")
        self.con_edt_rymax, row_ryx = lineedit_input("Y_max (mm)：",  "10")
        for row in [row_rxn, row_rxx, row_ryn, row_ryx]:
            g2.addLayout(row)

        self.con_lbl_circ = QLabel("── 圆形区域参数 ──")
        g2.addWidget(self.con_lbl_circ)
        self.con_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "10")
        self.con_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.con_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        for row in [row_cR, row_cxc, row_cyc]:
            g2.addLayout(row)
        layout.addWidget(grp2)

        grp3 = QGroupBox("轨迹参数")
        g3 = QVBoxLayout(grp3)
        self.con_cmb_traj = QComboBox()
        self.con_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g3, "轨迹类型：", self.con_cmb_traj)
        self.con_cmb_dir = QComboBox()
        self.con_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g3, "栅形方向：", self.con_cmb_dir)
        self.con_edt_spacing, row_sp  = lineedit_input("间距 (mm)：",   "0.8")
        self.con_edt_step,    row_st  = lineedit_input("步长 (mm)：",   "0.25")
        self.con_edt_pitch,   row_pit = lineedit_input("间距 (mm)：",   "0.8")
        self.con_edt_arcstep, row_as  = lineedit_input("步长 (mm)：",   "0.25")
        for row in [row_sp, row_st]:
            g3.addLayout(row)
        layout.addWidget(grp3)

        grp4 = QGroupBox("输出设置")
        g4 = QVBoxLayout(grp4)
        self.con_edt_fname, row_fn = lineedit_input("文件名：", "conical_traj")
        g4.addLayout(row_fn)
        layout.addWidget(grp4)

        btn_row = QHBoxLayout()
        self.con_btn_gen  = QPushButton("生成轨迹")
        self.con_btn_save = QPushButton("保存 TXT")
        self.con_btn_save.setEnabled(False)
        btn_row.addWidget(self.con_btn_gen)
        btn_row.addWidget(self.con_btn_save)
        layout.addLayout(btn_row)
        layout.addWidget(divider())
        self.con_info_lbl = QLabel("")
        self.con_info_lbl.setWordWrap(True)
        layout.addWidget(self.con_info_lbl)
        layout.addStretch()

        self.con_cmb_cover.currentIndexChanged.connect(self._con_cover_changed)
        self.con_cmb_traj.currentIndexChanged.connect(self._con_traj_changed)
        self.con_btn_gen.clicked.connect(self._do_generate_conical)
        self.con_btn_save.clicked.connect(
            lambda: self._do_save("锥面轨迹", self.con_edt_fname.text(), is_surface=True))
        self._con_cover_changed()
        self._con_traj_changed()
        return scroll

    def _con_cover_changed(self):
        idx = self.con_cmb_cover.currentIndex()
        show_rect = (idx == 1)
        show_circ = (idx == 2)
        self.con_lbl_rect.setVisible(show_rect)
        for w in [self.con_edt_rxmin, self.con_edt_rxmax,
                  self.con_edt_rymin, self.con_edt_rymax]:
            w.setVisible(show_rect)
        self.con_lbl_circ.setVisible(show_circ)
        for w in [self.con_edt_cR, self.con_edt_cxc, self.con_edt_cyc]:
            w.setVisible(show_circ)

    def _con_traj_changed(self):
        is_raster = (self.con_cmb_traj.currentIndex() == 0)
        self.con_cmb_dir.setVisible(is_raster)
        self.con_edt_spacing.setVisible(True)
        self.con_edt_step.setVisible(True)
        self.con_edt_pitch.setVisible(False)
        self.con_edt_arcstep.setVisible(False)

    def _do_generate_conical(self):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        try:
            alpha = f(self.con_edt_alpha, "半顶角α")
            H     = f(self.con_edt_H,     "高度H")
            ctype = self.con_cmb_type.currentIndex()  + 1  # 1=凸, 2=凹
            cover = self.con_cmb_cover.currentIndex() + 1  # 1/2/3
            traj  = "G" if self.con_cmb_traj.currentIndex() == 0 else "S"
            dire  = "X" if self.con_cmb_dir.currentIndex()  == 0 else "Y"
            step_len     = f(self.con_edt_step,    "步长")
            line_spacing = f(self.con_edt_spacing, "间距")
            pitch        = line_spacing
            arc_step     = step_len
            kwargs = dict(cone_type=ctype, alpha_deg=alpha, H=H,
                          cover_type=cover, traj_type=traj, direction=dire,
                          step_len=step_len, line_spacing=line_spacing,
                          pitch=pitch, arc_step=arc_step)
            if cover == 2:
                kwargs.update(rect_xmin=f(self.con_edt_rxmin,"X_min"),
                              rect_xmax=f(self.con_edt_rxmax,"X_max"),
                              rect_ymin=f(self.con_edt_rymin,"Y_min"),
                              rect_ymax=f(self.con_edt_rymax,"Y_max"))
            elif cover == 3:
                kwargs.update(circ_R=f(self.con_edt_cR,"圆形半径"),
                              circ_xc=f(self.con_edt_cxc,"圆心X"),
                              circ_yc=f(self.con_edt_cyc,"圆心Y"))
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_conical(**kwargs)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        surf_cn = "凸锥面" if ctype == 1 else "凹锥面"
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹"}
        self._finish(pts, params, self.con_btn_save, self.con_info_lbl,
                     f"{surf_cn}{tname}轨迹", is_surface=True)


# ════════════════════════════════════════════════════════════════════
# 主窗口
# ════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("平面轨迹生成软件")
        self.resize(1280, 800)
        self.setWindowIcon(get_icon("icon"))
        self.setStyleSheet(get_stylesheet("main"))

        self._build_ui()
        self._build_ribbon()

    # ── UI 框架（仿 ShowGui.py 的布局）─────────────────────────────
    def _build_ui(self):
        # 左侧中央：轨迹预览画布
        self.preview = PreviewCanvas(self)
        self.setCentralWidget(self.preview)

        # 右侧 DockWidget：控制台（参数输入）
        self.dock_ctrl = QDockWidget("控制台", self)
        self.dock_ctrl.setMinimumWidth(300)
        self.dock_ctrl.setMaximumWidth(380)
        self.dock_ctrl.setFeatures(QDockWidget.DockWidgetMovable)
        self.stacked_widget = ControlPanel(self)
        self.dock_ctrl.setWidget(self.stacked_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_ctrl)

        # 右侧 DockWidget：结果输出
        self.dock_term = QDockWidget("结果输出", self)
        self.dock_term.setMinimumWidth(300)
        self.dock_term.setMaximumWidth(380)
        self.dock_term.setFeatures(QDockWidget.DockWidgetMovable)
        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Consolas", 9))
        self.dock_term.setWidget(self.terminal_output)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_term)

        # 状态栏
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")

    # ── Ribbon 工具栏（完全仿照 ShowGui.py 的 init_ribbon 写法）────
    def _build_ribbon(self):
        self._ribbon = RibbonWidget(self)
        self.addToolBar(self._ribbon)

        # ── Tab：轨迹规划 ──────────────────────────────────────────
        tab_traj = self._ribbon.add_ribbon_tab("轨迹规划")

        pane_surf = tab_traj.add_ribbon_pane("轨迹规划")
        act_surf  = self._make_action("轨迹规划", "zhexian",
                                       "平面/非球面/球面/柱面/锥面轨迹规划",
                                       self._show_surface)
        pane_surf.add_ribbon_widget(RibbonButton(self, act_surf, True))

        pane_save = tab_traj.add_ribbon_pane("输出")
        act_save  = self._make_action("Save", "xlsx",
                                       "保存轨迹点到 TXT 文件",
                                       self._quick_save)
        pane_save.add_ribbon_widget(RibbonButton(self, act_save, True))
        tab_traj.add_spacer()

        # ── Tab：授权 ──────────────────────────────────────────────
        tab_lic = self._ribbon.add_ribbon_tab("authorization")
        pane_lic = tab_lic.add_ribbon_pane("授权管理")
        act_lic  = self._make_action("授权管理", "license",
                                      "查看授权状态或激活软件",
                                      self._show_license)
        pane_lic.add_ribbon_widget(RibbonButton(self, act_lic, True))
        tab_lic.add_spacer()

        # ── Tab：退出 ──────────────────────────────────────────────
        tab_exit = self._ribbon.add_ribbon_tab("退出")
        pane_exit = tab_exit.add_ribbon_pane("退出")
        act_exit  = self._make_action("退出", "exit", "关闭软件", self.close)
        pane_exit.add_ribbon_widget(RibbonButton(self, act_exit, True))
        tab_exit.add_spacer()

    def _make_action(self, caption, icon_name, tip, slot):
        act = QAction(get_icon(icon_name), caption, self)
        act.setStatusTip(tip)
        act.triggered.connect(slot)
        act.setIconVisibleInMenu(True)
        return act

    # ── Ribbon 按钮槽函数 ───────────────────────────────────────────
    def _show_license(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_license)

    def _show_surface(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_surface)

    def _quick_save(self):
        """Ribbon 上的 Save 按钮：直接触发当前活跃的保存动作"""
        idx = self.stacked_widget.currentIndex()
        sw  = self.stacked_widget
        if idx == sw.idx_surface:
            sub = sw.surf_stack.currentIndex()
            if sub == 1:
                sw._do_save("平面轨迹", sw.pl_edt_fname.text())
            elif sub == 2:
                sw._do_save("非球面轨迹", sw.asp_edt_fname.text(), is_surface=True)
            elif sub == 3:
                sw._do_save("球面轨迹", sw.sph_edt_fname.text(), is_surface=True)
            elif sub == 4:
                sw._do_save("柱面轨迹", sw.cyl_edt_fname.text(), is_surface=True)
            elif sub == 5:
                sw._do_save("锥面轨迹", sw.con_edt_fname.text(), is_surface=True)
            else:
                QMessageBox.information(self, "提示", "请先选择轨迹类型并生成轨迹")
        else:
            QMessageBox.information(self, "提示", "请先生成轨迹后再保存")


# ════════════════════════════════════════════════════════════════════
# 供应商激活码生成工具（独立窗口，仅开发用）
# ════════════════════════════════════════════════════════════════════
class KeygenDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("激活码生成工具（供应商专用）")
        self.setFixedSize(480, 220)
        from function.license_manager import generate_activation_code
        self._gen = generate_activation_code

        lay = QVBoxLayout(self)
        self._edt_hwid, r1 = lineedit_input("机器码：")
        self._edt_days, r2 = lineedit_input("授权天数：", "365")
        lay.addLayout(r1); lay.addLayout(r2)
        btn = QPushButton("生成激活码")
        lay.addWidget(btn)
        self._edt_code, r3 = lineedit_input("激活码：")
        self._edt_code.setReadOnly(True)
        self._edt_code.setStyleSheet("font-family:Consolas; color:#1a3f6f;")
        lay.addLayout(r3)
        btn.clicked.connect(self._gen_code)

    def _gen_code(self):
        from function.license_manager import generate_activation_code
        hwid = self._edt_hwid.text().strip()
        try: days = int(self._edt_days.text())
        except: QMessageBox.warning(self, "错误", "天数无效"); return
        self._edt_code.setText(generate_activation_code(hwid, days))


# ════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════
def main():
    QApplication.setStyle(QStyleFactory.create("Fusion"))
    app = QApplication.instance() or QApplication(sys.argv)

    # 启动授权检查
    ok, msg = verify_license()
    if not ok:
        # 弹出简单激活对话框
        dlg = QtWidgets.QDialog()
        dlg.setWindowTitle("软件激活")
        dlg.setFixedSize(500, 280)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"授权状态：{msg}"))

        hwid_row = QHBoxLayout()
        hwid_row.addWidget(QLabel("本机机器码："))
        hwid_edt = QLineEdit(get_hardware_id())
        hwid_edt.setReadOnly(True)
        hwid_edt.setStyleSheet("font-family:Consolas; color:#1a3f6f;")
        btn_c = QPushButton("复制")
        btn_c.setFixedWidth(46)
        btn_c.clicked.connect(lambda: QApplication.clipboard().setText(hwid_edt.text()))
        hwid_row.addWidget(hwid_edt); hwid_row.addWidget(btn_c)
        lay.addLayout(hwid_row)

        days_edt = QLineEdit("365")
        lay.addWidget(QLabel("授权天数："))
        lay.addWidget(days_edt)
        code_edt = QLineEdit()
        code_edt.setPlaceholderText("XXXXXXXX-XXXXXXXX-XXXXXXXX-XXXXXXXX")
        code_edt.setStyleSheet("font-family:Consolas;")
        lay.addWidget(QLabel("激活码："))
        lay.addWidget(code_edt)
        status_lbl = QLabel("")
        lay.addWidget(status_lbl)

        btn_row = QHBoxLayout()
        btn_ok  = QPushButton("立即激活")
        btn_skip = QPushButton("暂时跳过（试用）")
        btn_row.addWidget(btn_ok); btn_row.addWidget(btn_skip)
        lay.addLayout(btn_row)

        def do_act():
            try: d = int(days_edt.text())
            except: status_lbl.setText("天数无效"); return
            ok2, msg2 = activate(code_edt.text(), d)
            color = "#1a7a3c" if ok2 else "#c0392b"
            status_lbl.setText(msg2)
            status_lbl.setStyleSheet(f"color:{color};")
            if ok2:
                QtCore.QTimer.singleShot(1200, dlg.accept)

        btn_ok.clicked.connect(do_act)
        btn_skip.clicked.connect(dlg.accept)
        dlg.exec_()

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
