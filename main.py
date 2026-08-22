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

import numpy as np

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDockWidget, QStackedWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
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
from function.surface_trajectory import (
    SPHERICAL_WALL_THICKNESS_MM, CYLINDRICAL_WALL_THICKNESS_MM
)
from function.license_manager   import get_hardware_id, activate, verify_license
from trajectory_planning import TrajectoryPlanningMixin
from dwell_time import DwellTimeMixin



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
# 面形数据二维热力图预览（左=处理前原始面形，右=处理后面形，各带一条色标）
# ════════════════════════════════════════════════════════════════════
def _downsample_2d(data, max_side=1536):
    """大图按整数步长抽稀，避免超大干涉图拖慢 matplotlib 渲染。"""
    data = np.asarray(data, dtype=float)
    stride = 1
    while max(data.shape) / stride > max_side:
        stride += 1
    return data if stride == 1 else data[::stride, ::stride]


class SurfaceDataPreview(QWidget):
    """独立 matplotlib 热力图面板：面形/抛光斑页用 1×2 两面板，
    驻留时间求解页用 2×2 四面板（上下各两个），颜色样式保持一致。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #dfe9f5;")
        self._error = None
        self._colorbars = []
        self._axes = []
        self._canvases = []
        self._panel_hosts = []
        self._panel_count = 2
        # 四面板（驻留时间求解）大小调节：左/右/下三边留白像素数，
        # 数值越大四个图越小；顶部留白由横条避让自动决定，不在此处。
        # 两面板页面不受此设置影响。
        self._four_panel_inset = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        # 2×2 网格：两面板模式只显示第一行，四面板模式全部显示
        self._panels = QGridLayout()
        self._panels.setContentsMargins(0, 0, 0, 0)
        self._panels.setSpacing(2)

        try:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            for index in range(4):
                host = QWidget()
                host_layout = QVBoxLayout(host)
                host_layout.setContentsMargins(0, 0, 0, 0)
                host.plot_layout = host_layout
                figure = Figure(figsize=(3.6, 3.6), dpi=100)
                figure.patch.set_facecolor("#dfe9f5")
                canvas = FigureCanvasQTAgg(figure)
                canvas.setStyleSheet("background-color: #dfe9f5;")
                axis = figure.add_subplot(111)
                axis.set_facecolor("#dfe9f5")
                figure.subplots_adjust(
                    left=0.03, right=0.80, top=0.97, bottom=0.03)
                host_layout.addWidget(canvas)
                self._axes.append(axis)
                self._canvases.append(canvas)
                self._panel_hosts.append(host)
                self._panels.addWidget(host, index // 2, index % 2, 1, 1)
                self._panels.setColumnStretch(index % 2, 1)
                self._panels.setRowStretch(index // 2, 1)
        except Exception as exc:
            self._error = exc
            hint = QLabel(
                "二维图显示需要 matplotlib。请安装后重启软件。\n\n" + str(exc))
            hint.setAlignment(Qt.AlignCenter)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#c0392b; font-size:14px;")
            outer.addWidget(hint)
            return

        outer.addLayout(self._panels, 1)
        self.set_panel_count(2)

    def set_panel_count(self, count):
        """两面板（1×2）与四面板（2×2）布局切换，多余面板隐藏。"""
        count = 4 if count >= 3 else 2
        self._panel_count = count
        for index, host in enumerate(self._panel_hosts):
            host.setVisible(index < count)
            # 切回两面板时复位绘图区上边距（四面板的下移只属于求解页）
            if count == 2:
                plot_layout = getattr(host, "plot_layout", None)
                if plot_layout is not None and plot_layout.contentsMargins().top() != 0:
                    plot_layout.setContentsMargins(0, 0, 0, 0)
        # 隐藏部件不会取消其所在行的拉伸因子：两面板时必须把第二行拉伸清零，
        # 否则空行仍占据一半高度，两个面板会被挤到上半部分
        self._panels.setRowStretch(1, 1 if count == 4 else 0)
        # 四面板大小调节：只给求解页四面板加边距，两面板恢复贴边
        inset = self._four_panel_inset if count == 4 else 0
        self._panels.setContentsMargins(inset, 0, inset, inset)

    def set_panels(self, datasets, colorbar_label="面形误差 (nm)"):
        if self._error is not None:
            return
        self.set_panel_count(len(datasets))
        for cb in self._colorbars:
            try:
                cb.remove()
            except Exception:
                pass
        self._colorbars = []
        try:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap("jet").copy()
            cmap.set_bad("#d6e2f0")
        except Exception:
            cmap = "jet"
        for ax, data in zip(self._axes, datasets):
            ax.clear()
            ax.set_facecolor("#dfe9f5")
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(_downsample_2d(data), cmap=cmap, aspect="equal",
                           interpolation="nearest", origin="lower")
            cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
            if colorbar_label:
                cb.set_label(colorbar_label, color="#10243f", fontsize=9)
            cb.ax.tick_params(labelsize=8, colors="#10243f")
            cb.outline.set_edgecolor("#10243f")
            self._colorbars.append(cb)
        for canvas in self._canvases:
            canvas.draw_idle()

    def set_data(self, raw, processed, colorbar_label="面形误差 (nm)"):
        self.set_panels((raw, processed), colorbar_label)


# ════════════════════════════════════════════════════════════════════
# 预览画布（左侧中央区域）
# ════════════════════════════════════════════════════════════════════
class PreviewCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #dfe9f5;")
        self._occ_error = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)

        self._display2d = None
        self._display_surf = None
        self._display_traj = None
        self._viewer2d = None
        self._viewer_surf = None
        self._viewer_traj = None
        self._display_viewers = {}
        self._label2d = None
        self._label_surf = None
        self._label_traj = None
        self._label_hosts = []
        self._surface_view_mode = "split"
        self._surf_box = None
        self._traj_box = None
        self._surface_data_preview = None
        self._idx_surface_data = -1

        # 驻留时间功能介绍横幅：仿照 Trajectory 覆盖条的独立横条纹，
        # 显示在左侧预览区最上方（仅驻留时间页面激活时出现）。
        self._label_dwell = QLabel(self)
        self._label_dwell.setAlignment(Qt.AlignCenter)
        self._label_dwell.setWordWrap(True)
        self._label_dwell.setStyleSheet(
            "QLabel {"
            "background-color: #d6e2f0;"
            "color: #10243f;"
            "font-family: Microsoft YaHei, SimHei, Arial;"
            "font-size: 13px;"
            "font-weight: 700;"
            "padding: 3px 8px;"
            "border-bottom: 1px solid #b8c7d8;"
            "}"
        )
        self._label_dwell.hide()

        # 面形数据第二横条：处理前 / 处理后，位于介绍横幅下方，
        # 与左右两个热力图面板对齐。样式复用 _make_overlay_label，
        # 与建模/轨迹的“曲面 | …”“Trajectory | …”覆盖条完全一致。
        self._label_dwell_sub_left = None
        self._label_dwell_sub_right = None

        self._init_occ_widgets()
        self._init_surface_data_widget()

    def _init_surface_data_widget(self):
        self._surface_data_preview = SurfaceDataPreview(self)
        self._idx_surface_data = self._stack.count()
        self._stack.addWidget(self._surface_data_preview)
        hosts = getattr(self._surface_data_preview, "_panel_hosts", [])
        # 每个面板一条第二横条说明；第一行面板需避开顶部介绍横幅，第二行不用
        self._label_dwell_subs = []
        for index, host in enumerate(hosts):
            host.dwell_sub_bar_top_row = index < 2
            self._label_dwell_subs.append(self._make_overlay_label(host))
        if len(self._label_dwell_subs) >= 2:
            self._label_dwell_sub_left = self._label_dwell_subs[0]
            self._label_dwell_sub_right = self._label_dwell_subs[1]

    def _init_occ_widgets(self):
        try:
            self._occ_imports()
            self._viewer2d, self._display2d = self._make_viewer()
            page2d = QWidget()
            lay2d = QVBoxLayout(page2d)
            lay2d.setContentsMargins(0, 0, 0, 0)
            lay2d.addWidget(self._viewer2d)
            self._label2d = self._make_overlay_label(page2d)
            self._stack.addWidget(page2d)

            page3d = QWidget()
            lay3d = QHBoxLayout(page3d)
            lay3d.setContentsMargins(0, 0, 0, 0)
            lay3d.setSpacing(2)
            surf_box = QWidget(page3d)
            self._surf_box = surf_box
            surf_layout = QVBoxLayout(surf_box)
            surf_layout.setContentsMargins(0, 0, 0, 0)
            self._viewer_surf, self._display_surf = self._make_viewer()
            surf_layout.addWidget(self._viewer_surf)
            self._label_surf = self._make_overlay_label(surf_box)

            traj_box = QWidget(page3d)
            self._traj_box = traj_box
            traj_layout = QVBoxLayout(traj_box)
            traj_layout.setContentsMargins(0, 0, 0, 0)
            self._viewer_traj, self._display_traj = self._make_viewer()
            traj_layout.addWidget(self._viewer_traj)
            self._label_traj = self._make_overlay_label(traj_box)

            lay3d.addWidget(surf_box, 1)
            lay3d.addWidget(traj_box, 1)
            self._stack.addWidget(page3d)

            self._stack.setCurrentIndex(0)
            QtCore.QTimer.singleShot(0, self._sync_all_occ_views)
            QtCore.QTimer.singleShot(200, self._sync_all_occ_views)
        except Exception as exc:
            self._occ_error = exc
            label = QLabel(
                "OCC viewer initialization failed. Install pythonocc-core "
                f"in the current environment, then restart the software.\n\n{exc}"
            )
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("color:#c0392b; font-size:14px; background:#dfe9f5;")
            self._stack.addWidget(label)
            self._stack.setCurrentIndex(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_overlay_labels()
        if self._occ_error is None:
            QtCore.QTimer.singleShot(0, self._sync_all_occ_views)

    def _make_overlay_label(self, host):
        label = QLabel(host)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            "QLabel {"
            "background-color: #d6e2f0;"
            "color: #10243f;"
            "font-family: Microsoft YaHei, SimHei, Arial;"
            "font-size: 13px;"
            "font-weight: 700;"
            "padding: 3px 8px;"
            "border-bottom: 1px solid #b8c7d8;"
            "}"
        )
        label.hide()
        self._label_hosts.append((host, label))
        return label

    def _set_label(self, label, text):
        if label is None:
            return
        label.setText(text)
        label.show()
        label.raise_()
        self._place_overlay_labels()
        # 面板在半宽/全宽间切换时此刻布局尚未完成，host.width() 是旧值；
        # 延迟到事件循环再摆一次，让横条宽度跟随面板实际宽度占满
        QtCore.QTimer.singleShot(0, self._place_overlay_labels)

    def set_surface_view_mode(self, mode):
        self._surface_view_mode = "overlay" if mode == "overlay" else "split"
        self._apply_surface_view_mode()

    def _apply_surface_view_mode(self):
        if self._surf_box is None or self._traj_box is None:
            return
        overlay = self._surface_view_mode == "overlay"
        self._surf_box.setVisible(not overlay)
        self._traj_box.setVisible(True)
        if overlay and self._label_surf is not None:
            self._label_surf.hide()
        self._place_overlay_labels()
        QtCore.QTimer.singleShot(0, self._sync_all_occ_views)

    def _place_overlay_labels(self):
        offset = 0
        if self._label_dwell is not None and self._label_dwell.isVisible():
            width = max(1, self.width())
            height = max(30, self._label_dwell.heightForWidth(width))
            self._label_dwell.setGeometry(0, 0, width, height)
            self._label_dwell.raise_()
            offset = height
        # 四面板（驻留时间求解）模式：绘图区整体下移到各自横条下方，避免遮挡；
        # 两面板页面（面形/抛光斑）保持原有覆盖样式不变
        preview = self._surface_data_preview
        four_panel = preview is not None and getattr(preview, "_panel_count", 2) >= 3
        for host, label in self._label_hosts:
            if host is None or label is None:
                continue
            # 只有顶部一行面板（及 3D 覆盖条）需要避开介绍横幅
            host_offset = offset-6 if getattr(host, "dwell_sub_bar_top_row", True) else 0
            label.setGeometry(0, host_offset, max(1, host.width()), 30)
            label.raise_()
            plot_layout = getattr(host, "plot_layout", None)
            if plot_layout is not None:
                # 32 = 横条高 30 + 2px 空隙；想让四个图再偏下，把这个数调大即可
                top_gap = host_offset + 32 if four_panel else 0
                if plot_layout.contentsMargins().top() != top_gap:
                    plot_layout.setContentsMargins(0, top_gap, 0, 0)

    def set_dwell_banner(self, text):
        self._label_dwell.setText(text)
        self._label_dwell.show()
        self._place_overlay_labels()

    def clear_dwell_banner(self):
        if self._label_dwell.isVisible():
            self._label_dwell.hide()
            self._place_overlay_labels()

    def set_dwell_sub_bar(self, texts=("处理前", "处理后"), visible=True):
        """第二横条：每个热力图面板上方一条说明。
        两面板传 2 项（处理前/处理后），四面板传 4 项。"""
        if isinstance(texts, str):
            texts = (texts,)
        labels = getattr(self, "_label_dwell_subs", [])
        for index, label in enumerate(labels):
            if label is None:
                continue
            if index < len(texts):
                label.setText(texts[index])
                label.setVisible(visible)
            else:
                label.hide()
        self._place_overlay_labels()

    def hide_dwell_sub_bar(self):
        labels = getattr(self, "_label_dwell_subs", [])
        if not any(label is not None and label.isVisible() for label in labels):
            return
        for label in labels:
            if label is not None:
                label.hide()
        self._place_overlay_labels()

    def _sync_all_occ_views(self):
        for display in (self._display2d, self._display_surf, self._display_traj):
            if display is not None:
                self._sync_occ_view(display)
                try:
                    display.Repaint()
                except Exception:
                    pass

    @staticmethod
    def _occ_imports():
        from OCC.Display.backend import load_backend
        try:
            load_backend("pyqt5")
        except Exception:
            load_backend("qt-pyqt5")
        from OCC.Display.qtDisplay import qtViewer3d
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import (
            BRepBuilderAPI_MakeEdge,
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
            BRepBuilderAPI_MakeVertex,
        )
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from OCC.Core.AIS import AIS_TextLabel, AIS_Triangulation
        from OCC.Core.Poly import Poly_Triangle, Poly_Triangulation
        from OCC.Extend.DataExchange import read_stl_file
        from OCC.Core.gp import gp_Ax3, gp_Cylinder, gp_Dir, gp_Pnt, gp_Sphere
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCC.Core.TopoDS import TopoDS_Compound
        return {
            "qtViewer3d": qtViewer3d,
            "BRep_Builder": BRep_Builder,
            "BRepBuilderAPI_MakeEdge": BRepBuilderAPI_MakeEdge,
            "BRepBuilderAPI_MakeFace": BRepBuilderAPI_MakeFace,
            "BRepBuilderAPI_MakePolygon": BRepBuilderAPI_MakePolygon,
            "BRepBuilderAPI_MakeVertex": BRepBuilderAPI_MakeVertex,
            "BRepPrimAPI_MakeSphere": BRepPrimAPI_MakeSphere,
            "AIS_Triangulation": AIS_Triangulation,
            "Poly_Triangle": Poly_Triangle,
            "Poly_Triangulation": Poly_Triangulation,
            "read_stl_file": read_stl_file,
            "gp_Pnt": gp_Pnt,
            "gp_Ax3": gp_Ax3,
            "gp_Cylinder": gp_Cylinder,
            "gp_Dir": gp_Dir,
            "gp_Sphere": gp_Sphere,
            "Quantity_Color": Quantity_Color,
            "Quantity_TOC_RGB": Quantity_TOC_RGB,
            "TopoDS_Compound": TopoDS_Compound,
            "AIS_TextLabel": AIS_TextLabel,
        }

    def _make_viewer(self):
        occ = self._occ_imports()
        viewer = occ["qtViewer3d"](self)
        viewer.setMinimumSize(120, 120)
        viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viewer.InitDriver()
        viewer.qApp = QApplication.instance()
        display = viewer._display
        self._display_viewers[id(display)] = viewer
        try:
            display.set_bg_gradient_color([188, 205, 224], [223, 233, 245])
        except Exception:
            pass
        return viewer, display

    def _clear_display(self, display):
        if display is None:
            return
        try:
            display.Context.RemoveAll(True)
        except Exception:
            try:
                display.Context.EraseAll(True)
            except Exception:
                try:
                    display.EraseAll()
                except Exception:
                    pass

    def _show_hint(self, display, text):
        if display is None:
            return
        if display is self._display2d:
            self._set_label(self._label2d, text)
        elif display is self._display_surf:
            self._set_label(self._label_surf, text)
        elif display is self._display_traj:
            self._set_label(self._label_traj, text)

    def _display_text(self, display, text, point):
        if display is None:
            return
        occ = self._occ_imports()
        try:
            label = occ["AIS_TextLabel"]()
            label.SetText(str(text))
            label.SetPosition(occ["gp_Pnt"](float(point[0]), float(point[1]), float(point[2])))
            label.SetHeight(10.0)
            qc = occ["Quantity_Color"](0.0, 0.0, 0.0, occ["Quantity_TOC_RGB"])
            label.SetColor(qc)
            display.Context.Display(label, False)
        except Exception:
            pass

    def _label_anchor(self, points):
        if not points:
            return (0, 0, 0)
        arr = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        span = np.maximum(maxs - mins, 1.0)
        return (
            mins[0] + span[0] * 0.02,
            maxs[1] - span[1] * 0.06,
            maxs[2] + span[2] * 0.10 + 0.5,
        )

    def _make_compound(self, shapes):
        occ = self._occ_imports()
        compound = occ["TopoDS_Compound"]()
        builder = occ["BRep_Builder"]()
        builder.MakeCompound(compound)
        for shape in shapes:
            if shape is not None:
                builder.Add(compound, shape)
        return compound

    def _point_shape(self, x, y, z):
        occ = self._occ_imports()
        return occ["BRepBuilderAPI_MakeVertex"](occ["gp_Pnt"](float(x), float(y), float(z))).Vertex()

    def _edge_shape(self, p1, p2):
        if np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)) < 1e-9:
            return None
        occ = self._occ_imports()
        gp_Pnt = occ["gp_Pnt"]
        edge_builder = occ["BRepBuilderAPI_MakeEdge"](
            gp_Pnt(float(p1[0]), float(p1[1]), float(p1[2])),
            gp_Pnt(float(p2[0]), float(p2[1]), float(p2[2])),
        )
        try:
            return edge_builder.Edge()
        except RuntimeError:
            return None

    def _sphere_shape(self, point, radius):
        occ = self._occ_imports()
        gp_Pnt = occ["gp_Pnt"]
        return occ["BRepPrimAPI_MakeSphere"](
            gp_Pnt(float(point[0]), float(point[1]), float(point[2])), float(radius)
        ).Shape()

    def _face_shape(self, pts):
        arr = np.array(pts, dtype=float)
        if not np.isfinite(arr).all():
            return None
        unique = []
        for p in arr:
            if not any(np.linalg.norm(p - q) < 1e-8 for q in unique):
                unique.append(p)
        if len(unique) < 3:
            return None
        unique = unique[:3]
        tri = np.array(unique, dtype=float)
        area = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        if area < 1e-8:
            return None
        occ = self._occ_imports()
        gp_Pnt = occ["gp_Pnt"]
        polygon = occ["BRepBuilderAPI_MakePolygon"]()
        for p in tri:
            polygon.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        polygon.Close()
        try:
            return occ["BRepBuilderAPI_MakeFace"](polygon.Wire()).Face()
        except Exception:
            return None

    def _display_shapes(self, display, shapes, color=None, update=False,
                        hide_face_boundaries=False):
        if not shapes:
            return
        compound = self._make_compound(shapes)
        display_color = self._occ_color(color)
        displayed = None
        try:
            displayed = display.DisplayShape(compound, color=display_color, update=update)
        except TypeError:
            try:
                displayed = display.DisplayShape(compound, update=update)
            except Exception:
                pass
        except Exception:
            pass
        try:
            ais_items = displayed if isinstance(displayed, (list, tuple)) else [displayed]
            for ais in ais_items:
                if ais is not None:
                    display.Context.SetDisplayMode(ais, 1, False)
                    if display_color is not None and not isinstance(display_color, str):
                        display.Context.SetColor(ais, display_color, False)
                    if hide_face_boundaries:
                        drawer = ais.Attributes()
                        drawer.SetFaceBoundaryDraw(False)
                        drawer.SetWireDraw(False)
                        drawer.SetupOwnShadingAspect()
                        drawer.ShadingAspect().Aspect().SetEdgeOff()
                        display.Context.Redisplay(ais, False)
            display.Context.UpdateCurrentViewer()
        except Exception:
            pass

    def _occ_color(self, color):
        if color is None or isinstance(color, str):
            return color
        try:
            r, g, b = color
            occ = self._occ_imports()
            return occ["Quantity_Color"](float(r), float(g), float(b), occ["Quantity_TOC_RGB"])
        except Exception:
            return color

    def _display_surface_direct_overlay(self, display, points,
                                        break_long_edges=False, geom=None):
        """Display a narrow, depth-tested 3-D ribbon following the trajectory."""
        segments = self._trajectory_segments(
            points, max_edges=60000, break_long_edges=break_long_edges)
        if display is None or not segments:
            return False
        try:
            occ = self._occ_imports()
            nodes = []
            normals = []
            triangles = []
            span = np.ptp(np.asarray([point[:3] for point in points], dtype=float),
                          axis=0)
            half_width = max(float(np.max(span)) * 1.5e-3, 0.18)
            for p1, p2 in segments:
                if not self._cylindrical_overlay_segment_visible(
                        p1, p2, geom, half_width):
                    continue
                xyz1 = np.asarray(p1[:3], dtype=float)
                xyz2 = np.asarray(p2[:3], dtype=float)
                tangent = xyz2 - xyz1
                tangent_length = float(np.linalg.norm(tangent))
                if tangent_length <= 1.0e-9:
                    continue
                tangent /= tangent_length
                n1 = np.asarray(p1[3:6], dtype=float)
                n2 = np.asarray(p2[3:6], dtype=float)
                n1 /= max(float(np.linalg.norm(n1)), 1.0e-12)
                n2 /= max(float(np.linalg.norm(n2)), 1.0e-12)
                side1 = np.cross(n1, tangent)
                side2 = np.cross(n2, tangent)
                side1 /= max(float(np.linalg.norm(side1)), 1.0e-12)
                side2 /= max(float(np.linalg.norm(side2)), 1.0e-12)
                if float(np.dot(side1, side2)) < 0.0:
                    side2 = -side2
                first = len(nodes) + 1
                nodes.extend((
                    xyz1 + half_width * side1,
                    xyz1 - half_width * side1,
                    xyz2 + half_width * side2,
                    xyz2 - half_width * side2,
                ))
                normals.extend((n1, n1, n2, n2))
                triangles.extend((
                    (first, first + 1, first + 2),
                    (first + 1, first + 3, first + 2),
                ))
            if not triangles:
                return False

            mesh = occ["Poly_Triangulation"](
                len(nodes), len(triangles), False, True)
            for index, (node, normal) in enumerate(zip(nodes, normals), 1):
                mesh.SetNode(index, occ["gp_Pnt"](
                    float(node[0]), float(node[1]), float(node[2])))
                mesh.SetNormal(index, occ["gp_Dir"](
                    float(normal[0]), float(normal[1]), float(normal[2])))
            for index, triangle in enumerate(triangles, 1):
                mesh.SetTriangle(index, occ["Poly_Triangle"](*triangle))

            ribbon = occ["AIS_Triangulation"](mesh)
            blue = occ["Quantity_Color"](
                0.0, 0.0, 1.0, occ["Quantity_TOC_RGB"])
            drawer = ribbon.Attributes()
            drawer.SetupOwnShadingAspect()
            drawer.ShadingAspect().SetColor(blue)
            ribbon_aspect = drawer.ShadingAspect().Aspect()
            ribbon_aspect.AllowBackFace()
            display.Context.Display(ribbon, False)
            display.Context.SetDisplayMode(ribbon, 0, False)
            display.Context.UpdateCurrentViewer()
            return True
        except Exception:
            return False

    @staticmethod
    def _cylindrical_overlay_segment_visible(p1, p2, geom, half_width):
        """Hide only raster row transfers that run along the cylinder rim."""
        if not (geom and geom.get("type") == "cylindrical" and
                geom.get("surf_type", "C") == "V"):
            return True
        radius = abs(float(geom.get("R", 0.0)))
        thickness = abs(float(geom.get(
            "wall_thickness", CYLINDRICAL_WALL_THICKNESS_MM)))
        work_radius = radius - thickness
        axis_index = 1 if geom.get("axis_dir", "Y") == "Y" else 0
        cross_index = 0 if axis_index == 1 else 1

        delta_z = float(geom.get("k_cut", 0.0)) - float(
            geom.get("zc", 0.0))
        if work_radius <= 0.0 or abs(delta_z) >= work_radius:
            return True
        opening_angle = float(np.arccos(
            np.clip(abs(delta_z) / work_radius, 0.0, 1.0)))
        z_center = float(geom.get("zc", 0.0))
        z_origin = z_center - radius
        rim_gaps = []
        for point in (p1, p2):
            cross = abs(float(point[cross_index]))
            z_abs = float(point[2]) + z_origin
            theta = float(np.arctan2(cross, max(z_center - z_abs, 0.0)))
            rim_gaps.append(max(0.0, opening_angle - theta) * work_radius)

        # A serpentine row transfer changes only the cylinder-axis coordinate
        # while both endpoints remain on the same opening lip.  Do not remove
        # a real scan segment merely because one endpoint touches the lip.
        axis_delta = abs(float(p2[axis_index]) - float(p1[axis_index]))
        cross_delta = abs(float(p2[cross_index]) - float(p1[cross_index]))
        rim_tolerance = max(float(half_width), thickness)
        is_rim_transfer = (
            max(rim_gaps) <= rim_tolerance and
            axis_delta > 1.0e-9 and cross_delta <= 1.0e-6)
        return not is_rim_transfer

    def _fit_display(self, display, view="iso"):
        if display is None:
            return
        self._sync_occ_view(display)
        try:
            if view == "top":
                display.View_Top()
            else:
                display.View_Iso()
        except Exception:
            pass
        try:
            display.FitAll()
        except Exception:
            pass
        try:
            display.Repaint()
        except Exception:
            pass
        self._place_overlay_labels()

    def _refresh_empty_display(self, display):
        if display is None:
            return
        self._sync_occ_view(display)
        try:
            display.set_bg_gradient_color([188, 205, 224], [223, 233, 245])
        except Exception:
            pass
        try:
            display.View_Iso()
        except Exception:
            pass
        try:
            display.Context.UpdateCurrentViewer()
        except Exception:
            pass
        try:
            display.Repaint()
        except Exception:
            pass
        self._place_overlay_labels()

    def _sync_occ_view(self, display):
        viewer = self._display_viewers.get(id(display))
        if viewer is not None:
            try:
                viewer.show()
                viewer.resize(max(1, viewer.width()), max(1, viewer.height()))
                viewer.updateGeometry()
                viewer.update()
            except Exception:
                pass
            try:
                display.OnResize()
            except Exception:
                pass
        try:
            display.View.MustBeResized()
        except Exception:
            try:
                display.GetView().MustBeResized()
            except Exception:
                pass
        try:
            display.Context.UpdateCurrentViewer()
        except Exception:
            pass

    def _trajectory_shapes(self, points, max_points=0, max_edges=60000, break_long_edges=False):
        if not points:
            return [], []
        n_pts = len(points)
        point_shapes = []
        if max_points > 0:
            stride_pts = max(1, int(np.ceil(n_pts / max_points)))
            pts = [points[i] for i in range(0, n_pts, stride_pts)]
            point_shapes = [self._point_shape(p[0], p[1], p[2]) for p in pts]

        edge_shapes = [self._edge_shape(p1, p2) for p1, p2 in
                       self._trajectory_segments(
                           points, max_edges=max_edges,
                           break_long_edges=break_long_edges)]
        return point_shapes, edge_shapes

    def _trajectory_segments(self, points, max_edges=60000,
                             break_long_edges=False):
        if len(points) < 2 or max_edges <= 0:
            return []
        stride = max(1, int(np.ceil((len(points) - 1) / max_edges)))
        line_points = [points[index] for index in range(0, len(points), stride)]
        if line_points[-1] is not points[-1]:
            line_points.append(points[-1])
        break_limit = self._trajectory_break_limit(
            line_points) if break_long_edges else None
        segments = []
        for p1, p2 in zip(line_points, line_points[1:]):
            distance = np.linalg.norm(
                np.asarray(p1[:3], dtype=float) -
                np.asarray(p2[:3], dtype=float))
            if distance <= 1.0e-9:
                continue
            if break_limit is not None and distance > break_limit:
                continue
            segments.append((p1, p2))
        return segments

    @staticmethod
    def _surface_overlay_points(points, geom):
        """Lift display-only surface trajectories to avoid depth-buffer flicker."""
        surface_types = {"spherical", "aspherical", "cylindrical", "conical"}
        if not (geom and geom.get("type") in surface_types):
            return points
        xyz = np.asarray([point[:3] for point in points], dtype=float)
        max_span = float(np.max(np.ptp(xyz, axis=0))) if len(xyz) else 0.0
        clearance = max(max_span * 5.0e-3, 0.02)

        if geom.get("type") == "spherical":
            radius = abs(float(geom.get("R", 0.0)))
            thickness = abs(float(geom.get(
                "wall_thickness", SPHERICAL_WALL_THICKNESS_MM)))
            clearance = min(max(radius * 2.0e-3, 0.02),
                            max(thickness * 0.40, 0.02))
            if geom.get("surf_type", "convex") == "concave":
                clearance = max(
                    clearance,
                    min(max(radius * 1.0e-2, 0.02),
                        max(thickness * 2.0, 0.02)))
        elif (geom.get("type") == "cylindrical" and
              geom.get("surf_type", "C") == "V"):
            thickness = abs(float(geom.get(
                "wall_thickness", CYLINDRICAL_WALL_THICKNESS_MM)))
            clearance = max(thickness * 0.90, 0.02)
        lifted = []
        for point in points:
            if len(point) < 6:
                lifted.append(point)
                continue
            item = list(point)
            item[0] = float(item[0]) + clearance * float(point[3])
            item[1] = float(item[1]) + clearance * float(point[4])
            item[2] = float(item[2]) + clearance * float(point[5])
            lifted.append(item)
        return lifted

    @staticmethod
    def _should_break_long_edges(params):
        traj_name = str(params.get("traj_name", ""))
        if params.get("traj_type") == "S" or "螺旋" in traj_name or "Spiral" in traj_name:
            return True
        geom = params.get("geom") or {}
        return (geom.get("type") == "cylindrical" and
                int(geom.get("cover_type", 1)) in (2, 3))

    def _trajectory_break_limit(self, points):
        if len(points) < 4:
            return None
        arr = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        dists = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        dists = dists[np.isfinite(dists) & (dists > 1e-9)]
        if len(dists) < 3:
            return None
        med = float(np.median(dists))
        p95 = float(np.percentile(dists, 95))
        return max(med * 5.0, p95 * 2.0, 1e-6)

    def _marker_radius(self, points):
        arr = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        span = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        return max(float(np.nanmax(span)) * 0.015, 0.4)

    def _draw_scene_axes(self, display, points, include_z=True, geom=None):
        if not points:
            return
        try:
            arr = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if arr.size == 0:
                return
            mins = np.nanmin(arr, axis=0)
            maxs = np.nanmax(arr, axis=0)
            span = np.maximum(maxs - mins, 1.0)
            max_span = max(float(np.nanmax(span)), 1.0)
            origin = self._scene_axes_origin(arr, geom)
            length = max_span * 0.25
            if include_z and span[2] <= 1e-6:
                length_z = max_span * 0.10
            else:
                length_z = length

            axes = [
                ("X", (length, 0.0, 0.0), (0.85, 0.10, 0.08)),
                ("Y", (0.0, length, 0.0), (0.05, 0.62, 0.12)),
            ]
            if include_z:
                axes.append(("Z", (0.0, 0.0, length_z), (0.06, 0.18, 0.82)))

            marker_r = max_span * 0.012
            self._display_shapes(display, [self._sphere_shape(origin, marker_r)], color=(0.20, 0.20, 0.20), update=False)
            for name, vec, color in axes:
                end = origin + np.array(vec, dtype=float)
                edge = self._edge_shape(origin, end)
                self._display_shapes(display, [edge], color=color, update=False)
                self._display_shapes(display, [self._sphere_shape(end, marker_r * 0.8)], color=color, update=False)
                self._display_text(display, name, end + np.array(vec, dtype=float) * 0.08)
        except Exception:
            pass

    def _scene_axes_origin(self, arr, geom=None):
        if geom:
            try:
                kind = geom.get("type", "")
                if kind == "spherical":
                    R = float(geom["R"])
                    zc = float(geom.get("zc", 0.0))
                    h = float(geom["h"])
                    if geom.get("surf_type", "convex") == "convex":
                        z0 = zc + R - h
                    else:
                        z0 = zc - R
                    return np.array([0.0, 0.0, zc - z0], dtype=float)

                if kind == "cylindrical":
                    R = float(geom["R"])
                    zc = float(geom.get("zc", 0.0))
                    kcut = float(geom.get("k_cut", zc - R))
                    axis_mid = 0.5 * (float(geom["axis_min"]) + float(geom["axis_max"]))
                    if geom.get("surf_type", "C") == "C":
                        z0 = kcut
                    else:
                        z0 = zc - R
                    if geom.get("axis_dir", "Y") == "Y":
                        return np.array([0.0, axis_mid, zc - z0], dtype=float)
                    return np.array([axis_mid, 0.0, zc - z0], dtype=float)

                if kind == "conical":
                    H = float(geom["H"])
                    return np.array([0.0, 0.0, H * 0.5], dtype=float)

                if kind == "aspherical":
                    return np.array([0.0, float(geom.get("offcenter", 0.0)), 0.0], dtype=float)
            except Exception:
                pass

        mins = np.nanmin(arr, axis=0)
        maxs = np.nanmax(arr, axis=0)
        return (mins + maxs) * 0.5

    def close_occ(self):
        for viewer in (self._viewer2d, self._viewer_surf, self._viewer_traj):
            if viewer is None:
                continue
            try:
                viewer.close()
            except Exception:
                pass

    def plot(self, points, params):
        self._stack.setCurrentIndex(0)
        self.hide_dwell_sub_bar()
        QApplication.processEvents()
        if self._occ_error is not None:
            return
        display = self._display2d
        self._clear_display(display)
        if not points:
            self._show_hint(display, "No trajectory points")
            return
        self._set_label(self._label2d, f"Trajectory | points: {len(points)} | green: start | red: end")

        point_shapes, edge_shapes = self._trajectory_shapes(
            points, break_long_edges=self._should_break_long_edges(params))
        self._display_shapes(display, edge_shapes, color="BLUE", update=False)
        self._display_shapes(display, point_shapes, color="BLUE", update=False)
        self._draw_planar_boundary(display, params)
        self._draw_scene_axes(display, points, include_z=False)

        radius = self._marker_radius(points)
        self._display_shapes(display, [self._sphere_shape(points[0], radius)], color="GREEN", update=False)
        self._display_shapes(display, [self._sphere_shape(points[-1], radius)], color="RED", update=True)
        self._fit_display(display, view="top")

    def plot_surface(self, points, params):
        self._stack.setCurrentIndex(1)
        self._apply_surface_view_mode()
        self.hide_dwell_sub_bar()
        QApplication.processEvents()
        if self._occ_error is not None:
            return
        self._clear_display(self._display_surf)
        self._clear_display(self._display_traj)
        if not points:
            self._show_hint(self._display_traj, "No trajectory points")
            return
        overlay = self._surface_view_mode == "overlay"
        if not overlay:
            self._set_label(self._label_surf, f"曲面 | {params.get('surface_name', 'surface')}")
        elif self._label_surf is not None:
            self._label_surf.hide()
        traj_label = f"Trajectory | {params.get('traj_name', '')} | points: {len(points)}"
        if not overlay:
            traj_label += " | green: start | red: end"
        self._set_label(self._label_traj, traj_label)

        geom = params.get("geom")
        surface_shapes = self._surface_shapes_from_geom(geom)
        hide_surface_boundaries = bool(
            geom and geom.get("type") in {
                "spherical", "aspherical", "cylindrical", "conical"})
        display_points = self._surface_overlay_points(points, geom) if overlay else points
        direct_surface_overlay = bool(
            overlay and geom and geom.get("type") in {
                "spherical", "aspherical", "cylindrical", "conical"})
        if direct_surface_overlay:
            point_shapes, edge_shapes = [], []
        else:
            point_shapes, edge_shapes = self._trajectory_shapes(
                display_points, max_points=0, max_edges=60000,
                break_long_edges=self._should_break_long_edges(params))

        if overlay:
            if surface_shapes:
                self._display_shapes(
                    self._display_traj, surface_shapes,
                    color=(0.35, 0.42, 0.50), update=False,
                    hide_face_boundaries=hide_surface_boundaries)
            surface_direct = bool(
                direct_surface_overlay and
                self._display_surface_direct_overlay(
                    self._display_traj, display_points,
                    break_long_edges=self._should_break_long_edges(params),
                    geom=geom))
            if not surface_direct:
                if direct_surface_overlay:
                    point_shapes, edge_shapes = self._trajectory_shapes(
                        display_points, max_points=0, max_edges=60000,
                        break_long_edges=self._should_break_long_edges(params))
                self._display_shapes(
                    self._display_traj, edge_shapes, color="BLUE", update=False)
                self._display_shapes(
                    self._display_traj, point_shapes, color="BLUE", update=False)
            radius = self._marker_radius(points)
            self._display_shapes(self._display_traj, [self._sphere_shape(display_points[0], radius)], color="GREEN", update=False)
            self._display_shapes(self._display_traj, [self._sphere_shape(display_points[-1], radius)], color="RED", update=False)
            self._draw_scene_axes(self._display_traj, points, include_z=True, geom=geom)
            self._fit_display(self._display_traj, view="iso")
            return

        if surface_shapes:
            self._display_shapes(
                self._display_surf, surface_shapes,
                color=(0.35, 0.42, 0.50), update=True,
                hide_face_boundaries=hide_surface_boundaries)
        else:
            surf_point_shapes, _ = self._trajectory_shapes(points, max_points=12000, max_edges=0)
            self._display_shapes(self._display_surf, surf_point_shapes, color="BLUE", update=True)
        self._draw_scene_axes(self._display_surf, points, include_z=True, geom=geom)
        self._fit_display(self._display_surf, view="iso")

        self._display_shapes(self._display_traj, edge_shapes, color="BLUE", update=False)
        self._display_shapes(self._display_traj, point_shapes, color="BLUE", update=False)
        self._draw_scene_axes(self._display_traj, points, include_z=True, geom=geom)
        radius = self._marker_radius(points)
        self._display_shapes(self._display_traj, [self._sphere_shape(points[0], radius)], color="GREEN", update=False)
        self._display_shapes(self._display_traj, [self._sphere_shape(points[-1], radius)], color="RED", update=True)
        self._fit_display(self._display_traj, view="iso")

    def plot_dwell_surface(self, raw, processed, raw_title="处理前", processed_title="处理后",
                           colorbar_label="面形误差 (nm)"):
        """驻留时间面形数据：左侧处理前原始面形，右侧处理后面形，二维热力图。"""
        if self._surface_data_preview is None:
            return
        self._stack.setCurrentIndex(self._idx_surface_data)
        QApplication.processEvents()
        self._surface_data_preview.set_data(raw, processed, colorbar_label)
        self.set_dwell_sub_bar((raw_title, processed_title), True)

    def plot_dwell_solution(self, panels, titles, colorbar_label=""):
        """驻留时间求解：2×2 四面板（上面两个+下面两个），每面板一条第二横条。"""
        if self._surface_data_preview is None:
            return
        self._stack.setCurrentIndex(self._idx_surface_data)
        QApplication.processEvents()
        self._surface_data_preview.set_panels(panels, colorbar_label)
        self.set_dwell_sub_bar(tuple(titles), True)

    def plot_dwell_model(self, model, surface_name):
        """驻留时间建模：把模型点云以带法向着色的三角网格显示在左侧单个面板。"""
        self._stack.setCurrentIndex(1)
        self.hide_dwell_sub_bar()
        QApplication.processEvents()
        if self._occ_error is not None:
            return
        mesh = self._dwell_model_mesh(model)
        if mesh is None:
            self._show_hint(self._display_surf, "模型没有可显示的有效节点")
            return
        # 建模只显示一个曲面面板，不需要左右两栏；轨迹规划调用 plot_surface 时恢复。
        if self._surf_box is not None and self._traj_box is not None:
            self._surf_box.setVisible(True)
            self._traj_box.setVisible(False)
        self._clear_display(self._display_surf)
        self._clear_display(self._display_traj)
        self._set_label(self._label_surf, f"曲面 | {surface_name}")
        self._display_triangulation(self._display_surf, mesh)
        self._fit_display(self._display_surf, view="iso")

    def _dwell_model_mesh(self, model):
        """由建模点云构建 Poly_Triangulation；口径外无效节点跳过，过大网格降采样。"""
        occ = self._occ_imports()
        points = np.asarray(model["points"], dtype=float)
        normals = np.asarray(model["normals"], dtype=float)
        mask = np.asarray(model["mask"], dtype=bool)
        mask &= np.isfinite(points).all(axis=-1)
        rows, cols = points.shape[:2]
        stride = 1
        while (((rows + stride - 1) // stride) * ((cols + stride - 1) // stride)) > 40000:
            stride += 1
        rs = list(range(0, rows, stride)); cs = list(range(0, cols, stride))
        if rs[-1] != rows - 1:
            rs.append(rows - 1)
        if cs[-1] != cols - 1:
            cs.append(cols - 1)
        nodes = []; node_normals = []; index = {}
        for i, r in enumerate(rs):
            for j, c in enumerate(cs):
                if not mask[r, c]:
                    continue
                index[(i, j)] = len(nodes) + 1
                nodes.append(points[r, c])
                normal = np.nan_to_num(normals[r, c])
                norm = float(np.linalg.norm(normal))
                if norm < 1e-9:
                    normal = np.array((0.0, 0.0, 1.0))
                else:
                    normal = normal / norm
                node_normals.append(normal)
        triangles = []
        for i in range(len(rs) - 1):
            for j in range(len(cs) - 1):
                a = index.get((i, j)); b = index.get((i, j + 1))
                d = index.get((i + 1, j)); e = index.get((i + 1, j + 1))
                if a and b and e:
                    triangles.append((a, b, e))
                if a and e and d:
                    triangles.append((a, e, d))
        if not nodes or not triangles:
            return None
        mesh = occ["Poly_Triangulation"](len(nodes), len(triangles), False, True)
        for idx, (node, normal) in enumerate(zip(nodes, node_normals), 1):
            mesh.SetNode(idx, occ["gp_Pnt"](
                float(node[0]), float(node[1]), float(node[2])))
            mesh.SetNormal(idx, occ["gp_Dir"](
                float(normal[0]), float(normal[1]), float(normal[2])))
        for idx, tri in enumerate(triangles, 1):
            mesh.SetTriangle(idx, occ["Poly_Triangle"](*tri))
        return mesh

    def _display_triangulation(self, display, mesh, color=(0.35, 0.42, 0.50)):
        if display is None:
            return
        occ = self._occ_imports()
        ais = occ["AIS_Triangulation"](mesh)
        drawer = ais.Attributes()
        drawer.SetupOwnShadingAspect()
        drawer.ShadingAspect().SetColor(self._occ_color(color))
        drawer.ShadingAspect().Aspect().AllowBackFace()
        display.Context.Display(ais, False)
        # 模式 1（法向着色）在部分 GL 驱动下不渲染三角网格，
        # 沿用轨迹带验证过的模式 0。
        display.Context.SetDisplayMode(ais, 0, False)
        display.Context.UpdateCurrentViewer()

    def import_stl_to_shape(self, path):
        self._stack.setCurrentIndex(1)
        self._apply_surface_view_mode()
        self.hide_dwell_sub_bar()
        QApplication.processEvents()
        if self._occ_error is not None:
            raise RuntimeError(str(self._occ_error))

        occ = self._occ_imports()
        shape = occ["read_stl_file"](path)
        if shape is None or shape.IsNull():
            raise ValueError("STL model is empty or cannot be read.")

        self._clear_display(self._display_surf)
        self._clear_display(self._display_traj)
        name = os.path.basename(path)
        if self._surface_view_mode == "overlay":
            if self._label_surf is not None:
                self._label_surf.hide()
            self._set_label(self._label_traj, f"曲面 | {name}")
            self._display_shapes(self._display_traj, [shape], color=(0.35, 0.42, 0.50), update=True)
            self._fit_display(self._display_traj, view="iso")
        else:
            self._set_label(self._label_surf, f"曲面 | {name}")
            self._set_label(self._label_traj, "Trajectory | no generated trajectory")
            self._display_shapes(self._display_surf, [shape], color=(0.35, 0.42, 0.50), update=True)
            self._fit_display(self._display_surf, view="iso")
            QtCore.QTimer.singleShot(0, lambda: self._refresh_empty_display(self._display_traj))
            QtCore.QTimer.singleShot(120, lambda: self._refresh_empty_display(self._display_traj))
        return shape

    def _draw_planar_boundary(self, display, params):
        shape = params.get("shape", "R")
        if shape == "R":
            A = float(params.get("rect_A", 0.0))
            B = float(params.get("rect_B", 0.0))
            if A <= 0 or B <= 0:
                return
            corners = [
                (-A / 2, -B / 2, 0), (A / 2, -B / 2, 0),
                (A / 2, B / 2, 0), (-A / 2, B / 2, 0),
            ]
            edges = [self._edge_shape(corners[i], corners[(i + 1) % 4]) for i in range(4)]
            self._display_shapes(display, edges, color="BLACK", update=False)
            return

        R = float(params.get("circle_R", 0.0))
        if R <= 0:
            return
        circle = []
        samples = 96
        ring = [(R * np.cos(2 * np.pi * i / samples), R * np.sin(2 * np.pi * i / samples), 0) for i in range(samples)]
        for i in range(samples):
            circle.append(self._edge_shape(ring[i], ring[(i + 1) % samples]))
        self._display_shapes(display, circle, color="BLACK", update=False)

    def _surface_shapes_from_geom(self, geom):
        if not geom:
            return []
        kind = geom.get("type", "")
        if kind == "spherical":
            return self._sample_spherical_surface(geom)
        if kind == "aspherical":
            return self._sample_aspherical_surface(geom)
        if kind == "cylindrical":
            return self._sample_cylindrical_surface(geom)
        if kind == "conical":
            return self._sample_conical_surface(geom)
        return []

    def _grid_faces(self, X, Y, Z):
        shapes = []
        rows, cols = X.shape
        for r in range(rows - 1):
            for c in range(cols - 1):
                p00 = (X[r, c], Y[r, c], Z[r, c])
                p01 = (X[r, c + 1], Y[r, c + 1], Z[r, c + 1])
                p11 = (X[r + 1, c + 1], Y[r + 1, c + 1], Z[r + 1, c + 1])
                p10 = (X[r + 1, c], Y[r + 1, c], Z[r + 1, c])
                for tri in ((p00, p01, p11), (p00, p11, p10)):
                    face = self._face_shape(tri)
                    if face is not None:
                        shapes.append(face)
        return shapes

    def _sample_spherical_surface(self, geom):
        parts = self._spherical_surface_parts(geom)
        return parts["all_shapes"] if parts else []

    def _make_spherical_face(self, center_z, radius, v_min, v_max, reverse=False):
        occ = self._occ_imports()
        axis = occ["gp_Ax3"](
            occ["gp_Pnt"](0.0, 0.0, float(center_z)),
            occ["gp_Dir"](0.0, 0.0, 1.0),
        )
        sphere = occ["gp_Sphere"](axis, float(radius))
        face = occ["BRepBuilderAPI_MakeFace"](
            sphere, 0.0, 2.0 * np.pi, float(v_min), float(v_max)).Face()
        return face.Reversed() if reverse else face

    def _make_spherical_rim(self, r_outer, r_inner, z_value, samples=96):
        shapes = []
        for index in range(samples):
            a0 = 2.0 * np.pi * index / samples
            a1 = 2.0 * np.pi * (index + 1) / samples
            outer0 = (r_outer * np.cos(a0), r_outer * np.sin(a0), z_value)
            outer1 = (r_outer * np.cos(a1), r_outer * np.sin(a1), z_value)
            inner0 = (r_inner * np.cos(a0), r_inner * np.sin(a0), z_value)
            inner1 = (r_inner * np.cos(a1), r_inner * np.sin(a1), z_value)
            for tri in ((outer0, outer1, inner1), (outer0, inner1, inner0)):
                face = self._face_shape(tri)
                if face is not None:
                    shapes.append(face)
        return shapes

    def _spherical_surface_parts(self, geom):
        R = float(geom["R"])
        h = float(geom["h"])
        st = geom.get("surf_type", "convex")
        if R <= 0:
            return None

        if st == "convex":
            center_z = h - R
            v_min = float(np.arcsin(np.clip((R - h) / R, -1.0, 1.0)))
            outer_face = self._make_spherical_face(
                center_z, R, v_min, np.pi / 2.0, reverse=False)
            thickness = float(geom.get(
                "wall_thickness", SPHERICAL_WALL_THICKNESS_MM))
            inner_R = R - thickness
            opening_from_center = R - h
            r_outer = float(np.sqrt(max(0.0, R * R - opening_from_center ** 2)))
            if inner_R > 0.0 and abs(opening_from_center) < inner_R:
                v_inner = float(np.arcsin(np.clip(
                    opening_from_center / inner_R, -1.0, 1.0)))
                inner_face = self._make_spherical_face(
                    center_z, inner_R, v_inner, np.pi / 2.0, reverse=True)
                r_inner = float(np.sqrt(max(
                    0.0, inner_R * inner_R - opening_from_center ** 2)))
                occluders = [inner_face] + self._make_spherical_rim(
                    r_outer, r_inner, 0.0)
            else:
                # 极浅或接近整球的球冠没有内球开口，以平面底盖封闭厚壳。
                occluders = self._make_spherical_rim(r_outer, 0.0, 0.0)
            return {
                "all_shapes": [outer_face] + occluders,
                "occluder_shapes": occluders,
                "work_face": outer_face,
                "work_radius": R,
                "center_z": center_z,
                "v_min": v_min,
                "v_max": np.pi / 2.0,
            }

        thickness = float(geom.get(
            "wall_thickness", SPHERICAL_WALL_THICKNESS_MM))
        work_R = R - thickness
        if work_R <= 0:
            return None
        center_z = R
        opening_from_center = h - R
        if abs(opening_from_center) >= work_R:
            return None

        v_outer = float(np.arcsin(np.clip(opening_from_center / R, -1.0, 1.0)))
        v_inner = float(np.arcsin(np.clip(opening_from_center / work_R, -1.0, 1.0)))
        outer_face = self._make_spherical_face(
            center_z, R, -np.pi / 2.0, v_outer, reverse=False)
        inner_face = self._make_spherical_face(
            center_z, work_R, -np.pi / 2.0, v_inner, reverse=True)

        r_outer = float(np.sqrt(max(0.0, R * R - opening_from_center ** 2)))
        r_inner = float(np.sqrt(max(0.0, work_R * work_R - opening_from_center ** 2)))
        rim_shapes = self._make_spherical_rim(r_outer, r_inner, h)

        occluders = [outer_face] + rim_shapes
        return {
            "all_shapes": occluders + [inner_face],
            "occluder_shapes": occluders,
            "work_face": inner_face,
            "work_radius": work_R,
            "center_z": center_z,
            "v_min": -np.pi / 2.0,
            "v_max": v_inner,
        }

    def _sample_aspherical_surface(self, geom):
        R = float(geom["R"]); k = float(geom.get("k", 0.0))
        if R == 0:
            return []
        C = -1.0 / R
        off = float(geom.get("offcenter", 0.0))
        coefs = [float(geom.get(name, 0.0)) for name in ("A4", "A6", "A8", "A10", "A12", "A14")]

        def asp_z(x, y):
            ys = y - off
            r2 = x * x + ys * ys
            sq = np.sqrt(np.maximum(0.0, 1.0 - (1.0 + k) * C * C * r2))
            z = (C * r2) / (1.0 + sq)
            for idx, coef in enumerate(coefs, start=2):
                z += coef * r2 ** idx
            return z

        # Local bounds crop only the trajectory.  The preview always shows the
        # complete aspherical aperture defined by full_width/full_length.
        W = float(geom.get("full_width", 0.0))
        L = float(geom.get("full_length", 0.0))
        xs = np.linspace(-W / 2, W / 2, 36)
        ys = np.linspace(-L / 2, L / 2, 36)
        X, Y = np.meshgrid(xs, ys)
        Z = asp_z(X, Y)
        return self._grid_faces(X, Y, Z)

    def _sample_cylindrical_surface(self, geom):
        R = float(geom["R"])
        zc = float(geom.get("zc", 0.0))
        kcut = float(geom.get("k_cut", zc - R))
        axis_dir = geom.get("axis_dir", "Y")
        st = geom.get("surf_type", "C")
        amin = float(geom["axis_min"])
        amax = float(geom["axis_max"])
        thickness = float(geom.get(
            "wall_thickness", CYLINDRICAL_WALL_THICKNESS_MM))
        inner_R = R - thickness
        delta_z = kcut - zc
        outer_dmax = np.sqrt(max(0.0, R * R - delta_z * delta_z))
        sign = 1.0 if st == "C" else -1.0
        z0 = kcut if st == "C" else zc - R
        axis = np.linspace(amin, amax, 36)

        if st == "V" and inner_R > 0.0 and abs(delta_z) < inner_R:
            # Match the concave-sphere preview: use exact OCC surfaces for the
            # two working walls.  A coarse faceted cylinder does not reliably
            # occlude a trajectory on the far/outside half of a thin shell.
            occ = self._occ_imports()
            if axis_dir == "Y":
                cylinder_axis = occ["gp_Ax3"](
                    occ["gp_Pnt"](0.0, amin, R),
                    occ["gp_Dir"](0.0, 1.0, 0.0),
                    occ["gp_Dir"](1.0, 0.0, 0.0))
            else:
                cylinder_axis = occ["gp_Ax3"](
                    occ["gp_Pnt"](amin, 0.0, R),
                    occ["gp_Dir"](1.0, 0.0, 0.0),
                    occ["gp_Dir"](0.0, -1.0, 0.0))

            length = amax - amin

            def cylinder_face(radius, reverse=False):
                edge_angle = float(np.arcsin(np.clip(
                    -delta_z / radius, -1.0, 1.0)))
                face = occ["BRepBuilderAPI_MakeFace"](
                    occ["gp_Cylinder"](cylinder_axis, radius),
                    edge_angle, np.pi - edge_angle, 0.0, length).Face()
                return face.Reversed() if reverse else face

            inner_dmax = np.sqrt(max(
                0.0, inner_R * inner_R - delta_z * delta_z))
            shapes = [
                cylinder_face(R),
                cylinder_face(inner_R, reverse=True),
            ]

            # Close both opening lips between the outer shell and inner face.
            opening_z = kcut - z0
            for side in (-1.0, 1.0):
                D = np.column_stack((
                    np.full_like(axis, side * outer_dmax),
                    np.full_like(axis, side * inner_dmax),
                ))
                A = np.column_stack((axis, axis))
                Z = np.full_like(D, opening_z)
                if axis_dir == "Y":
                    X, Y = D, A
                else:
                    X, Y = A, D
                shapes.extend(self._grid_faces(X, Y, Z))

            # Close the two longitudinal ends of the 0.5 mm shell.
            u = np.linspace(-1.0, 1.0, 96)
            d_outer = outer_dmax * u
            d_inner = inner_dmax * u
            z_outer = (zc - np.sqrt(np.maximum(
                0.0, R * R - d_outer * d_outer)) - z0)
            z_inner = (zc - np.sqrt(np.maximum(
                0.0, inner_R * inner_R - d_inner * d_inner)) - z0)
            D = np.vstack((d_outer, d_inner))
            Z = np.vstack((z_outer, z_inner))
            for axis_value in (amin, amax):
                A = np.full_like(D, axis_value)
                if axis_dir == "Y":
                    X, Y = D, A
                else:
                    X, Y = A, D
                shapes.extend(self._grid_faces(X, Y, Z))
            return shapes

        def cylinder_grid(radius, dmax):
            d = np.linspace(-dmax, dmax, 36)
            D, A = np.meshgrid(d, axis)
            Z = (zc + sign * np.sqrt(
                np.maximum(0.0, radius * radius - D * D)) - z0)
            if axis_dir == "Y":
                return D, A, Z
            return A, D, Z

        shapes = self._grid_faces(*cylinder_grid(R, outer_dmax))
        if inner_R <= 0.0 or abs(delta_z) >= inner_R:
            return shapes

        inner_dmax = np.sqrt(max(
            0.0, inner_R * inner_R - delta_z * delta_z))
        shapes.extend(self._grid_faces(*cylinder_grid(inner_R, inner_dmax)))

        # Close both opening lips between the outer shell and inner work face.
        opening_z = kcut - z0
        for side in (-1.0, 1.0):
            D = np.column_stack((
                np.full_like(axis, side * outer_dmax),
                np.full_like(axis, side * inner_dmax),
            ))
            A = np.column_stack((axis, axis))
            Z = np.full_like(D, opening_z)
            if axis_dir == "Y":
                X, Y = D, A
            else:
                X, Y = A, D
            shapes.extend(self._grid_faces(X, Y, Z))

        # Close the two longitudinal ends so the preview is a 0.5 mm shell.
        u = np.linspace(-1.0, 1.0, 36)
        d_outer = outer_dmax * u
        d_inner = inner_dmax * u
        z_outer = (zc + sign * np.sqrt(np.maximum(
            0.0, R * R - d_outer * d_outer)) - z0)
        z_inner = (zc + sign * np.sqrt(np.maximum(
            0.0, inner_R * inner_R - d_inner * d_inner)) - z0)
        D = np.vstack((d_outer, d_inner))
        Z = np.vstack((z_outer, z_inner))
        for axis_value in (amin, amax):
            A = np.full_like(D, axis_value)
            if axis_dir == "Y":
                X, Y = D, A
            else:
                X, Y = A, D
            shapes.extend(self._grid_faces(X, Y, Z))
        return shapes

    def _sample_conical_surface(self, geom):
        ctype = int(geom.get("cone_type", 1)); alpha = np.radians(float(geom["alpha_deg"]))
        H = float(geom["H"]); tan_a = np.tan(alpha); R_base = H * tan_a
        u = np.linspace(0, 2 * np.pi, 49); v = np.linspace(max(R_base / 28.0, 1e-6), R_base, 28)
        U, V = np.meshgrid(u, v)
        X = V * np.cos(U); Y = V * np.sin(U)
        Z = H - V / tan_a if ctype == 1 else V / tan_a
        return self._grid_faces(X, Y, Z)

class ControlPanel(TrajectoryPlanningMixin, DwellTimeMixin, QStackedWidget):
    """
    仿照师兄软件：QDockWidget 里放 QStackedWidget，
    每个功能对应一个 page，Ribbon 按钮切换页面。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main = parent          # MainWindow 引用，用于访问 preview/statusbar
        self._surface_control_pairs = []
        self._init_dwell_state()

        # 记录各页面索引
        self.idx_blank   = self.count()
        self.addWidget(self._build_blank_page())    # page 0：空白（初始状态）

        self.idx_license = self.count()
        self.addWidget(self._build_license_page())

        # 驻留时间：由 appCNCFinishingV6.mlapp 的六个工作页迁移而来。
        self.idx_dwell_initial = self.count()
        self.addWidget(self._build_dwell_initial_page())
        self.idx_dwell_model = self.count()
        self.addWidget(self._build_dwell_model_page())
        self.idx_dwell_surface = self.count()
        self.addWidget(self._build_dwell_surface_page())
        self.idx_dwell_spot = self.count()
        self.addWidget(self._build_dwell_spot_page())
        self.idx_dwell_solve = self.count()
        self.addWidget(self._build_dwell_solve_page())
        self.idx_dwell_cnc = self.count()
        self.addWidget(self._build_dwell_cnc_page())

        # 曲面轨迹：统一入口页（下拉选择 + 子页面）
        self.idx_surface = self.count()
        self.addWidget(self._build_surface_selector_page())

        # 当前缓存的轨迹点和参数
        self._points = []
        self._params = {}
        self._last_is_surface = False

        # 驻留时间：切换页面时同步左侧顶部介绍横幅（横幅文本由各驻留页面自身携带）
        self.currentChanged.connect(self._sync_dwell_banner)
        self._sync_dwell_banner(self.currentIndex())

    def _sync_dwell_banner(self, index):
        """驻留时间页面在左侧顶部显示介绍横条，其余页面隐藏。"""
        widget = self.widget(index)
        text = getattr(widget, "dwell_banner_text", None)
        if text:
            self._main.preview.set_dwell_banner(text)
        else:
            self._main.preview.clear_dwell_banner()
        # 面形/抛光斑/求解页：已有数据时切回二维热力图并显示第二横条，其余情况隐藏。
        if (index == self.idx_dwell_surface and
                self._dwell_state.get("surface_raw") is not None and
                self._dwell_state.get("surface") is not None):
            self._dwell_refresh_surface_preview()
        elif (index == self.idx_dwell_spot and
                self._dwell_state.get("spot_before") is not None):
            self._dwell_refresh_spot_preview()
        elif (index == self.idx_dwell_solve and
                self._dwell_state.get("solution") is not None):
            self._dwell_refresh_solve_preview()
        else:
            self._main.preview.hide_dwell_sub_bar()

    # ── 共用：保存 TXT ──────────────────────────────────────────────
    def _build_blank_page(self):
        page = QWidget()
        page.setAutoFillBackground(True)
        page.setStyleSheet("background-color: #dfe9f5;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        blank = QLabel("")
        blank.setStyleSheet("background-color: #dfe9f5;")
        blank.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(blank)
        return page

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
    # 驻留时间页面（由 appCNCFinishingV6.mlapp 的六个 Tab 迁移）
    # ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光学曲面抛光工艺数据库")
        self.resize(1280, 800)
        self.setWindowIcon(get_icon("icon"))
        self.setStyleSheet(get_stylesheet("main"))

        self._build_ui()
        self._build_ribbon()

    def closeEvent(self, event):
        try:
            self.preview.close_occ()
        except Exception:
            pass
        super().closeEvent(event)

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
        self.status_message = QLabel("就绪")
        self.status_message.setStyleSheet(
            "QLabel {"
            "color: #10243f;"
            "font-family: Microsoft YaHei, SimHei, Arial;"
            "font-size: 12px;"
            "padding-left: 2px;"
            "}"
        )
        self.statusbar.addWidget(self.status_message, 1)
        self.statusbar.clearMessage()

    def set_status(self, text):
        self.statusbar.clearMessage()
        self.status_message.setText(text)

    # ── Ribbon 工具栏（完全仿照 ShowGui.py 的 init_ribbon 写法）────
    def _build_ribbon(self):
        self._ribbon = RibbonWidget(self)
        self.addToolBar(self._ribbon)

        # ── Tab：轨迹规划 ──────────────────────────────────────────
        tab_traj = self._ribbon.add_ribbon_tab("轨迹规划")

        pane_model = tab_traj.add_ribbon_pane("模型")
        act_import = self._make_action("导入模型", "import_file",
                                       "导入 STL 模型到曲面显示区",
                                       self._import_stl_model)
        pane_model.add_ribbon_widget(RibbonButton(self, act_import, True))

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

        # ── Tab：驻留时间 ────────────────────────────────────────────
        # 按 appCNCFinishingV6.mlapp 的工作流排列，位于轨迹规划与授权之间。
        tab_dwell = self._ribbon.add_ribbon_tab("驻留时间")
        dwell_actions = [
            ("初始设置", "gear", "设置抛光工具、工艺阶段和计算网格",
             self._show_dwell_initial),
            ("建模", "3d", "设置待加工曲面与口径参数", self._show_dwell_model),
            ("面形数据", "data", "导入和处理面形数据", self._show_dwell_surface),
            ("抛光斑", "central", "导入抛光斑并设置去除函数", self._show_dwell_spot),
            ("驻留时间求解", "gear", "仅使用最小二乘法求解驻留时间",
             self._show_dwell_solve),
            ("CNC程序生成", "NC", "生成和保存 CNC 程序", self._show_dwell_cnc),
        ]
        for caption, icon, tip, slot in dwell_actions:
            pane = tab_dwell.add_ribbon_pane(caption)
            action = self._make_action(caption, icon, tip, slot)
            pane.add_ribbon_widget(RibbonButton(self, action, True))
        tab_dwell.add_spacer()

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

    def _show_dwell_initial(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_initial)

    def _show_dwell_model(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_model)

    def _show_dwell_surface(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_surface)

    def _show_dwell_spot(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_spot)

    def _show_dwell_solve(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_solve)

    def _show_dwell_cnc(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_cnc)

    def _import_stl_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 STL 模型", "", "STL 模型 (*.stl *.STL)")
        if not path:
            return
        try:
            self.imported_stl_path = path
            self.imported_stl_shape = self.preview.import_stl_to_shape(path)
            self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_blank)
            name = os.path.basename(path)
            self.statusbar.showMessage(f"STL 模型导入成功：{name}")
            self.set_status(f"STL 模型导入成功：{name}")
            self.terminal_output.appendPlainText(f"[模型导入] STL: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

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
    app.aboutToQuit.connect(win.preview.close_occ)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
