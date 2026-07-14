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
from function.planar_trajectory import (
    generate_planar_raster, generate_planar_spiral, save_trajectory_txt
)
from function.surface_trajectory import (
    generate_aspherical, generate_spherical, SPHERICAL_WALL_THICKNESS_MM,
    generate_cylindrical, CYLINDRICAL_WALL_THICKNESS_MM, generate_conical,
    save_surface_trajectory_txt
)
from function.license_manager   import get_hardware_id, activate, verify_license



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

        self._init_occ_widgets()

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
        for host, label in self._label_hosts:
            if host is None or label is None:
                continue
            label.setGeometry(0, 0, max(1, host.width()), 30)
            label.raise_()

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

    def import_stl_to_shape(self, path):
        self._stack.setCurrentIndex(1)
        self._apply_surface_view_mode()
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

class ControlPanel(QStackedWidget):
    """
    仿照师兄软件：QDockWidget 里放 QStackedWidget，
    每个功能对应一个 page，Ribbon 按钮切换页面。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main = parent          # MainWindow 引用，用于访问 preview/statusbar
        self._surface_control_pairs = []

        # 记录各页面索引
        self.idx_blank   = self.count()
        self.addWidget(self._build_blank_page())    # page 0：空白（初始状态）

        self.idx_license = self.count()
        self.addWidget(self._build_license_page())

        # 曲面轨迹：统一入口页（下拉选择 + 子页面）
        self.idx_surface = self.count()
        self.addWidget(self._build_surface_selector_page())

        # 当前缓存的轨迹点和参数
        self._points = []
        self._params = {}
        self._last_is_surface = False

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
            self._main.set_status(f"已保存至 {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self._main, "保存失败", str(e))

    def _finish(self, points, params, save_btn, info_lbl, tname, is_surface=False):
        self._points = points
        self._params = params
        self._last_is_surface = is_surface
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
        self._main.set_status(f"{tname}生成完成，共 {len(points)} 个轨迹点")
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
        默认值：间距=2 mm，步长=1 mm（固定）。
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
        edt_spacing, row_sp  = lineedit_input("间距 (mm)：",  "2")
        edt_step,    row_st  = lineedit_input("步长 (mm)：",  "1")
        g.addLayout(row_sp)
        g.addLayout(row_st)

        # 保留 pitch/arc 对象供 _read_traj 使用，不显示
        edt_pitch = QLineEdit("2");   edt_pitch.setVisible(False)
        edt_arc   = QLineEdit("1");  edt_arc.setVisible(False)

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
        self.pl_grp_raster = QGroupBox("栅形参数")
        g3 = QVBoxLayout(self.pl_grp_raster)
        self.pl_cmb_dir = QComboBox()
        self.pl_cmb_dir.addItems(["平行于 X 轴（沿 Y 方向推进）",
                                   "平行于 Y 轴（沿 X 方向推进）"])
        combox_input(g3, "扫描方向：", self.pl_cmb_dir)
        self.pl_edt_step,    row_st  = lineedit_input("点间步长 (mm)：", "1.0")
        self.pl_edt_spacing, row_sp  = lineedit_input("线间距 (mm)：",   "5.0")
        g3.addLayout(row_st); g3.addLayout(row_sp)
        self.pl_cmb_cover = QComboBox()
        self.pl_cmb_cover.addItems(["全部覆盖", "局部子区域"])
        combox_input(g3, "覆盖范围：", self.pl_cmb_cover)
        # 子区域参数（全部覆盖时隐藏）
        self.pl_lbl_sub = QLabel("── 子区域参数 ──")
        g3.addWidget(self.pl_lbl_sub)
        self.pl_edt_sx0, row_sx0 = lineedit_input("左下角 X₀ (mm)：", "0")
        self.pl_edt_sy0, row_sy0 = lineedit_input("左下角 Y₀ (mm)：", "0")
        self.pl_edt_sC,  row_sC  = lineedit_input("区域长 C (mm)：",  "10")
        self.pl_edt_sD,  row_sD  = lineedit_input("区域宽 D (mm)：",  "10")
        self.pl_wrap_rsub = QWidget()
        wrs = QVBoxLayout(self.pl_wrap_rsub)
        wrs.setContentsMargins(0, 0, 0, 0); wrs.setSpacing(2)
        for row in [row_sx0, row_sy0, row_sC, row_sD]:
            wrs.addLayout(row)
        g3.addWidget(self.pl_wrap_rsub)
        layout.addWidget(self.pl_grp_raster)

        # ④ 螺旋线参数
        self.pl_grp_spiral = QGroupBox("螺旋线参数")
        g4 = QVBoxLayout(self.pl_grp_spiral)
        self.pl_edt_pitch,   row_pit = lineedit_input("螺距（每圈半径增量，mm）：", "5.0")
        self.pl_edt_arcstep, row_as  = lineedit_input("弧长步长（点间距，mm）：",   "1.0")
        g4.addLayout(row_pit); g4.addLayout(row_as)
        self.pl_cmb_spiral_cover = QComboBox()
        self.pl_cmb_spiral_cover.addItems(["圆形覆盖范围", "矩形覆盖范围"])
        combox_input(g4, "覆盖范围：", self.pl_cmb_spiral_cover)
        # 圆形参数
        self.pl_edt_Rmax, row_rm = lineedit_input("最大半径 R_max (mm)：", "50")
        g4.addLayout(row_rm)
        # 矩形参数
        self.pl_lbl_srect = QLabel("── 矩形范围参数 ──")
        g4.addWidget(self.pl_lbl_srect)
        self.pl_edt_sxmin, row_sxn = lineedit_input("X_min (mm)：",  "0")
        self.pl_edt_symin, row_syn = lineedit_input("Y_min (mm)：",  "0")
        self.pl_edt_sxmax, row_sxx = lineedit_input("X_max (mm)：", "100")
        self.pl_edt_symax, row_syx = lineedit_input("Y_max (mm)：", "100")
        self.pl_wrap_ssub = QWidget()
        wss = QVBoxLayout(self.pl_wrap_ssub)
        wss.setContentsMargins(0, 0, 0, 0); wss.setSpacing(2)
        for row in [row_sxn, row_syn, row_sxx, row_syx]:
            wss.addLayout(row)
        g4.addWidget(self.pl_wrap_ssub)
        layout.addWidget(self.pl_grp_spiral)

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
        self.pl_grp_raster.setVisible(is_raster)
        self.pl_grp_spiral.setVisible(not is_raster)
        if is_raster:
            self._pl_cover_changed()
        else:
            self._pl_spiral_cover_changed()

    def _pl_cover_changed(self):
        sub = (self.pl_cmb_cover.currentIndex() == 1)
        self.pl_lbl_sub.setVisible(sub)
        self.pl_wrap_rsub.setVisible(sub)

    def _pl_spiral_cover_changed(self):
        is_circ = (self.pl_cmb_spiral_cover.currentIndex() == 0)
        self.pl_edt_Rmax.setVisible(is_circ)
        self.pl_lbl_srect.setVisible(not is_circ)
        self.pl_wrap_ssub.setVisible(not is_circ)

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

        view_grp = QGroupBox("3D显示模式")
        view_layout = QVBoxLayout(view_grp)
        self.surface_view_cmb = QComboBox()
        self.surface_view_cmb.addItems([
            "左侧显示曲面，右侧显示轨迹",
            "轨迹覆盖在曲面上",
        ])
        self.surface_view_cmb.setFixedHeight(30)
        view_layout.addWidget(self.surface_view_cmb)
        outer_layout.addWidget(view_grp)

        self.surf_cmb.currentIndexChanged.connect(
            lambda idx: self.surf_stack.setCurrentIndex(idx))
        self.surface_view_cmb.currentIndexChanged.connect(self._surface_view_mode_changed)

        return outer

    def _surface_view_mode_changed(self, idx):
        mode = "overlay" if idx == 1 else "split"
        if self._main is not None and getattr(self._main, "preview", None) is not None:
            self._main.preview.set_surface_view_mode(mode)
            if getattr(self, "_last_is_surface", False) and self._points:
                self._main.preview.plot_surface(self._points, self._params)

    def _build_surface_control_group(self):
        grp = QGroupBox("曲面轨迹总控制")
        layout = QVBoxLayout(grp)
        edt_spacing, row_sp = lineedit_input("间距 (mm)：", "2")
        edt_step, row_st = lineedit_input("步长 (mm)：", "1")
        layout.addLayout(row_sp)
        layout.addLayout(row_st)
        self._surface_control_pairs.append((edt_step, edt_spacing))
        edt_step.textChanged.connect(
            lambda txt, s=edt_step: self._sync_surface_control_value("step", txt, s))
        edt_spacing.textChanged.connect(
            lambda txt, s=edt_spacing: self._sync_surface_control_value("spacing", txt, s))
        return grp, edt_step, edt_spacing

    def _sync_surface_control_value(self, field, text, source):
        idx = 0 if field == "step" else 1
        for pair in self._surface_control_pairs:
            edit = pair[idx]
            if edit is source or edit.text() == text:
                continue
            blocker = QtCore.QSignalBlocker(edit)
            edit.setText(text)
            del blocker

    def _read_surface_step_spacing(self, edt_step, edt_spacing):
        def f(e, n):
            try: return float(e.text())
            except: raise ValueError(f"参数「{n}」输入无效")
        step_len = f(edt_step, "步长")
        line_spacing = f(edt_spacing, "间距")
        return step_len, line_spacing, line_spacing, step_len

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
        self.asp_edt_W, row_W = lineedit_input("X方向总宽度 (mm)：", "100")
        self.asp_edt_L, row_L = lineedit_input("Y方向总长度 (mm)：", "100")
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
        self.asp_edt_xmin, row_xn = lineedit_input("X_min (mm)：", "-50")
        self.asp_edt_xmax, row_xx = lineedit_input("X_max (mm)：",  "50")
        self.asp_edt_ymin, row_yn = lineedit_input("Y_min (mm)：", "-50")
        self.asp_edt_ymax, row_yx = lineedit_input("Y_max (mm)：",  "50")
        self.asp_wrap_rect = QWidget()
        wr = QVBoxLayout(self.asp_wrap_rect)
        wr.setContentsMargins(0, 0, 0, 0); wr.setSpacing(2)
        for row in [row_xn, row_xx, row_yn, row_yx]:
            wr.addLayout(row)
        g4.addWidget(self.asp_wrap_rect)

        self.asp_lbl_circ = QLabel("── 圆形边界参数 ──")
        g4.addWidget(self.asp_lbl_circ)
        self.asp_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "50")
        self.asp_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.asp_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        self.asp_wrap_circ = QWidget()
        wc = QVBoxLayout(self.asp_wrap_circ)
        wc.setContentsMargins(0, 0, 0, 0); wc.setSpacing(2)
        for row in [row_cR, row_cxc, row_cyc]:
            wc.addLayout(row)
        g4.addWidget(self.asp_wrap_circ)
        layout.addWidget(grp4)

        grp5 = QGroupBox("轨迹参数")
        g5 = QVBoxLayout(grp5)
        self.asp_cmb_traj = QComboBox()
        self.asp_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g5, "轨迹类型：", self.asp_cmb_traj)
        self.asp_cmb_dir = QComboBox()
        self.asp_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g5, "栅形方向：", self.asp_cmb_dir)
        self.asp_edt_spacing = QLineEdit("2")
        self.asp_edt_step = QLineEdit("1")
        self.asp_edt_pitch = QLineEdit("2")
        self.asp_edt_arcstep = QLineEdit("1")
        for hidden in [self.asp_edt_spacing, self.asp_edt_step,
                       self.asp_edt_pitch, self.asp_edt_arcstep]:
            hidden.setParent(self)
            hidden.setVisible(False)
        layout.addWidget(grp5)

        grp6 = QGroupBox("输出设置")
        g6 = QVBoxLayout(grp6)
        self.asp_edt_fname, row_fn = lineedit_input("文件名：", "aspherical_traj")
        g6.addLayout(row_fn)
        layout.addWidget(grp6)

        ctrl_grp, self.asp_ctrl_step, self.asp_ctrl_spacing = self._build_surface_control_group()
        layout.addWidget(ctrl_grp)

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
        self.asp_wrap_rect.setVisible(show_rect)
        self.asp_lbl_circ.setVisible(show_circ)
        self.asp_wrap_circ.setVisible(show_circ)

    def _asp_traj_changed(self):
        is_raster = (self.asp_cmb_traj.currentIndex() == 0)
        self.asp_cmb_dir.setVisible(is_raster)
        # 间距/步长始终显示，不随轨迹类型切换
        self.asp_edt_spacing.setVisible(False)
        self.asp_edt_step.setVisible(False)
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
            step_len, line_spacing, pitch, arc_step = self._read_surface_step_spacing(
                self.asp_ctrl_step, self.asp_ctrl_spacing)
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
        # 把非球面几何参数收进 geom（左图按此渲染实体曲面）
        geom = {"type": "aspherical",
                "R": R, "k": k, "A4": A4, "A6": A6, "A8": A8,
                "A10": A10, "A12": A12, "A14": A14,
                "offcenter": off, "bound_type": bound,
                "full_width": W, "full_length": L}
        if bound == 2:
            geom.update(rect_xmin=kwargs["rect_xmin"], rect_xmax=kwargs["rect_xmax"],
                        rect_ymin=kwargs["rect_ymin"], rect_ymax=kwargs["rect_ymax"])
        elif bound == 3:
            geom.update(circ_R=kwargs["circ_R"], circ_xc=kwargs["circ_xc"],
                        circ_yc=kwargs["circ_yc"])
        params = {"surface_name": "非球面", "traj_name": tname + "轨迹", "geom": geom}
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
        self.sph_edt_R,  row_R  = lineedit_input("球体半径 R (正数, mm)：", "100")
        self.sph_edt_zc, row_zc = lineedit_input("球心 Z 坐标 zc：", "0")
        self.sph_edt_h,  row_h  = lineedit_input("球冠高度 h (mm，0 < h ≤ 2R)：", "100")
        g1.addLayout(row_R); g1.addLayout(row_zc); g1.addLayout(row_h)
        self.sph_cmb_type = QComboBox()
        self.sph_cmb_type.addItems(["凸球面 (Convex)", "凹球面 (Concave)"])
        combox_input(g1, "表面类型：", self.sph_cmb_type)
        layout.addWidget(grp1)

        # —— 覆盖范围 ——
        grp_cv = QGroupBox("覆盖范围")
        gcv = QVBoxLayout(grp_cv)
        self.sph_cmb_cover = QComboBox()
        self.sph_cmb_cover.addItems([
            "全部覆盖（整个球冠投影圆）",
            "局部矩形区域",
            "局部圆形区域",
        ])
        combox_input(gcv, "覆盖类型：", self.sph_cmb_cover)

        self.sph_lbl_rect = QLabel("── 矩形区域参数 ──")
        gcv.addWidget(self.sph_lbl_rect)
        self.sph_edt_rxmin, row_rxn = lineedit_input("X_min (mm)：", "-50")
        self.sph_edt_rxmax, row_rxx = lineedit_input("X_max (mm)：",  "50")
        self.sph_edt_rymin, row_ryn = lineedit_input("Y_min (mm)：", "-50")
        self.sph_edt_rymax, row_ryx = lineedit_input("Y_max (mm)：",  "50")
        self.sph_wrap_rect = QWidget()
        wr = QVBoxLayout(self.sph_wrap_rect)
        wr.setContentsMargins(0, 0, 0, 0); wr.setSpacing(2)
        for row in [row_rxn, row_rxx, row_ryn, row_ryx]:
            wr.addLayout(row)
        gcv.addWidget(self.sph_wrap_rect)

        self.sph_lbl_circ = QLabel("── 圆形区域参数 ──")
        gcv.addWidget(self.sph_lbl_circ)
        self.sph_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "50")
        self.sph_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.sph_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        self.sph_wrap_circ = QWidget()
        wc = QVBoxLayout(self.sph_wrap_circ)
        wc.setContentsMargins(0, 0, 0, 0); wc.setSpacing(2)
        for row in [row_cR, row_cxc, row_cyc]:
            wc.addLayout(row)
        gcv.addWidget(self.sph_wrap_circ)
        layout.addWidget(grp_cv)

        grp2 = QGroupBox("轨迹参数")
        g2 = QVBoxLayout(grp2)
        self.sph_cmb_traj = QComboBox()
        self.sph_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g2, "轨迹类型：", self.sph_cmb_traj)
        self.sph_cmb_dir = QComboBox()
        self.sph_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g2, "栅形方向：", self.sph_cmb_dir)
        self.sph_edt_spacing = QLineEdit("2")
        self.sph_edt_step = QLineEdit("1")
        self.sph_edt_pitch = QLineEdit("2")
        self.sph_edt_arcstep = QLineEdit("1")
        for hidden in [self.sph_edt_spacing, self.sph_edt_step,
                       self.sph_edt_pitch, self.sph_edt_arcstep]:
            hidden.setParent(self)
            hidden.setVisible(False)
        layout.addWidget(grp2)

        grp3 = QGroupBox("输出设置")
        g3 = QVBoxLayout(grp3)
        self.sph_edt_fname, row_fn = lineedit_input("文件名：", "spherical_traj")
        g3.addLayout(row_fn)
        layout.addWidget(grp3)

        ctrl_grp, self.sph_ctrl_step, self.sph_ctrl_spacing = self._build_surface_control_group()
        layout.addWidget(ctrl_grp)

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
        self.sph_cmb_cover.currentIndexChanged.connect(self._sph_cover_changed)
        self.sph_btn_gen.clicked.connect(self._do_generate_spherical)
        self.sph_btn_save.clicked.connect(
            lambda: self._do_save("球面轨迹", self.sph_edt_fname.text(), is_surface=True))
        self._sph_traj_changed()
        self._sph_cover_changed()
        return scroll

    def _sph_traj_changed(self):
        is_raster = (self.sph_cmb_traj.currentIndex() == 0)
        self.sph_cmb_dir.setVisible(is_raster)
        self.sph_edt_spacing.setVisible(False)
        self.sph_edt_step.setVisible(False)
        self.sph_edt_pitch.setVisible(False)
        self.sph_edt_arcstep.setVisible(False)

    def _sph_cover_changed(self):
        idx = self.sph_cmb_cover.currentIndex()
        show_rect = (idx == 1)
        show_circ = (idx == 2)
        self.sph_lbl_rect.setVisible(show_rect)
        self.sph_wrap_rect.setVisible(show_rect)
        self.sph_lbl_circ.setVisible(show_circ)
        self.sph_wrap_circ.setVisible(show_circ)

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
            cover = self.sph_cmb_cover.currentIndex() + 1  # 1/2/3
            step_len, line_spacing, pitch, arc_step = self._read_surface_step_spacing(
                self.sph_ctrl_step, self.sph_ctrl_spacing)
            kwargs = dict(R=R, zc=zc, surf_type=surf, h=h,
                          traj_type=traj, direction=dire,
                          step_len=step_len, line_spacing=line_spacing,
                          pitch=pitch, arc_step=arc_step,
                          cover_type=cover,
                          wall_thickness=SPHERICAL_WALL_THICKNESS_MM)
            if cover == 2:
                kwargs.update(rect_xmin=f(self.sph_edt_rxmin, "X_min"),
                              rect_xmax=f(self.sph_edt_rxmax, "X_max"),
                              rect_ymin=f(self.sph_edt_rymin, "Y_min"),
                              rect_ymax=f(self.sph_edt_rymax, "Y_max"))
            elif cover == 3:
                kwargs.update(circ_R=f(self.sph_edt_cR, "圆形半径"),
                              circ_xc=f(self.sph_edt_cxc, "圆心X"),
                              circ_yc=f(self.sph_edt_cyc, "圆心Y"))
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_spherical(**kwargs)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        surf_cn = "凸球面" if surf == "convex" else "凹球面"
        geom = {"type": "spherical",
                "R": R, "zc": zc, "h": h, "surf_type": surf,
                "cover_type": cover,
                "wall_thickness": SPHERICAL_WALL_THICKNESS_MM}
        if cover == 2:
            geom.update(rect_xmin=kwargs["rect_xmin"], rect_xmax=kwargs["rect_xmax"],
                        rect_ymin=kwargs["rect_ymin"], rect_ymax=kwargs["rect_ymax"])
        elif cover == 3:
            geom.update(circ_R=kwargs["circ_R"],
                        circ_xc=kwargs["circ_xc"], circ_yc=kwargs["circ_yc"])
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹",
                  "traj_type": traj, "direction": dire,
                  "step_len": step_len, "line_spacing": line_spacing,
                  "pitch": pitch, "arc_step": arc_step,
                  "geom": geom}
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
        self.cyl_edt_R,    row_R   = lineedit_input("圆柱截面半径 R (mm)：", "100")
        self.cyl_edt_zc,   row_zc  = lineedit_input("圆柱截面圆心 Z：", "0")
        self.cyl_edt_k,    row_k   = lineedit_input("切割平面高度 k：", "0")
        self.cyl_edt_amin, row_an  = lineedit_input("轴线方向起点 (mm)：", "-50")
        self.cyl_edt_amax, row_ax  = lineedit_input("轴线方向终点 (mm)：",  "50")
        for row in [row_R, row_zc, row_k, row_an, row_ax]:
            g1.addLayout(row)
        layout.addWidget(grp1)

        grp_cv = QGroupBox("覆盖范围")
        gcv = QVBoxLayout(grp_cv)
        self.cyl_cmb_cover = QComboBox()
        self.cyl_cmb_cover.addItems([
            "全部覆盖（完整柱面投影区域）",
            "局部矩形区域",
            "局部圆形区域",
        ])
        combox_input(gcv, "覆盖类型：", self.cyl_cmb_cover)

        self.cyl_lbl_rect = QLabel("── 矩形区域参数 ──")
        gcv.addWidget(self.cyl_lbl_rect)
        self.cyl_edt_rxmin, row_rxn = lineedit_input("X_min (mm)：", "-50")
        self.cyl_edt_rxmax, row_rxx = lineedit_input("X_max (mm)：",  "50")
        self.cyl_edt_rymin, row_ryn = lineedit_input("Y_min (mm)：", "-50")
        self.cyl_edt_rymax, row_ryx = lineedit_input("Y_max (mm)：",  "50")
        self.cyl_wrap_rect = QWidget()
        wr = QVBoxLayout(self.cyl_wrap_rect)
        wr.setContentsMargins(0, 0, 0, 0); wr.setSpacing(2)
        for row in [row_rxn, row_rxx, row_ryn, row_ryx]:
            wr.addLayout(row)
        gcv.addWidget(self.cyl_wrap_rect)

        self.cyl_lbl_circ = QLabel("── 圆形区域参数 ──")
        gcv.addWidget(self.cyl_lbl_circ)
        self.cyl_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "50")
        self.cyl_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.cyl_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        self.cyl_wrap_circ = QWidget()
        wc = QVBoxLayout(self.cyl_wrap_circ)
        wc.setContentsMargins(0, 0, 0, 0); wc.setSpacing(2)
        for row in [row_cR, row_cxc, row_cyc]:
            wc.addLayout(row)
        gcv.addWidget(self.cyl_wrap_circ)
        layout.addWidget(grp_cv)

        grp3 = QGroupBox("轨迹参数")
        g3 = QVBoxLayout(grp3)
        self.cyl_cmb_traj = QComboBox()
        self.cyl_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g3, "轨迹类型：", self.cyl_cmb_traj)
        self.cyl_cmb_dir = QComboBox()
        self.cyl_cmb_dir.addItems(["X方向步进", "Y方向步进"])
        combox_input(g3, "栅形方向：", self.cyl_cmb_dir)
        self.cyl_edt_spacing = QLineEdit("2")
        self.cyl_edt_step = QLineEdit("1")
        self.cyl_edt_pitch = QLineEdit("2")
        self.cyl_edt_arcstep = QLineEdit("1")
        for hidden in [self.cyl_edt_spacing, self.cyl_edt_step,
                       self.cyl_edt_pitch, self.cyl_edt_arcstep]:
            hidden.setParent(self)
            hidden.setVisible(False)
        layout.addWidget(grp3)

        grp4 = QGroupBox("输出设置")
        g4 = QVBoxLayout(grp4)
        self.cyl_edt_fname, row_fn = lineedit_input("文件名：", "cylindrical_traj")
        g4.addLayout(row_fn)
        layout.addWidget(grp4)

        ctrl_grp, self.cyl_ctrl_step, self.cyl_ctrl_spacing = self._build_surface_control_group()
        layout.addWidget(ctrl_grp)

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
        self.cyl_cmb_cover.currentIndexChanged.connect(self._cyl_cover_changed)
        self.cyl_btn_gen.clicked.connect(self._do_generate_cylindrical)
        self.cyl_btn_save.clicked.connect(
            lambda: self._do_save("柱面轨迹", self.cyl_edt_fname.text(), is_surface=True))
        self._cyl_traj_changed()
        self._cyl_cover_changed()
        return scroll

    def _cyl_traj_changed(self):
        is_raster = (self.cyl_cmb_traj.currentIndex() == 0)
        self.cyl_cmb_dir.setVisible(is_raster)
        self.cyl_edt_spacing.setVisible(False)
        self.cyl_edt_step.setVisible(False)
        self.cyl_edt_pitch.setVisible(False)
        self.cyl_edt_arcstep.setVisible(False)

    def _cyl_cover_changed(self):
        idx = self.cyl_cmb_cover.currentIndex()
        show_rect = (idx == 1)
        show_circ = (idx == 2)
        self.cyl_lbl_rect.setVisible(show_rect)
        self.cyl_wrap_rect.setVisible(show_rect)
        self.cyl_lbl_circ.setVisible(show_circ)
        self.cyl_wrap_circ.setVisible(show_circ)

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
            cover = self.cyl_cmb_cover.currentIndex() + 1
            traj  = "G" if self.cyl_cmb_traj.currentIndex() == 0 else "S"
            dire  = "X" if self.cyl_cmb_dir.currentIndex()  == 0 else "Y"
            step_len, line_spacing, pitch, arc_step = self._read_surface_step_spacing(
                self.cyl_ctrl_step, self.cyl_ctrl_spacing)
            coverage = {"cover_type": cover}
            if cover == 2:
                coverage.update(
                    rect_xmin=f(self.cyl_edt_rxmin, "X_min"),
                    rect_xmax=f(self.cyl_edt_rxmax, "X_max"),
                    rect_ymin=f(self.cyl_edt_rymin, "Y_min"),
                    rect_ymax=f(self.cyl_edt_rymax, "Y_max"))
            elif cover == 3:
                coverage.update(
                    circ_R=f(self.cyl_edt_cR, "圆形半径"),
                    circ_xc=f(self.cyl_edt_cxc, "圆心X"),
                    circ_yc=f(self.cyl_edt_cyc, "圆心Y"))
        except ValueError as e:
            QMessageBox.warning(self._main, "参数错误", str(e)); return
        try:
            pts = generate_cylindrical(R=R, zc=zc, k_cut=k_cut,
                                       axis_dir=axis, surf_type=surf,
                                       axis_min=amin, axis_max=amax,
                                       traj_type=traj, direction=dire,
                                       step_len=step_len, line_spacing=line_spacing,
                                       pitch=pitch, arc_step=arc_step,
                                       wall_thickness=CYLINDRICAL_WALL_THICKNESS_MM,
                                       **coverage)
        except ValueError as e:
            QMessageBox.warning(self._main, "生成失败", str(e)); return
        if not pts:
            QMessageBox.warning(self._main, "警告", "未生成任何轨迹点"); return
        tname = "栅形" if traj == "G" else "螺旋线"
        surf_cn = "凸柱面" if surf == "C" else "凹柱面"
        geom = {"type": "cylindrical",
                "R": R, "zc": zc, "k_cut": k_cut,
                "axis_dir": axis, "surf_type": surf,
                "axis_min": amin, "axis_max": amax,
                "cover_type": cover,
                "wall_thickness": CYLINDRICAL_WALL_THICKNESS_MM}
        geom.update(coverage)
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹",
                  "traj_type": traj, "direction": dire,
                  "step_len": step_len, "line_spacing": line_spacing,
                  "pitch": pitch, "arc_step": arc_step,
                  "geom": geom}
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
        self.con_edt_H,     row_H  = lineedit_input("高度 H (正数, mm)：", "100")
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
        self.con_edt_rxmin, row_rxn = lineedit_input("X_min (mm)：", "-50")
        self.con_edt_rxmax, row_rxx = lineedit_input("X_max (mm)：",  "50")
        self.con_edt_rymin, row_ryn = lineedit_input("Y_min (mm)：", "-50")
        self.con_edt_rymax, row_ryx = lineedit_input("Y_max (mm)：",  "50")
        self.con_wrap_rect = QWidget()
        wr = QVBoxLayout(self.con_wrap_rect)
        wr.setContentsMargins(0, 0, 0, 0); wr.setSpacing(2)
        for row in [row_rxn, row_rxx, row_ryn, row_ryx]:
            wr.addLayout(row)
        g2.addWidget(self.con_wrap_rect)

        self.con_lbl_circ = QLabel("── 圆形区域参数 ──")
        g2.addWidget(self.con_lbl_circ)
        self.con_edt_cR,  row_cR  = lineedit_input("圆形半径 (mm)：", "50")
        self.con_edt_cxc, row_cxc = lineedit_input("圆心 X (mm)：",   "0")
        self.con_edt_cyc, row_cyc = lineedit_input("圆心 Y (mm)：",   "0")
        self.con_wrap_circ = QWidget()
        wc = QVBoxLayout(self.con_wrap_circ)
        wc.setContentsMargins(0, 0, 0, 0); wc.setSpacing(2)
        for row in [row_cR, row_cxc, row_cyc]:
            wc.addLayout(row)
        g2.addWidget(self.con_wrap_circ)
        layout.addWidget(grp2)

        grp3 = QGroupBox("轨迹参数")
        g3 = QVBoxLayout(grp3)
        self.con_cmb_traj = QComboBox()
        self.con_cmb_traj.addItems(["栅形轨迹 (Raster)", "螺旋线轨迹 (Spiral)"])
        combox_input(g3, "轨迹类型：", self.con_cmb_traj)
        self.con_cmb_dir = QComboBox()
        self.con_cmb_dir.addItems(["X方向 (平行X轴)", "Y方向 (平行Y轴)"])
        combox_input(g3, "栅形方向：", self.con_cmb_dir)
        self.con_edt_spacing = QLineEdit("2")
        self.con_edt_step = QLineEdit("1")
        self.con_edt_pitch = QLineEdit("2")
        self.con_edt_arcstep = QLineEdit("1")
        for hidden in [self.con_edt_spacing, self.con_edt_step,
                       self.con_edt_pitch, self.con_edt_arcstep]:
            hidden.setParent(self)
            hidden.setVisible(False)
        layout.addWidget(grp3)

        grp4 = QGroupBox("输出设置")
        g4 = QVBoxLayout(grp4)
        self.con_edt_fname, row_fn = lineedit_input("文件名：", "conical_traj")
        g4.addLayout(row_fn)
        layout.addWidget(grp4)

        ctrl_grp, self.con_ctrl_step, self.con_ctrl_spacing = self._build_surface_control_group()
        layout.addWidget(ctrl_grp)

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
        self.con_wrap_rect.setVisible(show_rect)
        self.con_lbl_circ.setVisible(show_circ)
        self.con_wrap_circ.setVisible(show_circ)

    def _con_traj_changed(self):
        is_raster = (self.con_cmb_traj.currentIndex() == 0)
        self.con_cmb_dir.setVisible(is_raster)
        self.con_edt_spacing.setVisible(False)
        self.con_edt_step.setVisible(False)
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
            step_len, line_spacing, pitch, arc_step = self._read_surface_step_spacing(
                self.con_ctrl_step, self.con_ctrl_spacing)
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
        geom = {"type": "conical", "cone_type": ctype, "alpha_deg": alpha,
                "H": H, "cover_type": cover}
        if cover == 2:
            geom.update(rect_xmin=kwargs["rect_xmin"], rect_xmax=kwargs["rect_xmax"],
                        rect_ymin=kwargs["rect_ymin"], rect_ymax=kwargs["rect_ymax"])
        elif cover == 3:
            geom.update(circ_R=kwargs["circ_R"], circ_xc=kwargs["circ_xc"],
                        circ_yc=kwargs["circ_yc"])
        params = {"surface_name": surf_cn, "traj_name": tname + "轨迹", "geom": geom}
        self._finish(pts, params, self.con_btn_save, self.con_info_lbl,
                     f"{surf_cn}{tname}轨迹", is_surface=True)


# ════════════════════════════════════════════════════════════════════
# 主窗口
# ════════════════════════════════════════════════════════════════════
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
