# -*- coding: utf-8 -*-
"""Trajectory-planning feature pages, generation, preview and export wiring."""
import os

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)

from function.planar_trajectory import (
    generate_planar_raster, generate_planar_spiral, save_trajectory_txt,
)
from function.surface_trajectory import (
    CYLINDRICAL_WALL_THICKNESS_MM, SPHERICAL_WALL_THICKNESS_MM,
    generate_aspherical, generate_conical, generate_cylindrical,
    generate_spherical, save_surface_trajectory_txt,
)


def lineedit_input(label_text, default_value=""):
    label = QLabel(label_text)
    line_edit = QLineEdit(str(default_value))
    layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(line_edit)
    return line_edit, layout


def combox_input(layout, label_text, widget):
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    row.addWidget(widget)
    layout.addLayout(row)


def divider():
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    frame.setFrameShadow(QFrame.Sunken)
    return frame


class TrajectoryPlanningMixin:
    """Business/UI mixin for the top-level trajectory-planning feature."""

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
