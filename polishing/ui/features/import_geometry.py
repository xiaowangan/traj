"""Migrated model page behavior; existing layout preserved."""
from __future__ import annotations
import json
import math
import os
import numpy as np
from PyQt5.QtWidgets import QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from polishing.ui.widgets.inputs import lineedit_input, combox_input
from polishing.ui.widgets.stage_ui import collapsible
from polishing.workflow.engine import DwellTimeEngine

class GeometryImportMixin:
    def _build_dwell_model_page(self):
        scroll, layout = self._dwell_scroll_page(
            "期望曲面建模", "建立工件加工完成后希望达到的曲面，并计算 XYZ、口径和法向。")

        # ── 镜面口径选择：口径决定镜子的轮廓与尺寸，选类型后只显示对应参数 ──
        aperture_group = QGroupBox("镜面口径选择"); aperture_form = QVBoxLayout(aperture_group)
        self.dw_aperture = QComboBox()
        self.dw_aperture.addItems(["—— 请选择口径类型 ——", "圆口径", "矩形", "正多边形"])
        combox_input(aperture_form, "口径类型：", self.dw_aperture)
        self.dw_box_diameter = self._dwell_field_box(
            aperture_form, "dw_model_diameter", "外径/外接直径 (mm)：", "100")
        self.dw_box_inner = self._dwell_field_box(
            aperture_form, "dw_model_inner", "内径 (mm)：", "0")
        self.dw_box_lx = self._dwell_field_box(aperture_form, "dw_model_lx", "LX (mm)：", "100")
        self.dw_box_ly = self._dwell_field_box(aperture_form, "dw_model_ly", "LY (mm)：", "100")
        self.dw_box_corner = self._dwell_field_box(
            aperture_form, "dw_model_corner", "圆角半径 (mm)：", "0")
        self.dw_box_sides = self._dwell_field_box(aperture_form, "dw_model_sides", "边数：", "6")
        self.dw_box_first_angle = self._dwell_field_box(
            aperture_form, "dw_model_first_angle", "首顶点角度 (°)：", "0")
        layout.addWidget(aperture_group)

        # ── 面形参数：面形类型决定母镜表面形状，选类型后只显示对应参数 ──
        surface_group = QGroupBox("面形参数"); form = QVBoxLayout(surface_group)
        self.dw_model_load = QComboBox(); self.dw_model_load.addItems(["否", "是"])
        combox_input(form, "读取面形参数：", self.dw_model_load)
        self.dw_model_load.currentIndexChanged.connect(self._dwell_load_combo_changed)
        self.dw_model_type = QComboBox()
        self.dw_model_type.addItems(
            ["—— 请选择面形类型 ——", "平面", "球面", "非球面", "柱面/双曲率", "自由曲面"])
        combox_input(form, "面形类型：", self.dw_model_type)
        self.dw_box_plane_hint = QWidget()
        hint_layout = QVBoxLayout(self.dw_box_plane_hint)
        hint_layout.setContentsMargins(0, 0, 0, 0); hint_layout.setSpacing(0)
        hint = QLabel("平面无需面形参数,其半径R为无穷。")
        hint.setStyleSheet("color:#50647a; font-size:11px;")
        hint_layout.addWidget(hint)
        form.addWidget(self.dw_box_plane_hint)
        self.dw_box_r = self._dwell_field_box(form, "dw_model_r", "曲率半径 R (mm)：", "100")
        self.dw_box_k = self._dwell_field_box(form, "dw_model_k", "圆锥常数 K：", "0")
        self.dw_box_a4 = self._dwell_field_box(form, "dw_model_a4", "A4：", "0")
        self.dw_box_a6 = self._dwell_field_box(form, "dw_model_a6", "A6：", "0")
        self.dw_box_a8 = self._dwell_field_box(form, "dw_model_a8", "A8：", "0")
        self.dw_box_rx = self._dwell_field_box(form, "dw_model_rx", "X曲率半径 Rx (mm)：", "100")
        self.dw_box_kx = self._dwell_field_box(form, "dw_model_kx", "X圆锥常数 Kx：", "0")
        self.dw_box_ry = self._dwell_field_box(form, "dw_model_ry", "Y曲率半径 Ry (mm)：", "100")
        self.dw_box_ky = self._dwell_field_box(form, "dw_model_ky", "Y圆锥常数 Ky：", "0")
        self.dw_box_ax = self._dwell_field_box(
            form, "dw_model_ax", "双曲率 Ax 高次项（逗号分隔）：", "")
        self.dw_box_ay = self._dwell_field_box(
            form, "dw_model_ay", "双曲率 Ay 高次项（逗号分隔）：", "")
        self.dw_box_axy = self._dwell_field_box(
            form, "dw_model_axy", "自由曲面 Axy 1～9阶（最多54项）：", "")
        layout.addWidget(surface_group)

        # ── 离轴参数：单独成框，决定从母镜上切取离轴段的位置与姿态 ──
        offaxis_group = QGroupBox("离轴参数"); offaxis_form = QVBoxLayout(offaxis_group)
        self._dwell_add_fields(offaxis_form, [
            ("dw_model_xoff", "X 离轴量 (mm)：", "0"),
            ("dw_model_yoff", "Y 离轴量 (mm)：", "0"),
            ("dw_model_tilt", "倾斜角 (°)：", "0"),
        ])
        # 远轴角位 γ = atan2(By, Bx)：由离轴量自动算出，只读
        self.dw_model_gamma = QLineEdit("0")
        self.dw_model_gamma.setReadOnly(True)
        self.dw_model_gamma.setStyleSheet("background:#e8f0fa; color:#10243f;")
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("远轴角位 (°)："))
        gamma_row.addWidget(self.dw_model_gamma)
        offaxis_form.addLayout(gamma_row)
        self.dw_model_xoff.textChanged.connect(self._dwell_sync_gamma)
        self.dw_model_yoff.textChanged.connect(self._dwell_sync_gamma)
        self._dwell_sync_gamma()
        layout.addWidget(offaxis_group)

        # ── 案例计算网格：导入工件阶段设定，后续页面只读复用 ──
        grid_group = QGroupBox("案例计算网格"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_model_dx", "X 间距 (mm)：", "1"),
            ("dw_model_dy", "Y 间距 (mm)：", "1"),
        ])
        self.dw_model_dx.textChanged.connect(self._dwell_invalidate_model_grid)
        self.dw_model_dy.textChanged.connect(self._dwell_invalidate_model_grid)
        grid_note = QLabel(
            "此网格用于期望曲面、面形重采样和后续驻留计算；导入工件阶段可调整，"
            "生成模型后其余阶段只读。检测 PSD 会优先采用测量文件自身的采样间距。")
        grid_note.setWordWrap(True)
        grid_note.setStyleSheet("color:#52657a; font-size:11px;")
        grid_form.addWidget(grid_note)
        collapsible(layout, "高级：案例计算网格（默认 1×1 mm）", grid_group)

        self._model_box_attrs = (
            "dw_box_diameter", "dw_box_inner", "dw_box_lx", "dw_box_ly", "dw_box_corner",
            "dw_box_sides", "dw_box_first_angle", "dw_box_plane_hint",
            "dw_box_r", "dw_box_k", "dw_box_a4", "dw_box_a6", "dw_box_a8",
            "dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky",
            "dw_box_ax", "dw_box_ay", "dw_box_axy",
        )
        self.dw_aperture.currentIndexChanged.connect(self._dwell_model_sync_visibility)
        self.dw_model_type.currentIndexChanged.connect(self._dwell_model_sync_visibility)
        self._dwell_model_sync_visibility()

        row = QHBoxLayout()
        generate = QPushButton("生成曲面模型"); save = QPushButton("保存面形参数")
        clear = QPushButton("清空")
        for button in (generate, save, clear): row.addWidget(button)
        layout.addLayout(row)
        self.dw_model_info = QLabel(""); self.dw_model_info.setWordWrap(True)
        layout.addWidget(self.dw_model_info)
        self._dwell_add_next_button(
            layout, "dw_model_next", "下一步：导入加工前实测面形",
            "idx_dwell_surface", required=("model",),
            tooltip="生成期望曲面模型后进入加工前实测面形页面。")
        layout.addStretch()
        generate.clicked.connect(self._dwell_generate_model)
        save.clicked.connect(self._dwell_save_model_config)
        clear.clicked.connect(self._clear_dwell_model_fields)
        return scroll

    def _dwell_invalidate_model_grid(self, *_args):
        if self._dwell_state.get("model") is None:
            return
        self._dwell_state["model"] = None
        self.dw_model_info.setText("计算网格已更改，请重新生成期望曲面模型。")
        self.dw_model_info.setStyleSheet("color:#73510d;")
        if hasattr(self, "_stage_headers"):
            self._refresh_workflow_statuses()

    def _dwell_field_box(self, layout, attr, label, default):
        edit, row = lineedit_input(label, default)
        row.setContentsMargins(0, 0, 0, 0)
        setattr(self, attr, edit)
        box = QWidget(); box.setLayout(row)
        layout.addWidget(box)
        return box

    def _dwell_model_sync_visibility(self, *_args):
        """按口径/面形下拉框的当前选择，只显示该类型需要填写的参数行。"""
        aperture_visible = {
            "圆口径": ("dw_box_diameter", "dw_box_inner"),
            "矩形": ("dw_box_lx", "dw_box_ly", "dw_box_corner"),
            "正多边形": ("dw_box_diameter", "dw_box_sides", "dw_box_first_angle"),
        }
        surface_visible = {
            "平面": ("dw_box_plane_hint",),
            "球面": ("dw_box_r", "dw_box_k"),
            "非球面": ("dw_box_r", "dw_box_k", "dw_box_a4", "dw_box_a6", "dw_box_a8"),
            "柱面/双曲率": ("dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky",
                          "dw_box_ax", "dw_box_ay"),
            "自由曲面": ("dw_box_rx", "dw_box_kx", "dw_box_ry", "dw_box_ky", "dw_box_axy"),
        }
        shown = set(aperture_visible.get(self.dw_aperture.currentText(), ()))
        shown |= set(surface_visible.get(self.dw_model_type.currentText(), ()))
        for attr in self._model_box_attrs:
            getattr(self, attr).setVisible(attr in shown)

    def _dwell_sync_gamma(self, *_args):
        """远轴角位 γ = atan2(By, Bx)，由离轴量自动算出（只读显示）。"""
        try:
            bx = float(self.dw_model_xoff.text())
        except ValueError:
            bx = 0.0
        try:
            by = float(self.dw_model_yoff.text())
        except ValueError:
            by = 0.0
        gamma = math.degrees(math.atan2(by, bx)) if abs(bx) + abs(by) > 0 else 0.0
        self.dw_model_gamma.setText(f"{gamma:.3f}")

    def _dwell_load_combo_changed(self, index):
        """面形参数组顶部的“是否读取”开关：选“是”时读取参数存档并复位。"""
        if self.dw_model_load.currentText() != "是":
            return
        self.dw_model_load.blockSignals(True)
        try:
            self._dwell_load_model_config()
        finally:
            self.dw_model_load.setCurrentIndex(0)
            self.dw_model_load.blockSignals(False)

    def _dwell_model_config(self):
        aperture = self.dw_aperture.currentText()
        if aperture not in ("圆口径", "矩形", "正多边形"):
            raise ValueError("请先选择口径类型")
        kind = self.dw_model_type.currentText()
        if kind not in ("平面", "球面", "非球面", "柱面/双曲率", "自由曲面"):
            raise ValueError("请先选择面形类型")
        return {
            "type": kind, "aperture": aperture,
            "r": self._dwell_float(self.dw_model_r, "R"), "k": self._dwell_float(self.dw_model_k, "K"),
            "rx": self._dwell_float(self.dw_model_rx, "Rx"), "kx": self._dwell_float(self.dw_model_kx, "Kx"),
            "ry": self._dwell_float(self.dw_model_ry, "Ry"), "ky": self._dwell_float(self.dw_model_ky, "Ky"),
            "a": [self._dwell_float(self.dw_model_a4, "A4"),
                  self._dwell_float(self.dw_model_a6, "A6"), self._dwell_float(self.dw_model_a8, "A8")],
            "ax": self._dwell_coefficients(self.dw_model_ax, "Ax"),
            "ay": self._dwell_coefficients(self.dw_model_ay, "Ay"),
            "axy": self._dwell_coefficients(self.dw_model_axy, "Axy", 54),
            "x_offset": self._dwell_float(self.dw_model_xoff, "X离轴量"),
            "y_offset": self._dwell_float(self.dw_model_yoff, "Y离轴量"),
            "tilt": self._dwell_float(self.dw_model_tilt, "倾斜角"),
            "dx": float(self._dwell_state.get("dx", 1.0)),
            "dy": float(self._dwell_state.get("dy", 1.0)),
            "diameter": self._dwell_float(self.dw_model_diameter, "外径", 0.0, False),
            "inner_diameter": self._dwell_float(self.dw_model_inner, "内径", 0.0),
            "lx": self._dwell_float(self.dw_model_lx, "LX", 0.0, False),
            "ly": self._dwell_float(self.dw_model_ly, "LY", 0.0, False),
            "corner": self._dwell_float(self.dw_model_corner, "圆角半径", 0.0),
            "sides": int(self._dwell_float(self.dw_model_sides, "边数", 3.0)),
            "first_angle": self._dwell_float(self.dw_model_first_angle, "首顶点角度"),
        }

    def _dwell_generate_model(self):
        try:
            dx = self._dwell_float(self.dw_model_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_model_dy, "Y间距", 0.0, False)
            grid_changed = (self._dwell_state.get("dx") != dx or
                            self._dwell_state.get("dy") != dy)
            if self._dwell_state.get("dx") != dx:
                self._dwell_state["dx"] = dx
            if self._dwell_state.get("dy") != dy:
                self._dwell_state["dy"] = dy
            for attr, value in (("dw_dx", dx), ("dw_dy", dy),
                                ("dw_surface_dx", dx), ("dw_surface_dy", dy),
                                ("dw_spot_dx", dx), ("dw_spot_dy", dy),
                                ("dw_solve_dx", dx), ("dw_solve_dy", dy)):
                if hasattr(self, attr):
                    getattr(self, attr).setText(f"{value:g}")
            config = self._dwell_model_config()
            model = DwellTimeEngine.generate_model(config)
            self._dwell_state["model"] = model
            raw_surface = self._dwell_state.get("surface_raw")
            if raw_surface is not None and grid_changed:
                source_pixel = self._dwell_state.get("surface_pixel")
                if source_pixel is None:
                    resampled = np.asarray(raw_surface, dtype=float).copy()
                else:
                    resampled = DwellTimeEngine.resample_grid(
                        raw_surface, source_pixel, source_pixel, dx, dy)
                self._dwell_state["surface"] = resampled
                if hasattr(self, "dw_surface_info"):
                    self.dw_surface_info.setText(
                        "计算网格已更新，面形已从原始测量数据重新采样；请重新检查并应用面形处理。")
            valid = int(np.count_nonzero(model["mask"]))
            alignment_text = self._workflow_refresh_alignment()
            self._dwell_report(
                f"曲面模型生成完成：{valid} 个有效节点。"
                + (f"\n{alignment_text}" if self._dwell_state.get("surface") is not None else ""),
                self.dw_model_info)
            try:
                self._model_view_name = f"{config['type']} | 口径类型 {config['aperture']}"
                self._main.preview.plot_dwell_model(model, self._model_view_name)
            except Exception as exc:
                self._dwell_report(f"三维预览失败：{exc}", self.dw_model_info, error=True)
        except Exception as exc:
            self._dwell_error("建模失败", exc, self.dw_model_info)

    def _dwell_save_model_config(self):
        try: config = self._dwell_model_config()
        except ValueError as exc:
            self._dwell_error("保存失败", exc, self.dw_model_info); return
        path, _ = QFileDialog.getSaveFileName(self._main, "保存面形参数", "dwell_model.json", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as stream: json.dump(config, stream, ensure_ascii=False, indent=2)
            self._dwell_report(f"面形参数已保存：{os.path.basename(path)}", self.dw_model_info)
        except OSError as exc: self._dwell_error("保存失败", exc, self.dw_model_info)

    def _dwell_apply_model_config(self, config):
        """把面形参数字典写回建模页各输入框（读取参数文件与重新打开案例共用）。"""
        mapping = {
            "r":"dw_model_r", "k":"dw_model_k", "rx":"dw_model_rx", "kx":"dw_model_kx",
            "ry":"dw_model_ry", "ky":"dw_model_ky", "x_offset":"dw_model_xoff",
            "y_offset":"dw_model_yoff", "tilt":"dw_model_tilt",
            "diameter":"dw_model_diameter", "inner_diameter":"dw_model_inner", "lx":"dw_model_lx",
            "ly":"dw_model_ly", "corner":"dw_model_corner", "sides":"dw_model_sides",
            "first_angle":"dw_model_first_angle",
        }
        for key, attr in mapping.items():
            if key in config: getattr(self, attr).setText(str(config[key]))
        if "type" in config: self.dw_model_type.setCurrentText(config["type"])
        if "aperture" in config: self.dw_aperture.setCurrentText(config["aperture"])
        values = list(config.get("a", [])) + [0, 0, 0]
        for attr, value in zip(("dw_model_a4", "dw_model_a6", "dw_model_a8"), values): getattr(self, attr).setText(str(value))
        for key, attr in (("ax", "dw_model_ax"), ("ay", "dw_model_ay"), ("axy", "dw_model_axy")):
            getattr(self, attr).setText(",".join(str(value) for value in config.get(key, [])))

    def _dwell_load_model_config(self):
        path, _ = QFileDialog.getOpenFileName(self._main, "读取面形参数", "", "JSON (*.json)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as stream: config = json.load(stream)
            self._dwell_apply_model_config(config)
            self._dwell_report(f"面形参数已读取：{os.path.basename(path)}", self.dw_model_info)
        except Exception as exc: self._dwell_error("读取失败", exc, self.dw_model_info)

    def _clear_dwell_model_fields(self):
        for attr in ("dw_model_r", "dw_model_k", "dw_model_rx", "dw_model_kx", "dw_model_ry",
                     "dw_model_ky", "dw_model_a4", "dw_model_a6", "dw_model_a8", "dw_model_ax",
                     "dw_model_ay", "dw_model_axy", "dw_model_xoff",
                     "dw_model_yoff", "dw_model_tilt", "dw_model_diameter", "dw_model_inner",
                     "dw_model_lx", "dw_model_ly", "dw_model_corner",
                     "dw_model_sides", "dw_model_first_angle"):
            getattr(self, attr).clear()
        self.dw_aperture.setCurrentIndex(0)
        self.dw_model_type.setCurrentIndex(0)
        self._dwell_state["model"] = None
        self._dwell_report("建模参数和模型缓存已清空。", self.dw_model_info)
