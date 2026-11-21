"""Migrated cnc page behavior; existing layout preserved."""
from __future__ import annotations
import os
import numpy as np
from PyQt5.QtWidgets import QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from polishing.ui.widgets.inputs import combox_input
from polishing.workflow.engine import DwellTimeEngine

class ExecutionPageMixin:
    def _build_dwell_cnc_page(self):
        scroll, layout = self._dwell_scroll_page(
            "生成程序", "把已确认的方案转换为栅线轨迹、工具姿态、受限进给速度和 PATH 文件。")
        self.dw_cnc_vibration_note = QLabel(
            "当前确认的方案是共形振动。共形振动的执行文件尚未支持，这里不需要填写任何参数；"
            "请按方案中的振动参数在设备上手动设置。")
        self.dw_cnc_vibration_note.setWordWrap(True)
        self.dw_cnc_vibration_note.setStyleSheet(
            "background:#fff8e8; color:#73510d; padding:8px; border:1px solid #e4c276;")
        self.dw_cnc_vibration_note.setVisible(False)
        layout.addWidget(self.dw_cnc_vibration_note)

        self.dw_cnc_controls = QWidget()
        controls = QVBoxLayout(self.dw_cnc_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        speed_group = QGroupBox("1  进给速度"); speed_form = QVBoxLayout(speed_group)
        self._dwell_add_fields(speed_form, [
            ("dw_cnc_max_speed", "最高进给速度：", "50"),
            ("dw_cnc_min_speed", "最低进给速度：", "0.01"),
        ])
        controls.addWidget(speed_group)

        tool_group = QGroupBox("2  工具参数"); tool_form = QVBoxLayout(tool_group)
        self._dwell_add_fields(tool_form, [
            ("dw_cnc_tool_radius", "工具球半径 (mm)：", "0"),
            ("dw_cnc_cylinder", "气缸压入量 (mm)：", "8"),
        ])
        controls.addWidget(tool_group)

        offset_group = QGroupBox("3  偏置与姿态"); offset_form = QVBoxLayout(offset_group)
        self._dwell_add_fields(offset_form, [
            ("dw_cnc_xoffset", "X 偏置 (mm)：", "0"), ("dw_cnc_yoffset", "Y 偏置 (mm)：", "0"),
            ("dw_cnc_zoffset", "Z 偏置 (mm)：", "0"), ("dw_cnc_rx", "绕 X 角 (°)：", "0"),
            ("dw_cnc_ry", "绕 Y 角 (°)：", "0"),
        ])
        controls.addWidget(offset_group)

        # 路径在制定方案中确定、预测按它计算；这里只读显示，避免执行与已确认方案不一致
        path_group = QGroupBox("4  生成轨迹（路径来自制定方案）"); path_form = QVBoxLayout(path_group)
        self.dw_cnc_path_info = QLabel()
        self.dw_cnc_path_info.setWordWrap(True)
        self.dw_cnc_path_info.setStyleSheet("color:#52657a; font-size:11px;")
        path_form.addWidget(self.dw_cnc_path_info)
        self._dwell_add_fields(path_form, [
            ("dw_cnc_dx", "轨迹间距 X (mm)：", "1"),
            ("dw_cnc_dy", "轨迹间距 Y (mm)：", "1"),
        ], readonly=("dw_cnc_dx", "dw_cnc_dy"))
        self.dw_cnc_line = QComboBox(); self.dw_cnc_line.addItems(["X", "Y"])
        self.dw_cnc_line.setEnabled(False)
        combox_input(path_form, "线方向：", self.dw_cnc_line)
        self.dw_cnc_start = QComboBox(); self.dw_cnc_start.addItems(["左下", "右下", "右上", "左上"])
        combox_input(path_form, "起始点位：", self.dw_cnc_start)
        traj = QPushButton("表面轨迹生成"); traj.setObjectName("primaryBtn")
        path_form.addWidget(traj)
        controls.addWidget(path_group)

        cnc_group = QGroupBox("5  生成并保存 CNC"); cnc_form = QVBoxLayout(cnc_group)
        row = QHBoxLayout()
        cnc = QPushButton("CNC 程序生成"); save = QPushButton("保存 CNC 代码")
        row.addWidget(cnc); row.addWidget(save)
        cnc_form.addLayout(row)
        self.dw_cnc_info = QLabel(""); self.dw_cnc_info.setWordWrap(True)
        cnc_form.addWidget(self.dw_cnc_info)
        controls.addWidget(cnc_group)
        layout.addWidget(self.dw_cnc_controls)
        layout.addStretch()
        traj.clicked.connect(self._dwell_generate_trajectory)
        cnc.clicked.connect(self._dwell_generate_cnc)
        save.clicked.connect(self._dwell_save_cnc)
        return scroll

    def _refresh_execute_view(self):
        if not hasattr(self, "dw_cnc_controls"):
            return
        frozen = self._dwell_state.get("frozen_plan") or {}
        vibration = frozen.get("process_type") == "conformal_vibration"
        self.dw_cnc_vibration_note.setVisible(vibration)
        self.dw_cnc_controls.setVisible(not vibration)
        plan = self._dwell_state.get("process_plan") or {}
        coverage = plan.get("coverage")
        if coverage:
            self.dw_cnc_path_info.setText(
                f"来自制定方案：栅形，沿 {coverage['direction']} 方向走刀，行间距 "
                f"{coverage['line_spacing_mm']:g} mm，点间距 {coverage['point_step_mm']:g} mm。"
                "要改路径请回到制定方案第 4 步，并重新预测、确认。")
        else:
            self.dw_cnc_path_info.setText("制定方案中尚未确定覆盖路径。")

    def _dwell_generate_trajectory(self):
        try:
            solution = self._dwell_state.get("solution")
            if solution is None: raise ValueError("请先完成驻留时间求解")
            step_x = self._dwell_float(self.dw_cnc_dx, "轨迹间距X", 0.0, False)
            step_y = self._dwell_float(self.dw_cnc_dy, "轨迹间距Y", 0.0, False)
            samples = DwellTimeEngine.raster_from_dwell(
                solution, step_x, step_y, self.dw_cnc_line.currentText(), self.dw_cnc_start.currentText())
            model = self._dwell_state.get("model")
            if model is None:
                xyz = np.column_stack((samples[:, :2], np.zeros(len(samples))))
                normals = np.tile((0.0, 0.0, 1.0), (len(samples), 1))
            else:
                xyz, normals = DwellTimeEngine.sample_nearest(model, samples[:, 0], samples[:, 1])
            valid = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1)
            samples = samples[valid]; xyz = xyz[valid]; normals = normals[valid]
            if len(samples) < 2: raise ValueError("模型口径内的有效轨迹点不足")
            trajectory = {"samples": samples, "xyz": xyz, "normals": normals,
                          "step_x": step_x, "step_y": step_y}
            self._dwell_state["trajectory"] = trajectory
            points = np.column_stack((xyz, normals)).tolist()
            self._trajectory_view = (points, {
                "surface_name": "驻留时间轨迹", "traj_name": "栅形轨迹",
                "traj_type": "G", "direction": self.dw_cnc_line.currentText(), "geom": None})
            self._main.preview.plot_surface(*self._trajectory_view)
            self._dwell_report(f"表面轨迹生成完成：{len(samples)} 个点。", self.dw_cnc_info)
        except Exception as exc: self._dwell_error("轨迹生成失败", exc, self.dw_cnc_info)

    def _dwell_generate_cnc(self):
        try:
            trajectory = self._dwell_state.get("trajectory")
            if trajectory is None: raise ValueError("请先生成表面轨迹")
            config = {
                "max_speed": self._dwell_float(self.dw_cnc_max_speed, "最高进给速度", 0.0, False),
                "min_speed": self._dwell_float(self.dw_cnc_min_speed, "最低进给速度", 0.0, False),
                "tool_radius": self._dwell_float(self.dw_cnc_tool_radius, "工具球半径", 0.0),
                "depth": self._dwell_float(self.dw_cnc_cylinder, "气缸压入量", 0.0),
                "x_offset": self._dwell_float(self.dw_cnc_xoffset, "X偏置"),
                "y_offset": self._dwell_float(self.dw_cnc_yoffset, "Y偏置"),
                "z_offset": self._dwell_float(self.dw_cnc_zoffset, "Z偏置"),
                "alpha": self._dwell_float(self.dw_cnc_rx, "绕X角"),
                "beta": self._dwell_float(self.dw_cnc_ry, "绕Y角"),
                "step_x": trajectory["step_x"], "step_y": trajectory["step_y"],
            }
            if config["min_speed"] > config["max_speed"]:
                raise ValueError("最低进给速度不能大于最高进给速度")
            cnc = DwellTimeEngine.generate_cnc(
                trajectory["samples"], trajectory["xyz"], trajectory["normals"], config,
                self._dwell_float(self.dw_solve_dx, "求解X间距", 0.0, False),
                self._dwell_float(self.dw_solve_dy, "求解Y间距", 0.0, False))
            self._dwell_state["cnc"] = cnc
            self._dwell_report(
                f"CNC 程序生成完成：{len(cnc['data'])} 行，总加工时间 {cnc['total_time']:.6g}。",
                self.dw_cnc_info)
        except Exception as exc: self._dwell_error("CNC 程序生成失败", exc, self.dw_cnc_info)

    def _dwell_save_cnc(self):
        cnc = self._dwell_state.get("cnc")
        if cnc is None:
            self._dwell_error("保存失败", ValueError("请先生成 CNC 程序"), self.dw_cnc_info); return
        path, _ = QFileDialog.getSaveFileName(self._main, "保存 CNC 代码", "dwell_time.path", "PATH (*.path);;TXT (*.txt)")
        if not path: return
        try:
            np.savetxt(path, cnc["data"], fmt="%.6f")
            self._dwell_report(f"CNC 代码已保存：{os.path.basename(path)}", self.dw_cnc_info)
        except OSError as exc: self._dwell_error("保存失败", exc, self.dw_cnc_info)
