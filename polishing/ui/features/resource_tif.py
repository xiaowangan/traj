"""Migrated tool page behavior; existing layout preserved."""
from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QGroupBox, QLabel, QPushButton, QVBoxLayout
from polishing.workflow.engine import DwellTimeEngine

class TifResourceMixin:
    def _build_dwell_spot_page(self):
        scroll, layout = self._dwell_scroll_page(
            "工艺资源与标定", "读取抛光斑，按采斑时长、偏置和计算网格生成当前方案使用的 TIF。")

        # ── 数据导入 ──
        import_group = QGroupBox("数据导入"); import_form = QVBoxLayout(import_group)
        self._dwell_add_fields(import_form, [
            ("dw_spot_file", "文件路径：", ""),
        ])
        select = QPushButton("选择抛光斑文件"); select.clicked.connect(self._dwell_select_spot)
        import_form.addWidget(select)
        layout.addWidget(import_group)

        # ── 采样参数 ──
        sample_group = QGroupBox("采样参数"); sample_form = QVBoxLayout(sample_group)
        self._dwell_add_fields(sample_form, [
            ("dw_spot_duration", "采斑时长 (s)：", "1"),
            ("dw_spot_xoffset", "X 偏置 (mm)：", "0"),
            ("dw_spot_yoffset", "Y 偏置 (mm)：", "0"),
        ])
        offset_hint = QLabel(
            "若采斑时斑点不完全居中，X/Y 偏置用于抛光斑重新对中")
        offset_hint.setStyleSheet("color:#50647a; font-size:11px;")
        offset_hint.setWordWrap(True)
        sample_form.addWidget(offset_hint)
        layout.addWidget(sample_group)

        # ── 计算网格（只读，来自导入工件阶段）──
        grid_group = QGroupBox("计算网格"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_spot_dx", "X 间距 (mm)：", "1"), ("dw_spot_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_spot_dx", "dw_spot_dy"))
        layout.addWidget(grid_group)
        grid_group.setVisible(False)  # 网格只在导入工件“期望曲面”中设定，这里不再重复显示

        calculate = QPushButton("计算去除函数"); calculate.clicked.connect(self._dwell_calculate_spot)
        layout.addWidget(calculate)
        self.dw_spot_info = QLabel(""); self.dw_spot_info.setWordWrap(True)
        layout.addWidget(self.dw_spot_info)
        self._dwell_add_next_button(
            layout, "dw_spot_next", "下一步：返回制定方案并生成候选方案",
            "idx_workflow_plan", required=("spot",),
            tooltip="TIF 准备完成后返回制定方案页，继续设置覆盖参数并生成候选方案。")
        self._resource_flow_widgets.append(self.dw_spot_next)
        layout.addStretch()
        return scroll

    def _dwell_select_spot(self):
        path, _ = QFileDialog.getOpenFileName(
            self._main, "选择抛光斑数据", self.dw_spot_file.text().strip(),
            "数据文件 (*.dat *.npy *.csv *.txt);;所有文件 (*.*)")
        if not path: return
        try:
            raw, pixel = DwellTimeEngine.read_grid(path)
            before = -np.asarray(raw, dtype=float)
            if np.isfinite(before).any():
                before = before - np.nanmin(before)
            self.dw_spot_file.setText(path)
            self._dwell_state.update(spot_raw=raw, spot_pixel=pixel,
                                     spot_before=before, spot=None)
            self._dwell_report(f"抛光斑原始数据已读取：{raw.shape[1]}×{raw.shape[0]}。", self.dw_spot_info)
            self._dwell_refresh_spot_preview()
        except Exception as exc: self._dwell_error("抛光斑读取失败", exc, self.dw_spot_info)

    def _dwell_refresh_spot_preview(self):
        """把抛光斑数据同步到左侧二维预览（左=取反后处理前，右=去除函数处理后）。"""
        before = self._dwell_state.get("spot_before")
        if before is None:
            return
        spot = self._dwell_state.get("spot")
        after = spot["kernel"] if spot is not None else before
        try:
            self._main.preview.plot_dwell_surface(
                before, after, "处理前", "处理后", colorbar_label="抛光斑数据")
        except Exception as exc:
            self._dwell_report(f"二维预览失败：{exc}", self.dw_spot_info, error=True)

    def _dwell_calculate_spot(self):
        try:
            raw = self._dwell_state.get("spot_raw")
            if raw is None: raise ValueError("请先选择抛光斑数据文件")
            spot = DwellTimeEngine.build_spot(
                raw, self._dwell_state.get("spot_pixel"),
                self._dwell_float(self.dw_spot_dx, "X间距", 0.0, False),
                self._dwell_float(self.dw_spot_dy, "Y间距", 0.0, False),
                self._dwell_float(self.dw_spot_duration, "采斑时长", 0.0, False),
                self._dwell_float(self.dw_spot_xoffset, "X偏置"),
                self._dwell_float(self.dw_spot_yoffset, "Y偏置"))
            self._dwell_state["spot"] = spot
            self._dwell_report(
                f"去除函数计算完成：{spot['kernel'].shape[1]}×{spot['kernel'].shape[0]}，"
                f"脉冲={spot['impulse']:.6g}，体去除效率={spot['volume_efficiency']:.6g} mm³/min。",
                self.dw_spot_info)
            self._dwell_refresh_spot_preview()
        except Exception as exc: self._dwell_error("去除函数计算失败", exc, self.dw_spot_info)
