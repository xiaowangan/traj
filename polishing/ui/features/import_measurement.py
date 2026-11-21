"""Migrated measurement page behavior; existing layout preserved."""
from __future__ import annotations
import os
from PyQt5.QtWidgets import QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from polishing.workflow.engine import DwellTimeEngine

class MeasurementImportMixin:
    def _build_dwell_surface_page(self):
        scroll, layout = self._dwell_scroll_page(
            "加工前实测面形", "读取当前工件的实际面形误差，并执行裁边、延拓和滤波。")

        # ── 数据导入：选择文件后读入面形矩阵 ──
        import_group = QGroupBox("选择数据文件"); import_form = QVBoxLayout(import_group)
        self._dwell_add_fields(import_form, [
            ("dw_surface_file", "文件路径：", ""),
        ])
        select = QPushButton("选择面形数据文件"); select.clicked.connect(self._dwell_select_surface)
        import_form.addWidget(select)
        pv_hint = QLabel("导入后自动计算 PV 与 RMS：PV 为面形最高点与最低点之差，RMS 为整体面形偏差的均方根。")
        pv_hint.setStyleSheet("color:#50647a; font-size:11px;")
        pv_hint.setWordWrap(True)
        import_form.addWidget(pv_hint)
        layout.addWidget(import_group)

        # ── 处理参数：裁边、延拓与滤波 ──
        process_group = QGroupBox("裁边、延拓与滤波（可选）"); process_form = QVBoxLayout(process_group)
        self._dwell_add_fields(process_form, [
            ("dw_surface_trim", "裁边尺寸 (mm)：", "0"),
            ("dw_surface_extend", "延拓尺寸 (mm)：", "0"),
        ])
        self._dwell_add_fields(process_form, [
            ("dw_surface_filter", "均值滤波窗尺寸 (mm)：", "2"),
        ])
        filter_hint = QLabel("将窗口内平均值替换各点，以平滑高频噪声。")
        filter_hint.setStyleSheet("color:#50647a; font-size:11px;")
        filter_hint.setWordWrap(True)
        process_form.addWidget(filter_hint)
        self._dwell_add_fields(process_form, [
            ("dw_surface_rms", "RMS 倍数：", "3"),
        ])
        rms_hint = QLabel("偏离均值大于RMS倍数×RMS的点压回阈值，抑制坏点与突变。")
        rms_hint.setStyleSheet("color:#50647a; font-size:11px;")
        rms_hint.setWordWrap(True)
        process_form.addWidget(rms_hint)
        layout.addWidget(process_group)

        # ── 网格间距（只读，来自导入工件阶段的期望曲面）──
        grid_group = QGroupBox("网格间距"); grid_form = QVBoxLayout(grid_group)
        self._dwell_add_fields(grid_form, [
            ("dw_surface_dx", "X 间距 (mm)：", "1"),
            ("dw_surface_dy", "Y 间距 (mm)：", "1"),
        ], readonly=("dw_surface_dx", "dw_surface_dy"))
        layout.addWidget(grid_group)
        grid_group.setVisible(False)  # 网格只在导入工件“期望曲面”中设定，这里不再重复显示

        row = QHBoxLayout()
        actions = (("裁边", self._dwell_trim_surface), ("延拓/填充", self._dwell_extend_surface),
                   ("均值滤波", self._dwell_mean_surface), ("RMS 滤波", self._dwell_rms_surface))
        for caption, callback in actions:
            button = QPushButton(caption); button.clicked.connect(callback); row.addWidget(button)
        layout.addLayout(row)
        self.dw_surface_info = QLabel(""); self.dw_surface_info.setWordWrap(True)
        layout.addWidget(self.dw_surface_info)
        self._dwell_add_next_button(
            layout, "dw_surface_next", "下一步：面形初测",
            "idx_workflow_evaluate", required=("surface",),
            tooltip="加工前实测面形准备完成后进入面形初测阶段。")
        layout.addStretch()
        return scroll

    def _dwell_select_surface(self):
        path, _ = QFileDialog.getOpenFileName(
            self._main, "选择面形数据", self.dw_surface_file.text().strip(),
            "数据文件 (*.dat *.npy *.csv *.txt);;所有文件 (*.*)")
        if not path: return
        try:
            raw, pixel = DwellTimeEngine.read_grid(path)
            dx = self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False)
            dy = self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)
            data = raw if pixel is None else DwellTimeEngine.resample_grid(raw, pixel, pixel, dx, dy)
            self.dw_surface_file.setText(path)
            # The case grid is owned by the model/import stage. Writing dx/dy here
            # invalidates the just-loaded surface through WorkflowState dependents.
            meta = {"file": os.path.basename(path), "wavelength_nm_header": None}
            if path.lower().endswith(".dat"):
                try:
                    meta["wavelength_nm_header"] = DwellTimeEngine.read_zygo_header(path)["wavelength_nm"]
                except (OSError, ValueError):
                    pass
            self._dwell_state.update(
                surface_raw=raw, surface_pixel=pixel, surface=data, surface_meta=meta)
            stat = DwellTimeEngine.stats(data)
            wavelength = meta["wavelength_nm_header"]
            self._workflow_apply_surface_meta()
            self._dwell_report(
                f"面形已读取：{data.shape[1]}×{data.shape[0]}，PV={stat['pv']:.6g}，RMS={stat['rms']:.6g}。"
                + (f"文件头测量波长 {wavelength:g} nm（面形检测时自动换算为 nm）。" if wavelength else "")
                + "\n" + self._workflow_refresh_alignment(),
                self.dw_surface_info)
            self._dwell_refresh_surface_preview()
        except Exception as exc: self._dwell_error("面形读取失败", exc, self.dw_surface_info)

    def _require_surface(self):
        data = self._dwell_state.get("surface")
        if data is None: raise ValueError("请先选择面形数据文件")
        return data

    def _dwell_refresh_surface_preview(self):
        """按建模页同款逻辑，把面形数据同步到左侧二维预览（左=导入，右=处理后）。"""
        raw = self._dwell_state.get("surface_raw")
        processed = self._dwell_state.get("surface")
        if raw is None or processed is None:
            return
        try:
            self._main.preview.plot_dwell_surface(raw, processed)
        except Exception as exc:
            self._dwell_report(f"二维预览失败：{exc}", self.dw_surface_info, error=True)

    def _surface_operation(self, name, operation):
        try:
            data = operation(self._require_surface())
            self._dwell_state["surface"] = data
            stat = DwellTimeEngine.stats(data)
            self._dwell_report(f"{name}完成：有效点 {stat['count']}，PV={stat['pv']:.6g}，RMS={stat['rms']:.6g}。", self.dw_surface_info)
            self._dwell_refresh_surface_preview()
        except Exception as exc: self._dwell_error(name + "失败", exc, self.dw_surface_info)

    def _dwell_trim_surface(self):
        self._surface_operation("裁边", lambda data: DwellTimeEngine.trim_surface(
            data, self._dwell_float(self.dw_surface_trim, "裁边尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))

    def _dwell_mean_surface(self):
        self._surface_operation("均值滤波", lambda data: DwellTimeEngine.mean_filter(
            data, self._dwell_float(self.dw_surface_filter, "滤波窗尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))

    def _dwell_rms_surface(self):
        self._surface_operation("RMS滤波", lambda data: DwellTimeEngine.rms_filter(
            data, self._dwell_float(self.dw_surface_rms, "RMS倍数", 1.0)))

    def _dwell_extend_surface(self):
        self._surface_operation("延拓/填充", lambda data: DwellTimeEngine.extend_surface(
            data, self._dwell_float(self.dw_surface_extend, "延拓尺寸", 0.0),
            self._dwell_float(self.dw_surface_dx, "X间距", 0.0, False),
            self._dwell_float(self.dw_surface_dy, "Y间距", 0.0, False)))
