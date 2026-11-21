"""Stage two: evaluate the measured surface."""
import math
from PyQt5.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QVBoxLayout
from polishing.core.model_evaluation import AcceptanceTarget, EvaluationProtocol
from polishing.workflow.evaluation_service import check_current_alignment, evaluate_current_surface

_NOTE_STYLE = "color:#52657a; font-size:11px;"
_OK_STYLE = "background:#e8f5ec; color:#1a6f3a; padding:6px; border:1px solid #9bc7aa;"
_WARN_STYLE = "background:#fff8e8; color:#73510d; padding:6px; border:1px solid #e4c276;"

class EvaluationPageMixin:
    def _build_evaluation_controls(self, box):
        # 检测设置直接展开：1 检测标准 → 2 加工目标 → 3 计算并看结论
        controls_layout = box
        protocol = QGroupBox("1  检测标准")
        protocol_box = QVBoxLayout(protocol)
        self._dwell_add_fields(protocol_box, [
            ("workflow_eval_min", "中频下限 (周期/mm)：", ""),
            ("workflow_eval_max", "中频上限 (周期/mm)：", ""),
            ("workflow_eval_wl_min", "波长下限 (mm)：", ""),
            ("workflow_eval_wl_max", "波长上限 (mm)：", ""),
        ])
        self._band_syncing = False
        # 频率与波长互为倒数：频率下限 ↔ 波长上限，频率上限 ↔ 波长下限
        for source, target in ((self.workflow_eval_min, self.workflow_eval_wl_max),
                               (self.workflow_eval_wl_max, self.workflow_eval_min),
                               (self.workflow_eval_max, self.workflow_eval_wl_min),
                               (self.workflow_eval_wl_min, self.workflow_eval_max)):
            source.textChanged.connect(
                lambda _text, src=source, dst=target: self._workflow_link_reciprocal(src, dst))
        self.workflow_eval_detrend = QComboBox()
        self.workflow_eval_detrend.addItem("去除倾斜平面", "plane")
        self.workflow_eval_detrend.addItem("仅去除均值", "piston")
        self.workflow_eval_detrend.addItem("不去除低阶项", "none")
        self._add_inline_combo(protocol_box, "低阶去除：", self.workflow_eval_detrend)
        self.workflow_eval_unit_hint = QLabel("高度单位：nm。")
        self.workflow_eval_unit_hint.setWordWrap(True)
        self.workflow_eval_unit_hint.setStyleSheet(_NOTE_STYLE)
        protocol_box.addWidget(self.workflow_eval_unit_hint)
        self.workflow_eval_min.textChanged.connect(self._workflow_invalidate_evaluation)
        self.workflow_eval_max.textChanged.connect(self._workflow_invalidate_evaluation)
        self.workflow_eval_detrend.currentIndexChanged.connect(self._workflow_invalidate_evaluation)
        hint = QLabel(
            "频段按工件技术要求填写，不存在通用中频范围。频率与波长互为倒数，填写其中一边，另一边自动算出"
            "（频率下限对应波长上限，频率上限对应波长下限）。"
            "所有高度结果统一用 nm。"
            "分析只在最大连续有效矩形内进行；不足以覆盖所选频段时会提示。插值加密不会提高真实测量带宽。")
        hint.setWordWrap(True)
        hint.setStyleSheet(_NOTE_STYLE)
        protocol_box.addWidget(hint)
        controls_layout.addWidget(protocol)

        target = QGroupBox("2  加工目标（验收标准）")
        target_box = QVBoxLayout(target)
        self._dwell_add_fields(target_box, [
            ("workflow_target_msf", "中频 RMS 上限：", ""),
            ("workflow_target_rms", "去低阶 RMS 上限：", ""),
            ("workflow_target_time", "累计驻留/加工时间上限 (h)：", ""),
        ])
        target_hint = QLabel(
            "各项可留空表示不设限。RMS 上限的单位为 nm。效果预估时会用本标准和本目标判定能否达标。")
        target_hint.setWordWrap(True)
        target_hint.setStyleSheet(_NOTE_STYLE)
        target_box.addWidget(target_hint)
        for edit in (self.workflow_target_msf, self.workflow_target_rms, self.workflow_target_time):
            edit.textChanged.connect(self._workflow_sync_target)
        controls_layout.addWidget(target)

        result_group = QGroupBox("3  计算并查看结论")
        controls_layout.addWidget(result_group)
        controls_layout = QVBoxLayout(result_group)
        evaluate = QPushButton("计算检测结果")
        evaluate.setObjectName("primaryBtn")
        evaluate.clicked.connect(self._workflow_evaluate_surface)
        controls_layout.addWidget(evaluate)
        self.workflow_evaluation_result = QLabel("尚未检测。")
        self.workflow_evaluation_result.setWordWrap(True)
        self.workflow_evaluation_result.setStyleSheet(
            "background:#e8f0fa; color:#10243f; padding:8px; border:1px solid #b8c7d8;")
        controls_layout.addWidget(self.workflow_evaluation_result)
        self._dwell_add_next_button(
            controls_layout, "workflow_evaluate_next", "下一步：制定方案",
            "idx_workflow_plan", required=("evaluation",),
            tooltip="完成加工前面形检测后进入制定方案阶段。")

    def _workflow_process_suggestion(self):
        """按检测结果给出加工方式建议（只是建议，由用户在制定方案里决定）。"""
        evaluation = self._dwell_state.get("evaluation")
        if evaluation is None:
            return "完成“面形初测”后，这里会根据超差情况给出加工方式建议。"
        msf = evaluation["band_rms"]
        form = evaluation["detrended_stats"]["rms"]
        target = self._dwell_state.get("target") or {}
        msf_max, rms_max = target.get("msf_rms_max"), target.get("rms_max")
        unit = evaluation["height_unit"]
        if msf_max is None and rms_max is None:
            low = evaluation["low_frequency_rms"]
            kind = "扫描驻留（修低频面形）" if low >= msf else "共形振动（平滑中频）"
            return (f"建议：{kind}。未设加工目标，按分带 RMS 比较：低频 {low:.4g}、中频 {msf:.4g} {unit}，"
                    "较大的一项优先处理。")
        msf_bad = msf_max is not None and msf > msf_max
        form_bad = rms_max is not None and form > rms_max
        if msf_bad and form_bad:
            return ("建议：先扫描驻留修低频面形，再共形振动平滑中频。"
                    f"去低阶 RMS {form:.4g} > {rms_max:g}，中频 RMS {msf:.4g} > {msf_max:g} {unit}。")
        if form_bad:
            return f"建议：扫描驻留（修低频面形）。去低阶 RMS {form:.4g} > 目标 {rms_max:g} {unit}。"
        if msf_bad:
            return f"建议：共形振动（平滑中频）。中频 RMS {msf:.4g} > 目标 {msf_max:g} {unit}。"
        return "当前面形已满足所设加工目标，可以不加工；如仍要加工，请自行选择方式。"

    def _workflow_link_reciprocal(self, source, target):
        """频率与波长互为倒数：一边被修改时同步另一边（0 或留空对应无穷大，另一边留空）。"""
        if self._band_syncing:
            return
        text = source.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = None
        self._band_syncing = True
        try:
            target.setText(f"{1.0 / value:.6g}" if value and value > 0 else "")
        finally:
            self._band_syncing = False

    def _workflow_sync_band_fields(self):
        """案例恢复等程序写入频率后，补齐对应的波长框。"""
        self._workflow_link_reciprocal(self.workflow_eval_min, self.workflow_eval_wl_max)
        self._workflow_link_reciprocal(self.workflow_eval_max, self.workflow_eval_wl_min)

    def _workflow_height_unit(self):
        """高度统一按 nm 检测：Zygo 数据（波长数）用文件头波长换算，其余数据视为 nm。"""
        meta = self._dwell_state.get("surface_meta") or {}
        wavelength = meta.get("wavelength_nm_header")
        # 文件头是 float32（632.79998…），取 6 位有效数字，与手工填写 632.8 一致
        return ("waves", float(f"{wavelength:g}")) if wavelength else ("nm", None)

    def _workflow_read_protocol(self):
        unit, wavelength = self._workflow_height_unit()
        return EvaluationProtocol(
            band_min=self._dwell_float(self.workflow_eval_min, "中频下限", 0.0),
            band_max=self._dwell_float(self.workflow_eval_max, "中频上限", 0.0, False),
            detrend=self.workflow_eval_detrend.currentData(),
            window="hann",
            height_unit=unit, wavelength_nm=wavelength).validate()

    def _workflow_parse_target(self):
        values = []
        for edit, name in ((self.workflow_target_msf, "中频 RMS 上限"),
                           (self.workflow_target_rms, "去低阶 RMS 上限"),
                           (self.workflow_target_time, "时间上限")):
            values.append(None if not edit.text().strip() else self._dwell_float(edit, name, 0.0, False))
        time_s = None if values[2] is None else values[2] * 3600.0
        return AcceptanceTarget(msf_rms_max=values[0], rms_max=values[1], time_max_s=time_s)

    def _workflow_sync_target(self, *_args):
        """加工目标写入案例状态；输入不合法时视为尚未设定。"""
        try:
            target = self._workflow_parse_target()
        except ValueError:
            target = None
        empty = target is None or target.is_empty()
        self._dwell_state["target"] = None if empty else target.to_dict()
        self._refresh_workflow_statuses()

    @staticmethod
    def _workflow_alignment_text(report):
        if report is None:
            return "对齐检查：尚缺期望曲面或实测面形。"
        if report["ok"]:
            offset = math.hypot(*report["center_offset_mm"])
            return (f"对齐检查通过：实测数据覆盖期望口径 {report['aperture_coverage']:.1%}，"
                    f"中心偏移 {offset:.3g} mm。{report['note']}")
        return "对齐检查有警告：" + "；".join(report["warnings"]) + " " + report["note"]

    def _workflow_refresh_alignment(self):
        """期望曲面或实测面形变化后重新检查对齐，并返回给用户看的文字。"""
        try:
            report = check_current_alignment(self._dwell_state)
        except ValueError as exc:
            report = None
            text = f"对齐检查未完成：{exc}"
        else:
            text = self._workflow_alignment_text(report)
        self._dwell_state["alignment"] = report
        label = getattr(self, "workflow_alignment_label", None)
        if label is not None:
            label.setText(text)
            label.setStyleSheet(_OK_STYLE if report and report["ok"] else
                                _WARN_STYLE if report else _NOTE_STYLE)
        return text

    def _workflow_apply_surface_meta(self):
        """说明高度单位来源：检测结果一律是 nm。"""
        meta = self._dwell_state.get("surface_meta") or {}
        wavelength = meta.get("wavelength_nm_header")
        if not wavelength:
            self.workflow_eval_unit_hint.setText("高度单位：nm（导入数据按 nm 处理）。")
            return
        self.workflow_eval_unit_hint.setText(
            f"高度单位：nm（Zygo 数据按文件头测量波长 {wavelength:g} nm 自动换算）。")

    def _workflow_evaluate_surface(self):
        surface = self._dwell_state.get("surface")
        if surface is None:
            self.workflow_evaluation_result.setText("请先在“导入”阶段导入加工前面形。")
            self.workflow_evaluation_result.setStyleSheet(
                "background:#fff0ef; color:#a33b31; padding:8px; border:1px solid #d9a19b;")
            return
        try:
            protocol = self._workflow_read_protocol()
            result = evaluate_current_surface(self._dwell_state, protocol)
        except Exception as exc:
            self.workflow_evaluation_result.setText(f"检测未完成：{exc}")
            self.workflow_evaluation_result.setStyleSheet(
                "background:#fff0ef; color:#a33b31; padding:8px; border:1px solid #d9a19b;")
            return
        evaluation_source = result["evaluation_source"]
        band_min, band_max = protocol.band_min, protocol.band_max
        self._dwell_state["protocol"] = protocol.to_dict()
        self._dwell_state["evaluation"] = result
        self._dwell_state.mark_current("protocol")
        self._dwell_state.mark_current("evaluation")
        self._workflow_sync_target()
        y0, y1, x0, x1 = result["roi"]
        stats = result["stats"]
        detrended = result["detrended_stats"]
        unit_suffix = result["height_unit"]
        self.workflow_evaluation_result.setText(
            f"检测完成｜检测网格 {result['grid_shape'][1]}×{result['grid_shape'][0]}，"
            f"采样间距 {result['dx_mm']:g}×{result['dy_mm']:g} mm（{evaluation_source}），"
            f"有效率 {result['valid_fraction']:.1%}；PSD 区域 {result['roi_shape'][1]}×"
            f"{result['roi_shape'][0]}（X[{x0}:{x1}]，Y[{y0}:{y1}]）。\n"
            f"区域 PV={stats['pv']:.5g} {unit_suffix}，RMS={stats['rms']:.5g} {unit_suffix}；"
            f"去低阶后 RMS={detrended['rms']:.5g} {unit_suffix}。\n"
            f"按 PSD 分带：低频 RMS={result['low_frequency_rms']:.5g}，"
            f"中频 RMS={result['band_rms']:.5g}，高频 RMS={result['high_frequency_rms']:.5g} "
            f"{unit_suffix}（低频 < {band_min:g}，高频 > {band_max:g} 周期/mm）。\n"
            f"中频 {band_min:g}–{band_max:g} 周期/mm（"
            f"波长 {'≥' if band_min == 0 else ''}{1/band_max:g}"
            f"{'–' + format(1/band_min, 'g') if band_min else ''} mm）："
            f"主导空间频率方向约 {result['dominant_direction_deg']:.0f}°。\n"
            f"保守 Nyquist={result['nyquist_mm_inv']:.4g} 周期/mm；"
            f"最低可分辨频率约 {result['min_resolvable_mm_inv']:.4g} 周期/mm"
            f"{'（所选下限低于分辨能力）' if result['band_below_resolution'] else ''}；"
            f"PSD 单位 {unit_suffix}²·mm²。低阶去除：{result['detrend']}；"
            "未扣除仪器噪声底。")
        self.workflow_evaluation_result.setStyleSheet(
            "background:#e8f5ec; color:#1a6f3a; padding:8px; border:1px solid #9bc7aa;")
        self._refresh_workflow_statuses()
        self._main.preview.plot_dwell_evaluation(result)

    def _workflow_invalidate_evaluation(self, *_args):
        if self._dwell_state.get("evaluation") is not None:
            self._dwell_state["evaluation"] = None
            self._refresh_workflow_statuses()
            self.workflow_evaluation_result.setText("检测标准已更改，请重新计算。")
            self.workflow_evaluation_result.setStyleSheet(
                "background:#fff8e8; color:#73510d; padding:8px; border:1px solid #e4c276;")
            self._dwell_refresh_surface_preview()
