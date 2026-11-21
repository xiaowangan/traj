"""Bind a calibrated TIF to the current scanning candidate."""
from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout

class PlanTifMixin:
    def _build_plan_tif_section(self, scan_layout):
        tif_group = QGroupBox("3  抛光斑/TIF")
        tif_box = QVBoxLayout(tif_group)
        tif_info = QLabel()
        tif_info.setWordWrap(True)
        self.workflow_tif_info = tif_info
        choose_tif = QPushButton("去工艺资源导入或标定抛光斑/TIF")
        choose_tif.clicked.connect(
            lambda: self._open_resource_from_plan(self.idx_dwell_spot))
        tif_box.addWidget(tif_info)
        tif_box.addWidget(choose_tif)
        scan_layout.addWidget(tif_group)

    def _refresh_plan_tif_info(self):
        spot = self._dwell_state.get("spot")
        if spot is None:
            self.workflow_tif_info.setText("尚未选择或计算当前方案的 TIF。")
        else:
            self.workflow_tif_info.setText(
                f"当前 TIF：{spot['kernel'].shape[1]}×{spot['kernel'].shape[0]} 点，"
                f"单位脉冲 {spot['impulse']:.5g}。")
