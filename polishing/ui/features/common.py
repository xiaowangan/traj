"""Migrated common page behavior; existing layout preserved."""
from __future__ import annotations
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from polishing.ui.widgets.inputs import lineedit_input
from polishing.workflow.state import create_dwell_state

class CommonPageMixin:
    def _init_dwell_state(self):
        self._dwell_files = {}
        self._dwell_state = create_dwell_state()

    @staticmethod
    def _dwell_float(edit, name, minimum=None, allow_equal=True):
        try:
            value = float(edit.text())
        except ValueError as exc:
            raise ValueError(f"参数“{name}”输入无效") from exc
        if minimum is not None:
            invalid = value < minimum if allow_equal else value <= minimum
            if invalid:
                relation = "不小于" if allow_equal else "大于"
                raise ValueError(f"参数“{name}”必须{relation} {minimum}")
        return value

    @staticmethod
    def _dwell_coefficients(edit, name, limit=None):
        text = edit.text().strip()
        if not text:
            return []
        try:
            values = [float(item) for item in text.replace(";", ",").replace(" ", ",").split(",")
                      if item.strip()]
        except ValueError as exc:
            raise ValueError(f"参数“{name}”必须是用逗号分隔的数字") from exc
        if limit is not None and len(values) > limit:
            raise ValueError(f"参数“{name}”最多允许 {limit} 项")
        return values

    def _dwell_scroll_page(self, title, subtitle):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget(); content.setObjectName("panelContent"); scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(6); layout.setContentsMargins(8, 8, 8, 8)
        # 功能介绍不再放在右侧控制台，而是仿照轨迹规划 "Trajectory | ..." 覆盖条，
        # 作为独立横条显示在左侧预览区最上方；横幅文本随页面携带，切换页面时同步。
        scroll.dwell_banner_text = f" {title} | {subtitle}"
        return scroll, layout

    def _dwell_add_fields(self, layout, fields, readonly=()):
        for attr, label, default in fields:
            edit, row = lineedit_input(label, default)
            if attr in readonly:
                edit.setReadOnly(True)
                edit.setStyleSheet("background:#e8f0fa; color:#10243f;")
            setattr(self, attr, edit); layout.addLayout(row)

    def _dwell_add_next_button(self, layout, attr, caption, target_index_name,
                               required=(), tooltip=""):
        """Add an in-page workflow transition guarded by current case products."""
        button = QPushButton(caption)
        button.setObjectName(attr)
        button.setMaximumWidth(280)
        button.setFixedHeight(32)
        button.setStyleSheet(
            "QPushButton {"
            "font-size:12px; font-weight:600; padding:4px 12px; margin-top:4px;"
            "border:1px solid #b8c7d8; border-radius:4px;"
            "background:#f5f8fc; color:#34516f;"
            "}"
            "QPushButton:hover:enabled {background:#e6eef7; border-color:#829bb5;}"
            "QPushButton:pressed:enabled {background:#d9e5f1;}"
            "QPushButton:disabled {background:#edf1f5; color:#9aa7b4; border-color:#d9e0e7;}")
        if tooltip:
            button.setToolTip(tooltip)
        button.clicked.connect(
            lambda _checked=False, target=target_index_name: self._dwell_go_to(target))
        setattr(self, attr, button)
        layout.addWidget(button, 0, Qt.AlignRight)
        if not hasattr(self, "_flow_next_buttons"):
            self._flow_next_buttons = []
        self._flow_next_buttons.append((button, tuple(required)))
        self._dwell_refresh_flow_buttons()
        return button

    def _dwell_go_to(self, target_index_name):
        self.setCurrentIndex(getattr(self, target_index_name))

    def _dwell_refresh_flow_buttons(self):
        if hasattr(self, "_refresh_gates"):
            self._refresh_gates()
        for button, required in getattr(self, "_flow_next_buttons", ()):
            ready = all(self._dwell_state.get(key) is not None for key in required)
            button.setEnabled(ready)

    def _dwell_report(self, message, label=None, error=False):
        if label is not None:
            label.setText(("✘ " if error else "✔ ") + message)
            label.setStyleSheet(
                f"color:{'#c0392b' if error else '#1a7a3c'}; font-size:11px;")
        self._main.statusbar.showMessage(message)
        self._main.set_status(message)
        self._main.terminal_output.appendPlainText("[驻留时间] " + message)
        self._dwell_refresh_flow_buttons()

    def _dwell_error(self, title, error, label=None):
        self._dwell_report(str(error), label, error=True)
        QMessageBox.warning(self._main, title, str(error))

    @staticmethod
    def _add_inline_combo(layout, caption, combo):
        row = QHBoxLayout()
        label = QLabel(caption)
        label.setMinimumWidth(84)
        row.addWidget(label)
        row.addWidget(combo, 1)
        layout.addLayout(row)

