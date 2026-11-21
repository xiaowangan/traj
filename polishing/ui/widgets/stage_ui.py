"""控制台的通用外壳：页首（分页标签 + 白话说明 + 状态）、步骤卡片、前置条件提示和折叠区。

每个阶段页面都从用户角度回答三件事：我在做什么（说明）→ 先做哪步后做哪步（编号卡片）
→ 做到哪了（状态）。具体输入控件仍由各功能页自己提供。
"""
from __future__ import annotations
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

# 三种状态：未完成 / 已完成 / 已失效（上游改了，需要重做）
STATUS_TODO, STATUS_DONE, STATUS_STALE = "todo", "done", "stale"
_STATUS_MARK = {STATUS_TODO: "○ ", STATUS_DONE: "● ", STATUS_STALE: "! "}


def _repolish(*widgets):
    for widget in widgets:
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class StageHeader(QWidget):
    """页首：同一阶段内的分页标签 + 一段人话说明 + 当前案例状态。"""

    def __init__(self, tabs, current, intro, on_select, parent=None):
        super().__init__(parent)
        self.setObjectName("stageHeader")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        if len(tabs) > 1:
            row = QHBoxLayout()
            row.setSpacing(6)
            for caption, key in tabs:
                pill = QPushButton(caption)
                pill.setObjectName("pill")
                pill.setProperty("active", key == current)
                pill.setCursor(Qt.PointingHandCursor)
                if key != current:
                    pill.clicked.connect(lambda _c=False, k=key: on_select(k))
                row.addWidget(pill)
            row.addStretch()
            layout.addLayout(row)
        self.intro = QLabel(intro)
        self.intro.setObjectName("intro")
        self.intro.setWordWrap(True)
        self.intro.setTextFormat(Qt.RichText)
        layout.addWidget(self.intro)
        self.status = QLabel()
        self.status.setObjectName("stageStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_status(self, text, status):
        self.status.setText(_STATUS_MARK[status] + text)
        self.status.setProperty("status", status)
        _repolish(self.status)


class GateBanner(QFrame):
    """前置条件提示：缺什么、点哪里去补。前置齐全时自动隐藏。"""

    def __init__(self, on_select, parent=None):
        super().__init__(parent)
        self.setObjectName("gate")
        self._on_select = on_select
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(6)
        self._signature = None

    def set_missing(self, missing):
        """missing: [(显示名, 目标页面键)]；为空时隐藏。"""
        signature = tuple(missing)
        if signature == self._signature:
            return
        self._signature = signature
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.setVisible(bool(missing))
        if not missing:
            return
        title = QLabel("这一步还不能做，需要先完成：")
        title.setObjectName("gateTitle")
        self._layout.addWidget(title)
        for label, key in missing:
            button = QPushButton(f"去完成：{label}  →")
            button.setObjectName("gateLink")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=key: self._on_select(k))
            self._layout.addWidget(button, 0, Qt.AlignLeft)


class StepCard(QFrame):
    """总览页里的一个流程步骤：编号 + 标题 + 一句话 + 状态 (+ 附加内容) (+ 进入按钮)。"""

    def __init__(self, number, title, hint, button_text=None, on_open=None,
                 status_fn=None, extra=None, parent=None):
        super().__init__(parent)
        self.setObjectName("stepCard")
        self.status_fn = status_fn
        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        self.badge = QLabel(str(number))
        self.badge.setObjectName("stepBadge")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setFixedSize(24, 24)
        outer.addWidget(self.badge, 0, Qt.AlignTop)
        body = QVBoxLayout()
        body.setSpacing(3)
        name = QLabel(title)
        name.setObjectName("stepTitle")
        text = QLabel(hint)
        text.setObjectName("stepHint")
        text.setWordWrap(True)
        self.state = QLabel("待完成")
        self.state.setObjectName("stepState")
        body.addWidget(name)
        body.addWidget(text)
        body.addWidget(self.state)
        if extra is not None:
            body.addWidget(extra)
        if button_text:
            button = QPushButton(button_text)
            button.setObjectName("primaryBtn")
            button.clicked.connect(lambda _c=False: on_open())
            body.addWidget(button)
        outer.addLayout(body, 1)

    def refresh(self):
        self.state.setVisible(self.status_fn is not None)
        if self.status_fn is None:
            return
        status, text = self.status_fn()
        self.state.setText(text)
        self.badge.setProperty("status", status)
        self.state.setProperty("status", status)
        _repolish(self.badge, self.state)


def collapsible(layout, caption, content, expanded=False):
    """在 layout 里加一个“▸ 标题”开关按钮，点开/收起 content。返回开关按钮。"""
    toggle = QPushButton()
    toggle.setObjectName("foldToggle")
    toggle.setCheckable(True)
    toggle.setCursor(Qt.PointingHandCursor)

    def sync(checked):
        toggle.setText(("▾ " if checked else "▸ ") + caption)
        content.setVisible(checked)

    toggle.toggled.connect(sync)
    layout.addWidget(toggle, 0, Qt.AlignLeft)
    layout.addWidget(content)
    toggle.setChecked(expanded)
    sync(expanded)
    return toggle


def number_cards(layout):
    """给页面里直接排列的分组框按顺序加上步骤编号（标题已带编号或“第N步”的跳过）。"""
    count = 0
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if isinstance(widget, QGroupBox):
            count += 1
            title = widget.title()
            if not title or not (title[0].isdigit() or title.startswith("第")):
                widget.setTitle(f"{count}  {title}")
