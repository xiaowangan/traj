import os


from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QLabel, QFileDialog, QMessageBox, QPlainTextEdit, QAction, QActionGroup
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


from polishing.ui.widgets.RibbonWidget import RibbonWidget
from polishing.ui.widgets.RibbonButton import RibbonButton
from polishing.ui.widgets.Icons import get_icon
from polishing.ui.widgets.StyleSheets import get_stylesheet
from polishing.ui.viewers.occ import PreviewCanvas
from polishing.ui.features.shell import ControlPanel

CONSOLE_WIDTH = 420


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("光学曲面抛光工艺数据库")
        self.resize(1510, 840)
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
        self.preview.setMinimumSize(400, 300)   # 预览区再大的图也不会撑大窗口
        self.setCentralWidget(self.preview)

        # 右侧 DockWidget：控制台（参数输入）
        self.dock_ctrl = QDockWidget("控制台", self)
        # 右侧控制台与结果输出固定同宽：切换页面时宽度不再变化
        self.dock_ctrl.setFixedWidth(CONSOLE_WIDTH)
        self.dock_ctrl.setFeatures(QDockWidget.DockWidgetMovable)
        self.stacked_widget = ControlPanel(self)
        self.dock_ctrl.setWidget(self.stacked_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_ctrl)

        # 右侧 DockWidget：结果输出
        self.dock_term = QDockWidget("结果输出", self)
        self.dock_term.setFixedWidth(CONSOLE_WIDTH)
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
        tab_flow = self._ribbon.add_ribbon_tab("工艺流程")
        self._workflow_actions = {}
        self._workflow_action_group = QActionGroup(self)
        self._workflow_action_group.setExclusive(True)
        workflow = (
            ("import", "导入工件", "workflow_import", "建立工件、期望曲面和加工前实测状态"),
            ("evaluate", "面形初测", "workflow_evaluate", "检测加工前面形，设定加工目标"),
            ("plan", "制定方案", "workflow_plan", "选择工具、TIF和参数并形成候选方案"),
            ("predict", "效果预估", "workflow_predict", "比较方案并预测驻留、去除和加工后残差"),
            ("execute", "生成程序", "workflow_execute", "把冻结方案转换为 CNC/PATH 并记录执行"),
            ("remeasure", "加工复测", "workflow_remeasure", "导入加工后测量（入口预留）"),
            ("validate", "结果对比", "workflow_validate", "比较预测与实测（入口预留）"),
            ("archive", "结果存档", "workflow_archive", "保存完整案例证据（入口预留）"),
        )
        for stage, caption, icon, tip in workflow:
            pane = tab_flow.add_ribbon_pane(caption)
            action = self._make_action(
                caption, icon, tip,
                lambda _checked=False, current=stage: self._show_workflow(current))
            action.setCheckable(True)
            self._workflow_action_group.addAction(action)
            self._workflow_actions[stage] = action
            pane.add_ribbon_widget(RibbonButton(self, action, True))
        tab_flow.add_spacer()

        # 工艺资源：长期维护、可被多个案例复用；三个模块各自独立入口
        tab_resources = self._ribbon.add_ribbon_tab("工艺资源")
        resources = (
            ("抛光斑/TIF", "tool_tif", "导入打斑数据，标定去除函数（TIF）", "idx_dwell_spot"),
            ("工具与设备", "tool_resource", "记录抛光盘口径、材料、转速、压力和进给", "idx_dwell_initial"),
            ("覆盖轨迹", "tool_path", "独立的几何覆盖轨迹工具，不参与当前案例的求解或效果预估", "idx_surface"),
        )
        for caption, icon, tip, index_name in resources:
            pane = tab_resources.add_ribbon_pane(caption)
            pane.add_ribbon_widget(RibbonButton(self, self._make_action(
                caption, icon, tip,
                lambda _checked=False, key=index_name:
                    self.stacked_widget.setCurrentIndex(getattr(self.stacked_widget, key))), True))
        tab_resources.add_spacer()

        tab_tools = self._ribbon.add_ribbon_tab("工具")
        tools = (
            ("新建案例", "tool_new", "清空当前案例，从导入工件重新开始", self._new_case),
            ("保存案例", "tool_save", "保存当前加工案例（输入数据与冻结的预测记录）", self._save_case),
            ("打开案例", "tool_open", "重新打开已保存的加工案例", self._open_case),
            ("导入 STL", "tool_stl", "仅导入 STL 到三维预览", self._import_stl_model),
            ("授权管理", "tool_license", "查看授权状态或激活软件", self._show_license),
            ("退出", "tool_exit", "关闭软件", self.close),
        )
        for caption, icon, tip, slot in tools:
            pane = tab_tools.add_ribbon_pane(caption)
            pane.add_ribbon_widget(RibbonButton(
                self, self._make_action(caption, icon, tip, slot), True))
        tab_tools.add_spacer()
        self.set_workflow_stage("import")

    def _make_action(self, caption, icon_name, tip, slot):
        act = QAction(get_icon(icon_name), caption, self)
        act.setStatusTip(tip)
        act.triggered.connect(slot)
        act.setIconVisibleInMenu(True)
        return act

    # ── Ribbon 按钮槽函数 ───────────────────────────────────────────
    def _new_case(self):
        state = self.stacked_widget._dwell_state
        if any(state.get(key) is not None for key in ("model", "surface_raw", "spot", "evaluation")):
            answer = QMessageBox.question(
                self, "新建案例", "当前案例中未保存的内容会丢失。确定新建吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        old = self.stacked_widget
        self.stacked_widget = ControlPanel(self)
        self.dock_ctrl.setWidget(self.stacked_widget)
        old.deleteLater()
        self.set_workflow_stage("import")
        self.terminal_output.appendPlainText("[案例] 已新建空白案例。")

    def _save_case(self):
        self.stacked_widget._case_save()

    def _open_case(self):
        self.stacked_widget._case_open()

    def _show_license(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_license)

    def _show_workflow(self, stage):
        self.stacked_widget.setCurrentIndex(
            getattr(self.stacked_widget, f"idx_workflow_{stage}"))

    def clear_workflow_stage(self, title="工艺资源"):
        """进入工艺资源等非流程页面时，取消八个流程按钮的高亮。"""
        group = getattr(self, "_workflow_action_group", None)
        if group is not None:
            group.setExclusive(False)
            for action in group.actions():
                action.setChecked(False)
            group.setExclusive(True)
        if hasattr(self, "statusbar"):
            self.set_status(f"当前：{title}")

    def set_workflow_stage(self, stage):
        action = getattr(self, "_workflow_actions", {}).get(stage)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        title = {
            "import": "导入工件", "evaluate": "面形初测", "plan": "制定方案", "predict": "效果预估",
            "execute": "生成程序", "remeasure": "加工复测", "validate": "结果对比", "archive": "结果存档",
        }.get(stage, stage)
        if hasattr(self, "statusbar"):
            self.set_status(f"当前流程：{title}")

    def _show_surface(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_surface)

    def _show_dwell_initial(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_initial)

    def _show_dwell_model(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_model)

    def _show_dwell_surface(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_surface)

    def _show_dwell_spot(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_spot)

    def _show_dwell_solve(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_solve)

    def _show_dwell_cnc(self):
        self.stacked_widget.setCurrentIndex(self.stacked_widget.idx_dwell_cnc)

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
