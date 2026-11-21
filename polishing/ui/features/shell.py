
from PyQt5.QtWidgets import QApplication, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QSizePolicy


from polishing.ui.widgets.inputs import lineedit_input, divider
from polishing.ui.widgets.stage_ui import (
    STATUS_DONE, STATUS_STALE, STATUS_TODO, GateBanner, StageHeader, StepCard, number_cards)
from polishing.ui.widgets.StyleSheets import get_stylesheet
from polishing.io.license import get_hardware_id, activate, verify_license
from polishing.workflow.state import STAGE_TITLES, WorkflowContext
from .common import CommonPageMixin
from .import_geometry import GeometryImportMixin
from .import_measurement import MeasurementImportMixin
from .evaluation import EvaluationPageMixin
from .plan_tif import PlanTifMixin
from .plan_coverage import PlanCoverageMixin
from .resource_tif import TifResourceMixin
from .resource_trajectory import TrajectoryResourceMixin
from .resource_tool import ToolResourceMixin
from .prediction import PredictionPageMixin
from .execution import ExecutionPageMixin
from .case_page import CaseFileMixin

STAGE_DESCRIPTIONS = {
    "import": "建立工件、期望曲面和加工前实测状态。",
    "evaluate": "按统一检测标准检查加工前面形，为工艺目标提供依据。",
    "plan": "选择加工方式、工具和 TIF，设置覆盖路径，形成当前工件的候选方案。",
    "predict": "按方案中选定的加工方式试算加工结果，判定能否达标，确认后冻结方案。",
    "execute": "把已确认的方案转换为可执行轨迹、工具姿态、进给速度和 PATH 文件。",
    "remeasure": "导入加工后测量并与加工前数据配准。功能将在后续实现。",
    "validate": "比较预估与实测结果，计算偏差并判断是否达标。功能将在后续实现。",
    "archive": "冻结案例版本，保存原始数据、方案、程序与对比证据。功能将在后续实现。",
}

# 同一组页面之间的分页标签（标签文字，页面索引属性名）。
STAGE_TABS = {
    "import": (("总览", "idx_workflow_import"), ("期望曲面", "idx_dwell_model"),
               ("实测面形", "idx_dwell_surface")),
}
RESOURCE_TABS = (("抛光斑/TIF", "idx_dwell_spot"), ("工具与设备", "idx_dwell_initial"),
                 ("覆盖轨迹", "idx_surface"))

# 效果预估、生成程序不再单设总览页：点 Ribbon 直接进入按加工方式显示的功能页。
REUSED_STAGE_PAGES = {"predict": "idx_dwell_solve", "execute": "idx_dwell_cnc"}

# 页面首部的人话说明：先说“这一页帮你做什么”，再说“按什么顺序做”。
PAGE_INTROS = {
    "idx_workflow_import": "<b>先告诉软件两件事：想加工成什么样，现在是什么样。</b>"
                           "按下面三步依次完成，全部做完才能进入“面形初测”。",
    "idx_dwell_model": "<b>告诉软件你想把镜子加工成什么样。</b>"
                       "选口径 → 选面形并填参数 → 填离轴量 → 点“生成曲面模型”。",
    "idx_dwell_surface": "<b>导入镜子现在的实测面形。</b>"
                         "选择数据文件 → 按需要裁边、延拓、滤波 → 看 PV/RMS 是否合理。",
    "idx_workflow_evaluate": "<b>看看这块镜子现在差多少、要做到多好。</b>"
                             "定检测标准 → 设加工目标 → 计算检测结果，结论会给出加工方式建议。",
    "idx_workflow_plan": "<b>决定怎么加工。</b>"
                         "选加工方式 → 选工具并检查贴合 → 扫描驻留再选 TIF 和覆盖路径 → 生成候选方案。",
    "idx_dwell_solve": "<b>先在电脑里“试加工”一遍，看能不能达标。</b>"
                       "只显示方案中选定的加工方式：试算 → 看达标判定 → 确认方案后才能生成程序。",
    "idx_dwell_cnc": "<b>把确认过的方案变成机床能跑的文件。</b>"
                     "核对机床设置 → 生成轨迹（路径来自制定方案）→ 生成并保存 CNC。",
    "idx_workflow_remeasure": "<b>加工完后重新测一次。</b>导入加工后测量，并与加工前数据对齐。",
    "idx_workflow_validate": "<b>看看预估准不准、加工有没有达标。</b>比较预估与实测，计算偏差。",
    "idx_workflow_archive": "<b>把整个案例存档。</b>冻结版本，保存数据、方案、程序与对比证据。",
    "idx_dwell_spot": "<b>标定抛光斑（TIF）：工具在一点停留时能去掉多少材料。</b>"
                      "选打斑数据 → 填采斑时长和对中偏置 → 计算去除函数，供制定方案时选用。",
    "idx_dwell_initial": "<b>记录抛光盘和设备参数。</b>"
                         "这里填一次，“制定方案”里的“工具与贴合”和方案记录都直接取用。",
    "idx_surface": "<b>独立的几何覆盖轨迹工具。</b>"
                   "按曲面类型生成通用覆盖轨迹，不参与当前案例的求解或效果预估。",
}

# 页面前置条件（软锁）：(状态键, 显示名, 去完成的页面)。缺少时页面能看、不能操作。
PAGE_GATES = {
    "idx_dwell_surface": (("model", "期望曲面", "idx_dwell_model"),),
    "idx_workflow_evaluate": (("model", "期望曲面", "idx_dwell_model"),
                              ("surface", "加工前实测面形", "idx_dwell_surface")),
    "idx_workflow_plan": (("evaluation", "加工前面形检测", "idx_workflow_evaluate"),),
    "idx_dwell_solve": (("process_plan", "候选方案（制定方案最后一步）", "idx_workflow_plan"),),
    "idx_dwell_cnc": (("frozen_plan", "确认方案（效果预估最后一步）", "idx_dwell_solve"),),
}


class WorkflowShellMixin:
    def _page_layout(self, index):
        widget = self.widget(index)
        content = widget.widget() if hasattr(widget, "widget") and widget.widget() else widget
        return content.layout()

    def _make_stage_header(self, stage, page_key, tabs=None):
        header = StageHeader(
            STAGE_TABS.get(stage, ()) if tabs is None else tabs, page_key, PAGE_INTROS[page_key],
            lambda key: self.setCurrentIndex(getattr(self, key)))
        if stage is None:
            header.status.hide()
        else:
            self._stage_headers.append((stage, header))
        return header

    def _stage_status(self, stage):
        """返回 (状态, 给用户看的文字)。"""
        context = self._workflow_context
        frozen = self._dwell_state.get("frozen_plan") or {}
        if stage == "execute" and frozen.get("process_type") == "conformal_vibration":
            return STATUS_TODO, "当前方案是共形振动：无需生成执行文件，按方案参数在设备上手动设置"
        if context.stage_status(stage) == "已完成":
            return STATUS_DONE, "当前案例状态：已完成"
        if context.stage_is_stale(stage):
            return STATUS_STALE, "已失效：前面步骤的参数改过了，这一步需要重新做"
        return STATUS_TODO, "当前案例状态：待完成"

    def _key_status(self, key, ok=None):
        state = self._dwell_state
        done = state.get(key) is not None if ok is None else ok(state.get(key))
        if done:
            return STATUS_DONE, "已完成"
        if key in state.invalidated:
            return STATUS_STALE, "已失效：前面的参数改过了，需要重新做"
        return STATUS_TODO, "待完成"

    def _workflow_page(self, stage, placeholder=False):
        title = STAGE_TITLES[stage]
        page, layout = self._dwell_scroll_page(title, STAGE_DESCRIPTIONS[stage])
        page.workflow_stage = stage
        layout.addWidget(self._make_stage_header(stage, f"idx_workflow_{stage}"))
        if stage == "import":
            self._build_import_steps(layout)
        elif stage == "evaluate":
            self._build_evaluation_controls(layout)
        elif stage == "plan":
            self._build_plan_controls(layout)
        if placeholder:
            pending = QLabel(
                "入口和页面位置已经建立。本阶段暂不执行数据写入或计算，"
                "后续实现时会继续使用当前案例状态，不另建孤立流程。")
            pending.setWordWrap(True)
            pending.setStyleSheet(
                "background:#fff8e8; color:#73510d; padding:8px; border:1px solid #e4c276;")
            layout.addWidget(pending)
        number_cards(layout)
        layout.addStretch()
        self._workflow_pages[stage] = page
        return page

    def _build_import_steps(self, layout):
        go = lambda key: (lambda: self.setCurrentIndex(getattr(self, key)))
        self.workflow_alignment_label = QLabel("尚缺期望曲面或实测面形，两步完成后自动检查。")
        self.workflow_alignment_label.setWordWrap(True)
        self.workflow_alignment_label.setStyleSheet("color:#52657a; font-size:11px;")
        steps = (
            StepCard(1, "期望曲面建模", "设置加工完成后希望达到的曲面、离轴量和口径。",
                     "去设置", go("idx_dwell_model"), lambda: self._key_status("model")),
            StepCard(2, "加工前实测面形", "读取并处理加工前的测量数据（需要先完成第 1 步）。",
                     "去导入", go("idx_dwell_surface"), lambda: self._key_status("surface")),
            StepCard(3, "对齐检查", "自动核对实测数据与期望曲面是否对得上（覆盖率、中心偏移）。",
                     status_fn=lambda: self._key_status(
                         "alignment", ok=lambda report: bool(report and report.get("ok"))),
                     extra=self.workflow_alignment_label),
        )
        for card in steps:
            self._stage_cards.append(card)
            layout.addWidget(card)

    def _install_page_header(self, index_name, stage, tabs=None):
        layout = self._page_layout(getattr(self, index_name))
        layout.insertWidget(0, self._make_stage_header(stage, index_name, tabs))
        number_cards(layout)

    def _install_detail_headers(self):
        """给各功能页顶部补上分页标签与说明，并给分组框编号。"""
        for stage, tabs in STAGE_TABS.items():
            for _caption, key in tabs:
                if not key.startswith("idx_workflow_"):
                    self._install_page_header(key, stage)
        for stage, key in REUSED_STAGE_PAGES.items():
            self._install_page_header(key, stage, tabs=())
        for _caption, key in RESOURCE_TABS:
            self._install_page_header(key, None, tabs=RESOURCE_TABS)

    @staticmethod
    def _layout_widgets(layout):
        """布局里直接放置的控件（嵌套的行布局展开到其中的控件）。"""
        widgets = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                widgets.append(item.widget())
            elif item.layout() is not None:
                widgets.extend(WorkflowShellMixin._layout_widgets(item.layout()))
        return widgets

    def _install_gates(self):
        """在页首下方插入前置条件提示；前置不满足时，页首以下的内容原地变灰。
        注意：不要把这些内容搬进新建的容器——那样会让卡片高度不随内容更新、互相重叠。"""
        self._page_gates = []
        for key, requirements in PAGE_GATES.items():
            layout = self._page_layout(getattr(self, key))
            widgets = self._layout_widgets(layout)[1:]   # 第 0 项是页首
            gate = GateBanner(lambda target: self.setCurrentIndex(getattr(self, target)))
            gate.hide()
            layout.insertWidget(1, gate)
            self._page_gates.append((requirements, gate, widgets))

    def _refresh_gates(self):
        state = self._dwell_state
        for requirements, gate, widgets in getattr(self, "_page_gates", ()):
            missing = [(label, target) for key, label, target in requirements if state.get(key) is None]
            gate.set_missing(missing)
            for widget in widgets:
                widget.setEnabled(not missing)

    def _refresh_workflow_statuses(self):
        self._refresh_gates()
        self._refresh_plan_readiness()
        self._refresh_predict_view()
        self._refresh_execute_view()
        self._dwell_refresh_flow_buttons()
        for stage, header in self._stage_headers:
            header.set_status(*reversed(self._stage_status(stage)))
        for card in self._stage_cards:
            card.refresh()

    def _build_workflow_pages(self):
        for stage in STAGE_TITLES:
            if stage in REUSED_STAGE_PAGES:
                index = getattr(self, REUSED_STAGE_PAGES[stage])
                page = self.widget(index)
                page.workflow_stage = stage
                self._workflow_pages[stage] = page
            else:
                index = self.count()
                self.addWidget(self._workflow_page(
                    stage, placeholder=stage in ("remeasure", "validate", "archive")))
            setattr(self, f"idx_workflow_{stage}", index)
            self._page_stages[index] = stage
        self._install_detail_headers()
        self._install_gates()
        self._refresh_workflow_statuses()


class ControlPanel(
        TrajectoryResourceMixin, CommonPageMixin, ToolResourceMixin,
        GeometryImportMixin, MeasurementImportMixin, TifResourceMixin,
        PredictionPageMixin, ExecutionPageMixin, EvaluationPageMixin, CaseFileMixin,
        PlanTifMixin, PlanCoverageMixin, WorkflowShellMixin, QStackedWidget):
    """
    仿照师兄软件：QDockWidget 里放 QStackedWidget，
    每个功能对应一个 page，Ribbon 按钮切换页面。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(get_stylesheet("console"))
        self._main = parent          # MainWindow 引用，用于访问 preview/statusbar
        self._surface_control_pairs = []
        self._init_dwell_state()
        self._workflow_context = WorkflowContext(self._dwell_state)
        self._workflow_pages = {}
        self._stage_headers = []
        self._stage_cards = []
        self._page_stages = {}
        self._flow_next_buttons = []
        self._resource_flow_widgets = []   # 工艺资源页里只在方案流程内显示的按钮
        self._resource_entry_from_plan = False
        self._resource_in_flow = False
        self._last_index = None

        # 记录各页面索引
        self.idx_blank   = self.count()
        self.addWidget(self._build_blank_page())    # page 0：空白（初始状态）

        self.idx_license = self.count()
        self.addWidget(self._build_license_page())

        # 驻留时间：由 appCNCFinishingV6.mlapp 的六个工作页迁移而来。
        # 工艺资源 ·「工具与设备」先建：制定方案第 2 步要引用它的输入框。
        self.idx_dwell_initial = self.count()
        self.addWidget(self._build_dwell_initial_page())
        self.idx_dwell_model = self.count()
        self.addWidget(self._build_dwell_model_page())
        self.idx_dwell_surface = self.count()
        self.addWidget(self._build_dwell_surface_page())
        self.idx_dwell_spot = self.count()
        self.addWidget(self._build_dwell_spot_page())
        self.idx_dwell_solve = self.count()
        self.addWidget(self._build_dwell_solve_page())
        self.idx_dwell_cnc = self.count()
        self.addWidget(self._build_dwell_cnc_page())

        # 曲面轨迹：统一入口页（下拉选择 + 子页面）
        self.idx_surface = self.count()
        surface_page = self._build_surface_selector_page()
        surface_page.dwell_banner_text = " 工艺资源 | 独立几何覆盖轨迹工具（不参与扫描驻留候选的求解或预测）。"
        self.addWidget(surface_page)

        self._page_stages.update({
            self.idx_dwell_model: "import",
            self.idx_dwell_surface: "import",
            self.idx_dwell_solve: "predict",
            self.idx_dwell_cnc: "execute",
        })
        self._build_workflow_pages()

        # 当前缓存的轨迹点和参数
        self._points = []
        self._params = {}
        self._last_is_surface = False

        # 切换页面时同步左侧顶部介绍横幅（横幅文本由各页面自身携带）。
        self.currentChanged.connect(self._sync_dwell_banner)
        self._flatten_containers(self)
        self.setCurrentIndex(self.idx_workflow_import)
        self._sync_dwell_banner(self.currentIndex())
        self._dwell_refresh_flow_buttons()

    def _sync_dwell_banner(self, index):
        """驻留时间页面在左侧顶部显示介绍横条，其余页面隐藏。"""
        widget = self.widget(index)
        stage = self._page_stages.get(index)
        self._update_resource_flow(index)
        if stage:
            self._workflow_context.set_stage(stage)
            self._refresh_workflow_statuses()
            if hasattr(self._main, "set_workflow_stage"):
                self._main.set_workflow_stage(stage)
        elif hasattr(self._main, "clear_workflow_stage"):
            self._main.clear_workflow_stage()
        text = getattr(widget, "dwell_banner_text", None)
        if text:
            self._main.preview.set_dwell_banner(text)
        else:
            self._main.preview.clear_dwell_banner()
        try:
            self._show_page_preview(index, stage)
        except Exception as exc:  # 预览失败只影响左侧显示，不能让切页中断
            self._main.preview.show_placeholder(f"左侧预览未能显示：{exc}")
        self._flatten_containers(widget)

    def _flatten_containers(self, root):
        """纯容器控件（装一行输入框的 QWidget）去掉全局底色，避免卡片里出现深色底纹。"""
        for child in root.findChildren(QWidget):
            if type(child) is QWidget and not child.objectName():
                child.setObjectName("plainBox")

    def _show_page_preview(self, index, stage):
        """左侧预览始终跟着当前页走：有结果就显示本页相关的图，没有就明确提示。"""
        state, preview = self._dwell_state, self._main.preview
        has_surface = state.get("surface_raw") is not None and state.get("surface") is not None
        if index == self.idx_dwell_initial:
            preview.show_placeholder("这里只记录抛光盘和设备参数。贴合检查在“制定方案”第 2 步进行，曲线在那里显示。")
        elif index == self.idx_dwell_spot:
            if state.get("spot_before") is not None:
                self._dwell_refresh_spot_preview()
            else:
                preview.show_placeholder("尚未导入抛光斑数据。在右侧选择数据文件后在这里显示。")
        elif index == self.idx_dwell_model:
            self._preview_model("尚未生成期望曲面。按右侧步骤填好参数，点“生成曲面模型”后在这里显示。")
        elif index == self.idx_dwell_surface:
            self._preview_surface("尚未导入加工前实测面形。在右侧选择数据文件后在这里显示。")
        elif stage == "import":
            if has_surface:
                self._dwell_refresh_surface_preview()
            else:
                self._preview_model("导入工件尚未开始。先在右侧设置期望曲面，再导入实测面形。")
        elif stage == "evaluate":
            if state.get("evaluation") is not None:
                preview.plot_dwell_evaluation(state["evaluation"])
            else:
                self._preview_surface("尚未检测。确认检测标准后点“计算检测结果”，结果在这里显示。")
        elif stage == "plan":
            if state.get("coverage_preview") is not None and getattr(self, "_coverage_view", None):
                model, paths, pattern = self._coverage_view
                preview.plot_dwell_coverage(model, paths, pattern=pattern)
            elif state.get("disk_fit") is not None:
                self._dwell_refresh_fit_preview()
            else:
                preview.show_placeholder("尚无覆盖轨迹预览。确认加工方式并设好覆盖参数后点“轨迹预览”。")
        elif stage == "predict":
            if state.get("solution") is not None:
                self._dwell_refresh_solve_preview()
            elif state.get("assessment") is not None:
                preview.plot_dwell_assessment(state["assessment"])
            else:
                preview.show_placeholder("尚无预测结果。完成制定方案后在“效果预估”页求解，结果在这里显示。")
        elif stage == "execute":
            if state.get("trajectory") is not None and getattr(self, "_trajectory_view", None):
                preview.plot_surface(*self._trajectory_view)
            else:
                frozen = state.get("frozen_plan") or {}
                preview.show_placeholder(
                    "共形振动方案没有轨迹文件，按方案参数在设备上手动设置。"
                    if frozen.get("process_type") == "conformal_vibration" else
                    "尚未生成轨迹。确认方案后在右侧生成轨迹与 CNC/PATH。")
        elif stage in ("remeasure", "validate", "archive"):
            preview.show_placeholder(STAGE_DESCRIPTIONS[stage])
        else:
            preview.hide_dwell_sub_bar()

    def _preview_model(self, empty_text):
        model = self._dwell_state.get("model")
        if model is None:
            self._main.preview.show_placeholder(empty_text)
            return
        try:
            self._main.preview.plot_dwell_model(
                model, getattr(self, "_model_view_name", "期望曲面"))
        except Exception:
            self._main.preview.show_placeholder(empty_text)

    def _preview_surface(self, empty_text):
        if (self._dwell_state.get("surface_raw") is not None and
                self._dwell_state.get("surface") is not None):
            self._dwell_refresh_surface_preview()
        else:
            self._main.preview.show_placeholder(empty_text)

    # ── 共用：保存 TXT ──────────────────────────────────────────────
    def _build_blank_page(self):
        page = QWidget()
        page.setAutoFillBackground(True)
        page.setStyleSheet("background-color: #dfe9f5;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        blank = QLabel("")
        blank.setStyleSheet("background-color: #dfe9f5;")
        blank.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(blank)
        return page

    def _build_license_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("软件授权管理")
        g   = QVBoxLayout(grp)

        # 机器码
        hwid_row = QHBoxLayout()
        hwid_row.addWidget(QLabel("本机机器码："))
        self.lic_edt_hwid = QLineEdit(get_hardware_id())
        self.lic_edt_hwid.setReadOnly(True)
        self.lic_edt_hwid.setStyleSheet(
            "font-family:Consolas; color:#1a3f6f; background:#e8f0fa;")
        hwid_row.addWidget(self.lic_edt_hwid)
        btn_copy = QPushButton("复制")
        btn_copy.setFixedWidth(46)
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self.lic_edt_hwid.text()))
        hwid_row.addWidget(btn_copy)
        g.addLayout(hwid_row)
        g.addWidget(QLabel("  将机器码发送给供应商以获取激活码"))

        self.lic_edt_days, row_d = lineedit_input("授权天数：", "365")
        g.addLayout(row_d)

        self.lic_edt_code, row_c = lineedit_input("激活码：")
        self.lic_edt_code.setPlaceholderText("XXXXXXXX-XXXXXXXX-XXXXXXXX-XXXXXXXX")
        self.lic_edt_code.setStyleSheet("font-family:Consolas;")
        g.addLayout(row_c)

        self.lic_btn_act = QPushButton("立即激活")
        g.addWidget(self.lic_btn_act)
        g.addWidget(divider())

        self.lic_lbl_status = QLabel("（点击'查询状态'刷新）")
        self.lic_lbl_status.setWordWrap(True)
        g.addWidget(self.lic_lbl_status)
        btn_q = QPushButton("查询当前授权状态")
        g.addWidget(btn_q)

        layout.addWidget(grp)
        layout.addStretch()

        self.lic_btn_act.clicked.connect(self._do_activate)
        btn_q.clicked.connect(self._do_query)
        return page

    def _do_activate(self):
        code = self.lic_edt_code.text().strip()
        if not code:
            QMessageBox.warning(self._main, "提示", "请输入激活码"); return
        try:
            days = int(self.lic_edt_days.text())
            assert days > 0
        except:
            QMessageBox.warning(self._main, "提示", "授权天数必须为正整数"); return
        ok, msg = activate(code, days)
        color = "#1a7a3c" if ok else "#c0392b"
        mark  = "✔" if ok else "✘"
        self.lic_lbl_status.setText(f"{mark} {msg}")
        self.lic_lbl_status.setStyleSheet(f"color:{color}; font-size:12px;")
        if ok:
            QMessageBox.information(self._main, "激活成功", msg)
        else:
            QMessageBox.warning(self._main, "激活失败", msg)

    def _do_query(self):
        ok, msg = verify_license()
        color = "#1a7a3c" if ok else "#c0392b"
        mark  = "✔" if ok else "✘"
        self.lic_lbl_status.setText(f"{mark} {msg}")
        self.lic_lbl_status.setStyleSheet(f"color:{color}; font-size:12px;")
