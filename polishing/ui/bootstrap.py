import sys


from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QStyleFactory
from PyQt5.QtGui import QFont


from polishing.io.license import get_hardware_id, activate, verify_license
from polishing.ui.window import MainWindow

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Microsoft YaHei", 9))

    # 启动授权检查
    ok, msg = verify_license()
    if not ok:
        # 弹出简单激活对话框
        dlg = QtWidgets.QDialog()
        dlg.setWindowTitle("软件激活")
        dlg.setFixedSize(500, 280)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(f"授权状态：{msg}"))

        hwid_row = QHBoxLayout()
        hwid_row.addWidget(QLabel("本机机器码："))
        hwid_edt = QLineEdit(get_hardware_id())
        hwid_edt.setReadOnly(True)
        hwid_edt.setStyleSheet("font-family:Consolas; color:#1a3f6f;")
        btn_c = QPushButton("复制")
        btn_c.setFixedWidth(46)
        btn_c.clicked.connect(lambda: QApplication.clipboard().setText(hwid_edt.text()))
        hwid_row.addWidget(hwid_edt); hwid_row.addWidget(btn_c)
        lay.addLayout(hwid_row)

        days_edt = QLineEdit("365")
        lay.addWidget(QLabel("授权天数："))
        lay.addWidget(days_edt)
        code_edt = QLineEdit()
        code_edt.setPlaceholderText("XXXXXXXX-XXXXXXXX-XXXXXXXX-XXXXXXXX")
        code_edt.setStyleSheet("font-family:Consolas;")
        lay.addWidget(QLabel("激活码："))
        lay.addWidget(code_edt)
        status_lbl = QLabel("")
        lay.addWidget(status_lbl)

        btn_row = QHBoxLayout()
        btn_ok  = QPushButton("立即激活")
        btn_skip = QPushButton("暂时跳过（试用）")
        btn_row.addWidget(btn_ok); btn_row.addWidget(btn_skip)
        lay.addLayout(btn_row)

        def do_act():
            try: d = int(days_edt.text())
            except: status_lbl.setText("天数无效"); return
            ok2, msg2 = activate(code_edt.text(), d)
            color = "#1a7a3c" if ok2 else "#c0392b"
            status_lbl.setText(msg2)
            status_lbl.setStyleSheet(f"color:{color};")
            if ok2:
                QtCore.QTimer.singleShot(1200, dlg.accept)

        btn_ok.clicked.connect(do_act)
        btn_skip.clicked.connect(dlg.accept)
        dlg.exec_()

    win = MainWindow()
    app.aboutToQuit.connect(win.preview.close_occ)
    win.show()
    sys.exit(app.exec_())
