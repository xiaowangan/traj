from PyQt5.QtWidgets import QLabel, QLineEdit, QHBoxLayout, QFrame

def lineedit_input(label_text, default_value=""):
    label = QLabel(label_text)
    edit = QLineEdit(str(default_value))
    row = QHBoxLayout()
    row.addWidget(label)
    row.addWidget(edit)
    return edit, row


def combox_input(layout, label_text, widget):
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    row.addWidget(widget)
    layout.addLayout(row)


def divider():
    frame = QFrame()
    frame.setFrameShape(QFrame.HLine)
    frame.setFrameShadow(QFrame.Sunken)
    return frame
