from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from log import LOG

class About(QWidget):
    def __init__(self):
        super(About, self).__init__()
        text = QTextEdit()
        text.setPlainText("版本：\n    0.001\n" \
                            "提交：\n    0238e8ba2aaf4fcc90771ff5bc342c8d\n"\
                            "日期：\n    2020-06-07 20:22:39\n")
        text.setReadOnly(True)
        text.setFixedSize(400, 300)
        text.setStyleSheet('background-color: white')

        tool_frame = QFrame()
        tool_frame.setFrameShape(QFrame.StyledPanel)
        tool_frame.setFixedHeight(20)

        win_layout = QHBoxLayout()
        win_layout.addStretch(1)
        win_layout.addWidget(text)
        win_layout.addStretch(1)

        win_frame = QFrame()
        win_frame.setFrameShape(QFrame.StyledPanel)
        win_frame.setLineWidth(1)
        win_frame.setLayout(win_layout)

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(0,0,0,0)
        v_layout.setSpacing(0)
        v_layout.addWidget(tool_frame)
        v_layout.addWidget(win_frame)
        self.setLayout(v_layout)
