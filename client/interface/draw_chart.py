from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from log import LOG
import pyqtgraph as pg

class DrawChart(QWidget):
    def __init__(self, lines=None, left='', bottom=''):
        super(DrawChart, self).__init__()
        self.lines = lines
        self.win = pg.GraphicsWindow()
        # 添加绘图项，禁用鼠标右键目录
        self.plot = self.win.addPlot(left=left, bottom=bottom, enableMenu=False)
        self.plot.showGrid(x=True, y=True)
        # 禁止鼠标左右移动
        self.plot.setMouseEnabled(x=False,y=False)

        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)
        layout.addWidget(self.win)
        self.setLayout(layout)

    def refresh(self, x, data_list):
        # 自适应大小
        self.plot.enableAutoRange()
        for i, data in data_list.items():
            self.plot.plot(x, data, pen=self.lines[i])

