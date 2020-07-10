from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from log import LOG
from .draw_chart import DrawChart
from .training_config_dialog import TrainingConfigDialog
import random

class Training(QWidget):
    def __init__(self):
        super(Training, self).__init__()
        self.setStyleSheet("QPushButton:hover{background-color:rgb(160,200,240)}" #光标移动到上面后的前景色
                            "QPushButton:pressed{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                            "QPushButton:checked{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                            )
        tool_bt_w = 80
        tool_bt_h = 20
        # 工具按钮设置
        config_bt = QPushButton("配置", self)
        config_bt.setFixedSize(tool_bt_w, tool_bt_h)
        config_bt.clicked.connect(self.openConfigDlg)
        start_bt = QPushButton("开始", self)
        start_bt.setFixedSize(tool_bt_w, tool_bt_h)
        start_bt.clicked.connect(self.startCB)
        stop_bt = QPushButton("停止", self)
        stop_bt.setFixedSize(tool_bt_w, tool_bt_h)
        stop_bt.clicked.connect(self.stopCB)
        self.resume_value = QComboBox()
        self.resume_value.setFixedSize(3*tool_bt_w, tool_bt_h)
        resume_bt = QPushButton("恢复", self)
        resume_bt.setFixedSize(tool_bt_w, tool_bt_h)
        resume_bt.clicked.connect(self.resumeCB)
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(config_bt)
        tool_layout.addWidget(start_bt)
        tool_layout.addWidget(stop_bt)
        tool_layout.addWidget(self.resume_value)
        tool_layout.addWidget(resume_bt)
        tool_layout.addStretch(1)
        tool_layout.setContentsMargins(0,0,0,0)
        tool_layout.setSpacing(0)

        # 损失值和验证结果绘制组件
        a = ['g','r','b','y']
        self.loss_chart = DrawChart(lines=a, left='损失值', bottom='次数')
        b= ['g']
        self.accuracy_chart = DrawChart(lines=b, left='准确度', bottom='次数')
        line_frame = QFrame()
        line_frame.setFrameShape(QFrame.VLine)
        line_frame.setFrameShadow(QFrame.Sunken)
        chart_layout = QHBoxLayout()
        chart_layout.setContentsMargins(0,0,0,0)
        chart_layout.setSpacing(0)
        chart_layout.addWidget(self.loss_chart)
        chart_layout.addWidget(line_frame)
        chart_layout.addWidget(self.accuracy_chart)
        chart_farme = QFrame()
        chart_farme.setFrameShape(QFrame.StyledPanel)
        chart_farme.setLayout(chart_layout)

        # 测试区域绘制
        self.test_model = QComboBox()
        self.test_model.setFixedSize(3*tool_bt_w, tool_bt_h)
        self.test_thresh = QDoubleSpinBox()
        self.test_thresh.setRange(0.3, 1.0)
        self.test_thresh.setSingleStep(0.01)
        test_bt = QPushButton("测试")
        test_bt.clicked.connect(self.testCB)
        save_bt = QPushButton("保存结果")
        save_bt.clicked.connect(self.saveCB)
        test_layout = QHBoxLayout()
        test_layout.setContentsMargins(50,5,0,5)
        test_layout.setSpacing(1)
        test_layout.addWidget(QLabel("模型:"))
        test_layout.addWidget(self.test_model)
        test_layout.addWidget(QLabel(""))
        test_layout.addWidget(QLabel("阈值:"))
        test_layout.addWidget(self.test_thresh)
        test_layout.addWidget(QLabel(""))
        test_layout.addWidget(test_bt)
        test_layout.addWidget(QLabel(""))
        test_layout.addWidget(save_bt)
        test_layout.addStretch(1)

        win_layout = QVBoxLayout()
        win_layout.setContentsMargins(0,0,0,0)
        win_layout.setSpacing(0)
        win_layout.addWidget(chart_farme)
        win_layout.addLayout(test_layout)
        
        win_frame = QFrame()
        win_frame.setFrameShape(QFrame.StyledPanel)
        win_frame.setLineWidth(1)
        win_frame.setLayout(win_layout)

        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0,0,0,0)
        vlayout.setSpacing(0)
        vlayout.addLayout(tool_layout)
        vlayout.addWidget(win_frame)
        self.setLayout(vlayout)
        self.timer_start()
        # self.x = []
        # self.a = {}
        # self.b = {}
        # self.a[0] = []
        # self.a[1] = []
        # self.a[2] = []
        # self.a[3] = []
        # self.b[0] = []
        
    def wheelEvent(self, ev):
        ev.ignore()

    def openConfigDlg(self):
        dlg = TrainingConfigDialog(parent=self)
        if QDialog.Accepted == dlg.exec_():
            pass
    
    def startCB(self):
        pass

    def stopCB(self):
        pass

    def resumeCB(self):
        pass

    def testCB(self):
        pass

    def saveCB(self):
        pass

    # 启动定时器 时间间隔秒
    def timer_start(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.get_training_info)
        self.timer.start(10000)

    def get_training_info(self):
        return 

        if len(self.x) == 0:
            self.x.append(10)
        else:
            the_x = self.x[-1]+10
            self.x.append(the_x)
        for i in range(4):
            self.a[i].append(random.uniform(0,100))
        self.b[0].append(random.uniform(0,100))

        self.loss_chart.refresh(self.x, self.a)
        self.accuracy_chart.refresh(self.x, self.b)
