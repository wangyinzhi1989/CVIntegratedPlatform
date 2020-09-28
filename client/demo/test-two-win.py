from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QButtonGroup, QFrame, QToolButton, QStackedLayout,\
    QWidget, QStatusBar

import sys


class Demo(QWidget):
    def __init__(self):
        super().__init__()

        self.__setup_ui__()

    def __setup_ui__(self):
        self.setWindowTitle("测试")
        #窗口大小
        self.resize(1400,800)
        # 工具栏
        self.frame_tool = QFrame(self)
        self.frame_tool.setObjectName("frame_tool")
        self.frame_tool.setGeometry(0, 0, self.width(), 25)
        self.frame_tool.setStyleSheet("border-color: rgb(0, 0, 0);")
        self.frame_tool.setFrameShape(QFrame.Panel)
        self.frame_tool.setFrameShadow(QFrame.Raised)

        # 1.1 界面1按钮
        self.window1_btn = QToolButton(self.frame_tool)
        self.window1_btn.setCheckable(True)
        self.window1_btn.setText("window1")
        self.window1_btn.setObjectName("menu_btn")
        self.window1_btn.resize(100, 25)
        self.window1_btn.clicked.connect(self.click_window1)
        self.window1_btn.setAutoRaise(True)

        # 1.2 界面2按钮
        self.window2_btn = QToolButton(self.frame_tool)
        self.window2_btn.setCheckable(True)
        self.window2_btn.setText("window2")
        self.window2_btn.setObjectName("menu_btn")
        self.window2_btn.resize(100, 25)
        self.window2_btn.move(self.window1_btn.width(), 0)
        self.window2_btn.clicked.connect(self.click_window2)
        self.window2_btn.setAutoRaise(True)

        self.btn_group = QButtonGroup(self.frame_tool)
        self.btn_group.addButton(self.window1_btn, 1)
        self.btn_group.addButton(self.window2_btn, 2)

        # 2. 工作区域
        self.main_frame = QFrame(self)
        self.main_frame.setGeometry(0, 25, self.width(), self.height() - self.frame_tool.height())
        # self.main_frame.setStyleSheet("background-color: rgb(65, 95, 255)")

        # 创建堆叠布局
        self.stacked_layout = QStackedLayout(self.main_frame)

        # 第一个布局界面
        self.main_frame1 = QMainWindow()
        self.frame1_bar = QStatusBar()
        self.frame1_bar.setObjectName("frame1_bar")
        self.main_frame1.setStatusBar(self.frame1_bar)
        self.frame1_bar.showMessage("欢迎进入frame1")

        rom_frame = QFrame(self.main_frame1)
        rom_frame.setGeometry(0, 0 , self.width(), self.main_frame.height() - 25)
        rom_frame.setFrameShape(QFrame.Panel)
        rom_frame.setFrameShadow(QFrame.Raised)

        frame1_bar_frame = QFrame(self.main_frame1)
        frame1_bar_frame.setGeometry(0, self.main_frame.height(), self.width(), 25)

        # 第二个布局界面
        self.main_frame2 = QMainWindow()
        self.frame2_bar = QStatusBar()
        self.frame2_bar.setObjectName("frame2_bar")
        self.main_frame2.setStatusBar(self.frame2_bar)
        self.frame2_bar.showMessage("欢迎进入frame2")

        custom_frame = QFrame(self.main_frame2)
        custom_frame.setGeometry(0, 0 , self.width(), self.main_frame.height() - 25)
        custom_frame.setFrameShape(QFrame.Panel)
        custom_frame.setFrameShadow(QFrame.Raised)

        frame2_bar_frame = QFrame(self.main_frame2)
        frame2_bar_frame.setGeometry(0, self.main_frame.height(), self.width(), 25)

        # 把两个布局界面放进去
        self.stacked_layout.addWidget(self.main_frame1)
        self.stacked_layout.addWidget(self.main_frame2)

    def click_window1(self):
        if self.stacked_layout.currentIndex() != 0:
            self.stacked_layout.setCurrentIndex(0)
            self.frame1_bar.showMessage("欢迎进入frame1")

    def click_window2(self):
        if self.stacked_layout.currentIndex() != 1:
            self.stacked_layout.setCurrentIndex(1)
            self.frame2_bar.showMessage("欢迎进入frame2")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Demo()
    window.show()
    sys.exit(app.exec_())