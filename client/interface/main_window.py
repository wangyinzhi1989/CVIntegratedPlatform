from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from ctypes.wintypes import *
from .annotation import Annotation
from .training import Training
from .library import Library
from .about import About
from res.res import *
from log import LOG

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 无窗体窗口
        self.setWindowFlags(Qt.FramelessWindowHint)
        # 初始化窗口大小
        self.resize(1024,800)
        # 窗口最小大小
        self.setMinimumSize(1024,800)
        #self.setStyleSheet('background-color: rgb(227,244,244)')
        self.setStyleSheet('background-color: rgb(232,249,248)')
        
        self.top_bts = []
        self.buildLayout()

        self.pages = []
        self.pages.append(Annotation())
        self.pages.append(Training())
        self.pages.append(Library())
        self.pages.append(About())
        self.centre_layout.addWidget(self.pages[0])
        self.centre_layout.addWidget(self.pages[1])
        self.centre_layout.addWidget(self.pages[2])
        self.centre_layout.addWidget(self.pages[3])
        
        # 默认显示标注界面
        self.current = 0
        self.centre_layout.setCurrentIndex(self.current)
        self.anno_bt.setChecked(True)

    def mousePressEvent(self, event):
        ''' 重写鼠标按下事件，以实现窗口的移动 '''
        if event.buttons() == Qt.LeftButton:
            self.last_position = event.globalPos()-self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        ''' 重写鼠标移动事件，以实现窗口的移动 '''
        try:
            if event.buttons() and Qt.LeftButton:
                # move相对于鼠标按下时窗口位置的偏移大小
                self.move(event.globalPos()-self.last_position)
                event.accept()
        except AttributeError:
            pass

    def nativeEvent(self, eventType, message):
        ''' 重写native事件，以实现窗口的放缩 '''
        result = 0
        msg2 = ctypes.wintypes.MSG.from_address(message.__int__())
        # 捕获改变窗口大小标志的范围，即鼠标在边框向内的第1-5个像素出现改变窗口大小标志
        minV,maxV = 1,5
        if msg2.message == 0x0084:
            xPos = (msg2.lParam & 0xffff) - self.frameGeometry().x()
            yPos = (msg2.lParam >> 16) - self.frameGeometry().y()

            if(xPos > minV and xPos < maxV):
                result = 10
            elif(xPos > (self.width() - maxV) and xPos < (self.width() - minV)):
                result = 11
            elif(yPos > minV and yPos < maxV):
                result = 12
            elif(yPos > (self.height() - maxV) and yPos < (self.height() - minV)):
                result = 15
            elif(xPos > minV and xPos < maxV and yPos > minV and yPos < maxV):
                result = 13
            elif(xPos > (self.width() - maxV) and xPos < (self.width() - minV) and yPos > minV and yPos < maxV):
                result = 14
            elif(xPos > minV and xPos < maxV and yPos > (self.height() - maxV) and yPos < (self.height() - minV)):
                result = 16
            elif(xPos > (self.width() - maxV) and xPos < (self.width() - minV) and yPos > (self.height() - maxV) and yPos < (self.height() - minV)):
                result = 17
            else:
                return (False,2)
            return (True,result)
        ret= QWidget.nativeEvent(self,eventType,message)
        return ret

    def buildLayout(self):
        wid_h = 30
        wid_w = 80
        # 顶部容器，使用顶部水平布局对象布局
        top_back_widget = QWidget()
        # logo
        logo_img = QPixmap(':/icons/logo')
        logo_label = QLabel()
        logo_label.setFixedSize(wid_h,wid_h)
        logo_label.setPixmap(logo_img)
        # title
        app_title = QLabel()
        #app_title.setText("深度学习一体化平台")
        app_title.setText("tt")
        app_title.setStyleSheet('font-size:20px;padding-left:1px;')
        app_title.setFixedSize(210,wid_h)
        # 数据标注button
        #anno_bt = QPushButton("数据标注", top_back_widget)
        self.anno_bt = QPushButton("dd", top_back_widget)
        self.anno_bt.setFlat(True)
        self.anno_bt.setFixedSize(wid_w,wid_h)
        self.anno_bt.clicked.connect(lambda:self.switchPage(0))
        self.anno_bt.setCheckable(True)
        self.anno_bt.setAutoExclusive(True)
        self.top_bts.append(self.anno_bt)
        # 模型训练button
        #train_bt = QPushButton("模型训练", top_back_widget)
        train_bt = QPushButton("mm", top_back_widget)
        train_bt.setFlat(True)
        train_bt.setFixedSize(wid_w,wid_h)
        train_bt.clicked.connect(lambda:self.switchPage(1))
        train_bt.setCheckable(True)
        train_bt.setAutoExclusive(True)
        self.top_bts.append(train_bt)
        # 工程打包button
        #lib_bt = QPushButton("工程打包", top_back_widget)
        lib_bt = QPushButton("pp", top_back_widget)
        lib_bt.setFlat(True)
        lib_bt.setFixedSize(wid_w,wid_h)
        lib_bt.clicked.connect(lambda:self.switchPage(2))
        lib_bt.setCheckable(True)
        lib_bt.setAutoExclusive(True)
        self.top_bts.append(lib_bt)
        # 关于我们button
        #about_bt = QPushButton("关于我们", top_back_widget)
        about_bt = QPushButton("aa", top_back_widget)
        about_bt.setFlat(True)
        about_bt.setFixedSize(wid_w,wid_h)
        about_bt.clicked.connect(lambda:self.switchPage(3))
        about_bt.setCheckable(True)
        about_bt.setAutoExclusive(True)
        self.top_bts.append(about_bt)
        # 最小化button
        min_bt = QPushButton("-")
        min_bt.setFlat(True)
        min_bt.setFixedSize(wid_h,wid_h)
        min_bt.clicked.connect(self.showMinimized)
        # 最大化button
        max_bt = QPushButton("□")
        max_bt.setFlat(True)
        max_bt.setFixedSize(wid_h,wid_h)
        max_bt.clicked.connect(lambda: self.showNormal() if self.isMaximized()  else self.showMaximized())
        # 关闭button
        close_bt = QPushButton("×")
        close_bt.setFlat(True)
        close_bt.setFixedSize(wid_h,wid_h)
        close_bt.clicked.connect(self.close)
        # 顶部水平布局
        top_hBox = QHBoxLayout()
        top_hBox.setContentsMargins(0,0,0,0)
        top_hBox.setSpacing(0)
        top_hBox.addWidget(logo_label)
        top_hBox.addWidget(app_title)
        top_hBox.addWidget(self.anno_bt)
        top_hBox.addWidget(train_bt)
        top_hBox.addWidget(lib_bt)
        top_hBox.addWidget(about_bt)
        top_hBox.addStretch(1)
        top_hBox.addWidget(min_bt)
        top_hBox.addWidget(max_bt)
        top_hBox.addWidget(close_bt)

        top_back_widget.setStyleSheet('QWidget{background-color: rgb(0,102,204);font-weight: bold;font-family:"宋体";font-size:16px;color:white;}'
                                 "QPushButton:hover{background-color:rgb(160,200,240)}" #光标移动到上面后的前景色
                                 "QPushButton{ border-top-left-radius:3px;border-top-right-radius:3px}"  #圆角半径
                                 "QPushButton:pressed{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                                 "QPushButton:checked{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                                 )
        top_back_widget.setFixedHeight(wid_h)
        top_back_widget.setLayout(top_hBox)

        # 中心容器
        self.centre_layout = QStackedWidget()
        #self.centre_widget = QStackedWidget()
        #centre_widget.setLayout(self.centre_layout)
        #self.centre_widget.setStyleSheet('background-color: rgb(227,244,244)')
        #self.centre_widget.setStyleSheet('border:1px solid red;background-color: rgb(227,244,244)')

        #状态栏
        self.state_label1 = QLabel("")
        self.state_label1.setFixedHeight(20)

        self.state_label2 = QLabel("")
        self.state_label2.setFixedHeight(20)
        #self.state_label2.setAlignment(Qt.AlignVCenter) 
        state_layout = QHBoxLayout()
        state_layout.setContentsMargins(0,0,0,0)
        state_layout.setSpacing(0)
        state_layout.addWidget(QLabel("  "))
        state_layout.addWidget(self.state_label1)
        state_layout.addStretch(1)
        state_layout.addWidget(self.state_label2)
        state_layout.addWidget(QLabel("  "))


        # 整个界面是一个垂直布局
        self.all_vbox = QVBoxLayout()
        # 外边距为0，内间距为0
        self.all_vbox.setContentsMargins(0,0,0,0)
        self.all_vbox.setSpacing(0)
        self.all_vbox.addWidget(top_back_widget)
        self.all_vbox.addWidget(self.centre_layout)
        self.all_vbox.addLayout(state_layout)
        self.setLayout(self.all_vbox)


    def switchPage(self, id):
        if self.current == id:
            return
        if id < 0 or id > 4:
            LOG.warning('the page id:', id)
            return
        self.state_label1.setText("")
        self.state_label2.setText("")
        self.current = id
        self.centre_layout.setCurrentIndex(self.current)


