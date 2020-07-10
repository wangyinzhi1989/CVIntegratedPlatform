from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from log import LOG

class Library(QWidget):
    def __init__(self):
        super(Library, self).__init__()
        tool_frame = QFrame()
        tool_frame.setFrameShape(QFrame.StyledPanel)
        tool_frame.setFixedHeight(20)

        self.open=True
        def_w = 240

        model_lb = QLabel("    模型:")
        font = model_lb.font()
        size = font.pointSize()*0.9
        font.setPointSize(size)
        self.model_value = QComboBox()
        self.model_value.setFixedWidth(def_w)
        tips_lb = QLabel("训练产生的模型文件")
        tips_lb.setStyleSheet("color:red")
        tips_lb.setFont(font)
        model_ly = QHBoxLayout()
        model_ly.setContentsMargins(0,0,0,0)
        model_ly.setSpacing(2)
        model_ly.addWidget(model_lb)
        model_ly.addWidget(self.model_value)
        model_ly.addWidget(tips_lb)
        model_ly.addStretch(1)

        self.lib_tensorrt = QRadioButton("tensorrt")
        self.lib_libtorch = QRadioButton("libtorch")
        self.lib_libtorch.setChecked(True)
        lib_group = QButtonGroup(self)
        lib_group.addButton(self.lib_tensorrt)
        lib_group.addButton(self.lib_libtorch)
        lib_ly = QHBoxLayout()
        lib_ly.setContentsMargins(0,0,0,0)
        lib_ly.setSpacing(2)
        lib_ly.addWidget(QLabel("  推理库:"))
        lib_ly.addWidget(self.lib_tensorrt)
        lib_ly.addWidget(self.lib_libtorch)
        lib_ly.addStretch(1)

        self.save_path = QLineEdit()
        self.save_path.setFixedWidth(1.5*def_w)
        l_save_path_bt = QPushButton("请选择")
        l_save_path_bt.clicked.connect(self.localSavePathSel)
        save_ly = QHBoxLayout()
        save_ly.setContentsMargins(0,0,0,0)
        save_ly.setSpacing(2)
        save_ly.addWidget(QLabel("保存路径:"))
        save_ly.addWidget(self.save_path)
        save_ly.addWidget(l_save_path_bt)
        save_ly.addStretch(1)

        self.pkg = QPushButton("打包")
        self.pkg.clicked.connect(self.package)
        self.pkg.setDisabled(self.open)
        self.pkg.setFixedWidth(80)
        self.pkg_dl = QPushButton("打包并下载")
        self.pkg_dl.clicked.connect(self.packageAndDownload)
        self.pkg_dl.setDisabled(self.open)
        self.pkg_dl.setFixedWidth(80)
        pkg_ly = QHBoxLayout()
        pkg_ly.setContentsMargins(0,0,0,0)
        pkg_ly.setSpacing(40)
        pkg_ly.addWidget(self.pkg)
        pkg_ly.addWidget(self.pkg_dl)
        #pkg_ly.addStretch(1)

        win_layout = QVBoxLayout()
        win_layout.setContentsMargins(60,60,60,60)
        win_layout.setSpacing(40)
        win_layout.addLayout(model_ly)
        win_layout.addLayout(lib_ly)
        win_layout.addLayout(save_ly)
        win_layout.addLayout(pkg_ly)
        win_layout.setAlignment(pkg_ly, Qt.AlignCenter)
        win_layout.addStretch(1)

        cent_frame = QFrame()
        cent_frame.setFrameShape(QFrame.StyledPanel)
        cent_frame.setLineWidth(1)
        cent_frame.setLayout(win_layout)
        cent_layout = QVBoxLayout()
        cent_layout.addWidget(cent_frame)
        cent_layout.setAlignment(cent_frame, Qt.AlignCenter)

        win_frame = QFrame()
        win_frame.setFrameShape(QFrame.StyledPanel)
        win_frame.setLineWidth(1)
        win_frame.setLayout(cent_layout)

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(0,0,0,0)
        v_layout.setSpacing(0)
        v_layout.addWidget(tool_frame)
        v_layout.addWidget(win_frame)
        self.setLayout(v_layout)


    def localSavePathSel(self):
        text = QFileDialog.getExistingDirectory(self, '保存路径', '.',
          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.save_path.setText(text)

    def package(self):
      pass

    def packageAndDownload(self):
      pass