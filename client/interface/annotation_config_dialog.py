from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from res.res import *

from log import LOG
from config import CONF

class AnnotationConfigDialog(QDialog):
    def __init__(self, parent=None):
        super(AnnotationConfigDialog, self).__init__(parent)
        self.setFixedSize(600,400)
        self.setWindowTitle("数据标注配置")
        self.setWindowIcon(QIcon(':/icons/config'))
        self.setModal(True)
        
        flgs = self.windowFlags()
        self.setWindowFlags(flgs & ~Qt.WindowContextHelpButtonHint)
        #self.setWindowFlags(Qt.WindowTitleHint | | Qt.WindowCloseButtonHint)

        # 界面元素构成
        img_path_lb = QLabel("图片路径:")
        self.img_path_te = QLineEdit()
        #self.img_path_te.setFixedHeight(30)
        self.img_path_te.setReadOnly(True)
        img_path_bt = QPushButton("请选择")
        img_path_bt.clicked.connect(self.imgPathSel)

        save_path_lb = QLabel("保存路径:")
        self.save_path_te = QLineEdit()
        #self.save_path_te.setFixedHeight(30)
        self.save_path_te.setReadOnly(True)
        save_path_bt = QPushButton("请选择")
        save_path_bt.clicked.connect(self.savePathSel)

        label_list_lb = QLabel("标签：以,分割")
        self.label_list_te = QTextEdit()
        self.label_list_te.setFixedHeight(120)
        self.label_list_te.setStyleSheet("background-color:white")
        ok_bt = QDialogButtonBox(QDialogButtonBox.Ok, self)
        ok_bt.clicked.connect(self.finished)

        #布局
        img_h_layout = QHBoxLayout()
        img_h_layout.setContentsMargins(0,0,0,0)
        img_h_layout.setSpacing(5)
        img_h_layout.addWidget(img_path_lb)
        img_h_layout.addWidget(self.img_path_te)
        img_h_layout.addWidget(img_path_bt)
        #img_h_layout.addStretch(1)

        save_h_layout = QHBoxLayout()
        save_h_layout.setContentsMargins(0,0,0,0)
        save_h_layout.setSpacing(5)
        save_h_layout.addWidget(save_path_lb)
        save_h_layout.addWidget(self.save_path_te)
        save_h_layout.addWidget(save_path_bt)
        #save_h_layout.addStretch(1)

        label_v_layout = QVBoxLayout()
        label_v_layout.addWidget(label_list_lb)
        label_v_layout.addWidget(self.label_list_te)
        label_v_layout.setContentsMargins(0,0,0,0)
        label_v_layout.setSpacing(5)
        
        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(20,20,20,20)
        v_layout.setSpacing(20)
        v_layout.addStretch(1)
        v_layout.addLayout(img_h_layout)
        v_layout.addLayout(save_h_layout)
        v_layout.addLayout(label_v_layout)
        v_layout.addWidget(ok_bt)
        v_layout.addStretch(1)
        self.setLayout(v_layout)
        self.setConfig()
        self.show()

    def setConfig(self):
        self.img_path_te.setText(CONF.anno_conf["img_path"])
        self.save_path_te.setText(CONF.anno_conf["save_path"])
        self.label_list_te.setText(CONF.anno_conf["labels"])

    def imgPathSel(self):
        text = QFileDialog.getExistingDirectory(self, '图片路径', '.',
          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.img_path_te.setText(text)

    def savePathSel(self):
        text = QFileDialog.getExistingDirectory(self, '保存路径', '.',
          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.save_path_te.setText(text)

    def finished(self):
        CONF.anno_conf["img_path"] = self.img_path_te.text()
        CONF.anno_conf["save_path"] = self.save_path_te.text()
        CONF.anno_conf["labels"] = self.label_list_te.toPlainText()
        CONF.save()
        self.accept()
