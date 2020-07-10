from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from res.res import *

from log import LOG
from config import CONF

class TrainingConfigDialog(QDialog):
    def __init__(self, parent=None):
        super(TrainingConfigDialog, self).__init__(parent)
        self.setFixedSize(650,400)
        self.setWindowTitle("模型训练配置")
        self.setWindowIcon(QIcon(':/icons/config'))
        self.setModal(True)

        flgs = self.windowFlags()
        self.setWindowFlags(flgs & ~Qt.WindowContextHelpButtonHint)

        le_width = 40

        self.open=True

        # 界面元素构成
        self.l_data_path_te = QLineEdit()
        self.l_data_path_te.setReadOnly(True)
        l_data_path_bt = QPushButton("请选择")
        l_data_path_bt.clicked.connect(self.localDataPathSel)

        self.l_save_path_te = QLineEdit()
        self.l_save_path_te.setReadOnly(True)
        l_save_path_bt = QPushButton("请选择")
        l_save_path_bt.clicked.connect(self.localSavePathSel)

        self.model_sel_h = QRadioButton('精确慢速')
        self.model_sel_h.setDisabled(self.open)
        self.model_sel_m = QRadioButton('均衡')
        self.model_sel_m.setDisabled(self.open)
        self.model_sel_l = QRadioButton('欠精确快速')
        self.model_sel_h.setChecked(True)

        self.dev_sel_c = QRadioButton('CPU')
        self.dev_sel_c.setChecked(True)
        self.dev_sel_g = QRadioButton('GPU')
        self.dev_sel_g.setDisabled(self.open)
        self.dev_value = QLineEdit()
        self.dev_value.setFixedWidth(3*le_width)
        self.dev_value.setStyleSheet("background-color:white")

        self.remote_sel = QCheckBox('远端训练')
        self.remote_sel.setChecked(False)
        self.remote_sel.setDisabled(self.open)
        self.ip_value = QLineEdit()
        self.ip_value.setFixedWidth(4*le_width)
        self.ip_value.setStyleSheet("background-color:white")
        self.port_value = QLineEdit()
        self.port_value.setFixedWidth(le_width)
        self.port_value.setStyleSheet("background-color:white")
        self.r_data_path = QLineEdit()
        self.r_data_path.setStyleSheet("background-color:white")
        r_save_path_lb = QLabel('模型保存路径:')
        self.r_pro_path = QLineEdit()
        self.r_pro_path.setStyleSheet("background-color:white")
        remote_hbl1 = QHBoxLayout()
        remote_hbl1.addWidget(QLabel('      ip地址:'))
        remote_hbl1.addWidget(self.ip_value)
        remote_hbl1.addStretch(1)
        remote_hbl1.addWidget(QLabel('端口:'))
        remote_hbl1.addWidget(self.port_value)
        remote_hbl1.addStretch(1)
        remote_hbl2 = QHBoxLayout()
        remote_hbl2.addWidget(QLabel('    数据路径:'))
        remote_hbl2.addWidget(self.r_data_path)
        remote_hbl3 = QHBoxLayout()
        remote_hbl3.addWidget(QLabel('    工程路径:'))
        remote_hbl3.addWidget(self.r_pro_path)
        remote_vbl = QVBoxLayout()
        remote_vbl.addLayout(remote_hbl1)
        remote_vbl.addLayout(remote_hbl2)
        remote_vbl.addLayout(remote_hbl3)
        self.remote_frame = QFrame()
        self.remote_frame.setFrameShape(QFrame.StyledPanel)
        self.remote_frame.setLayout(remote_vbl)
        remote_layout = QVBoxLayout()
        remote_layout.setContentsMargins(0,0,0,0)
        remote_layout.setSpacing(0)
        remote_layout.addWidget(self.remote_sel)
        remote_layout.addWidget(self.remote_frame)

        self.senior_sel = QCheckBox('高级选项')
        self.senior_sel.setChecked(False)
        self.senior_sel.setDisabled(self.open)
        self.lr_value = QLineEdit()
        self.lr_value.setFixedWidth(2*le_width)
        self.lr_value.setStyleSheet("background-color:white")
        self.iter_value = QLineEdit()
        self.iter_value.setFixedWidth(le_width)
        self.iter_value.setStyleSheet("background-color:white")
        self.batch_size = QLineEdit()
        self.batch_size.setFixedWidth(le_width)
        self.batch_size.setStyleSheet("background-color:white")
        self.refresh_time = QComboBox()
        self.refresh_time.addItems(['5','10','15','20','30','45','60'])
        self.refresh_time.setFixedWidth(le_width)
        self.refresh_time.setStyleSheet("background-color:white")
        senior_hbl1 = QHBoxLayout()
        senior_hbl1.addWidget(QLabel('学习率:'))
        senior_hbl1.addWidget(self.lr_value)
        senior_hbl1.addWidget(QLabel('  训练代数:'))
        senior_hbl1.addWidget(self.iter_value)
        senior_hbl1.addWidget(QLabel('  批大小:'))
        senior_hbl1.addWidget(self.batch_size)
        senior_hbl1.addWidget(QLabel('  过程刷新频率(s):'))
        senior_hbl1.addWidget(self.refresh_time)
        senior_hbl1.addStretch(1)
        senior_vbl = QVBoxLayout()
        senior_vbl.addLayout(senior_hbl1)
        self.senior_frame = QFrame()
        self.senior_frame.setFrameShape(QFrame.StyledPanel)
        self.senior_frame.setLayout(senior_vbl)
        senior_layout = QVBoxLayout()
        senior_layout.setContentsMargins(0,0,0,0)
        senior_layout.setSpacing(0)
        senior_layout.addWidget(self.senior_sel)
        senior_layout.addWidget(self.senior_frame)

        ok_bt = QDialogButtonBox(QDialogButtonBox.Ok, self)
        ok_bt.clicked.connect(self.finished)

        # 布局
        l_data_hbl = QHBoxLayout()
        l_data_hbl.setContentsMargins(0,0,0,0)
        l_data_hbl.setSpacing(5)
        l_data_hbl.addWidget(QLabel("本地数据路径:"))
        l_data_hbl.addWidget(self.l_data_path_te)
        l_data_hbl.addWidget(l_data_path_bt)
        
        l_save_hbl = QHBoxLayout()
        l_save_hbl.setContentsMargins(0,0,0,0)
        l_save_hbl.setSpacing(5)
        l_save_hbl.addWidget(QLabel("模型保存路径:"))
        l_save_hbl.addWidget(self.l_save_path_te)
        l_save_hbl.addWidget(l_save_path_bt)

        l_model_hbl = QHBoxLayout()
        l_model_hbl.setContentsMargins(0,0,0,0)
        l_model_hbl.setSpacing(5)
        l_model_hbl.addWidget(QLabel("    模型选择:"))
        l_model_hbl.addWidget(self.model_sel_h)
        l_model_hbl.addWidget(self.model_sel_m)
        l_model_hbl.addWidget(self.model_sel_l)
        l_model_hbl.addStretch(1)
        
        l_dev_hbl = QHBoxLayout()
        l_dev_hbl.setContentsMargins(0,0,0,0)
        l_dev_hbl.setSpacing(5)
        l_dev_hbl.addWidget(QLabel("    训练设备:"))
        l_dev_hbl.addWidget(self.dev_sel_c)
        l_dev_hbl.addWidget(self.dev_sel_g)
        l_dev_hbl.addWidget(self.dev_value)
        l_dev_hbl.addWidget(QLabel("以,分割"))
        l_dev_hbl.addStretch(1)

        v_layout = QVBoxLayout()
        v_layout.setContentsMargins(15,20,15,20)
        v_layout.setSpacing(15)
        v_layout.addLayout(l_data_hbl)
        v_layout.addLayout(l_save_hbl)
        v_layout.addLayout(l_model_hbl)
        v_layout.addLayout(l_dev_hbl)
        v_layout.addLayout(remote_layout)
        v_layout.addLayout(senior_layout)
        v_layout.addWidget(ok_bt)
        v_layout.addStretch(1)
        self.setLayout(v_layout)

        # 单选框分组
        model_group = QButtonGroup(self)
        model_group.addButton(self.model_sel_h)
        model_group.addButton(self.model_sel_m)
        model_group.addButton(self.model_sel_l)
        dev_group = QButtonGroup(self)
        dev_group.addButton(self.dev_sel_c)
        dev_group.addButton(self.dev_sel_g)

        # 复选框事件
        self.senior_sel.stateChanged.connect(lambda: self.senior_frame.show() if self.senior_sel.isChecked() else self.senior_frame.hide())
        self.remote_sel.stateChanged.connect(lambda: self.remote_frame.show() if self.remote_sel.isChecked() else self.remote_frame.hide())
        self.dev_sel_g.toggled.connect(lambda: self.dev_value.setDisabled(not self.dev_sel_g.isChecked()))
        self.setConfig()

    def setConfig(self):
        self.l_data_path_te.setText(CONF.train_conf["l_data_path"])
        self.l_save_path_te.setText(CONF.train_conf["l_save_path"])
        # 模型选择1-精确慢速  2-均衡  3-欠精确快速
        if CONF.train_conf["model_sel"] == 1 and (not self.open):
            self.model_sel_h.setChecked(True)
        elif CONF.train_conf["model_sel"] == 2 and (not self.open):
            self.model_sel_m.setChecked(True)
        else:
            self.model_sel_l.setChecked(True)
        if CONF.train_conf["gpu"] and (not self.open):
            self.dev_sel_g.setChecked(True)
        else:
            self.dev_sel_c.setChecked(True)
        self.dev_value.setText(','.join([str(dev) for dev in CONF.train_conf["gpu_dev"]]))
        self.dev_value.setDisabled(not self.dev_sel_g.isChecked())

        self.remote_sel.setChecked(CONF.train_conf["remote"] and (not self.open))
        if not CONF.train_conf["remote"] or self.open:
            self.remote_frame.hide()
        self.ip_value.setText(CONF.train_conf["r_ip"])
        self.port_value.setText(str(CONF.train_conf["r_port"]))
        self.r_data_path.setText(CONF.train_conf["r_data_path"])
        self.r_pro_path.setText(CONF.train_conf["r_pro_path"])

        self.senior_sel.setChecked(CONF.train_conf["senior"] and (not self.open))
        if not CONF.train_conf["senior"] or self.open:
            self.senior_frame.hide()
        self.lr_value.setText(str(CONF.train_conf["s_lr"]))
        self.iter_value.setText(str(CONF.train_conf["s_iter"]))
        self.batch_size.setText(str(CONF.train_conf["s_batch"]))
        self.refresh_time.setCurrentText(str(CONF.train_conf["s_refresh"]))


    def localDataPathSel(self):
        text = QFileDialog.getExistingDirectory(self, '本地数据路径', '.',
          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.l_data_path_te.setText(text)

    def localSavePathSel(self):
        text = QFileDialog.getExistingDirectory(self, '模型保存路径', '.',
          QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.l_save_path_te.setText(text)

    def finished(self):
        CONF.train_conf["l_data_path"] = self.l_data_path_te.text()
        CONF.train_conf["l_save_path"] = self.l_save_path_te.text()
        if self.model_sel_h.isChecked():
            CONF.train_conf["model_sel"] = 1
        elif self.model_sel_m.isChecked():
            CONF.train_conf["model_sel"] = 2
        else:
            CONF.train_conf["model_sel"] = 3
        if self.dev_sel_c.isChecked():
            CONF.train_conf["gpu"] = False
        else:
            CONF.train_conf["gpu"] = True
        CONF.train_conf["gpu_dev"] = [int(dev) for dev in self.dev_value.text().split(',')]

        CONF.train_conf["remote"] = self.remote_sel.isChecked()
        CONF.train_conf["r_ip"] = self.ip_value.text()
        CONF.train_conf["r_port"] = int(self.port_value.text())
        CONF.train_conf["r_data_path"] = self.r_data_path.text()
        CONF.train_conf["r_pro_path"] = self.r_pro_path.text()

        CONF.train_conf["senior"] = self.senior_sel.isChecked()
        CONF.train_conf["s_lr"] = float(self.lr_value.text())
        CONF.train_conf["s_iter"] = int(self.iter_value.text())
        CONF.train_conf["s_batch"] = int(self.batch_size.text())
        CONF.train_conf["s_refresh"] = int(self.refresh_time.currentText())
        CONF.save()
        self.accept()