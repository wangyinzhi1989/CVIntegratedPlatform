from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from res.res import *

class WaringDlg(QMessageBox):
    def __init__(self, titile = None, text=None, parent=None):
        super(WaringDlg, self).__init__(parent)
        #self.setParent(parent)
        self.setWindowTitle(titile)
        self.setText(text)
        self.setWindowIcon(QIcon(r':/icons/waring'))
        self.exec_()
