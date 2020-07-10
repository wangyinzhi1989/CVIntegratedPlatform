import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from interface import *
import log
from log import LOG
from config import CONF

def main():
    log.log_setting(CONF.logger["file"], CONF.logger["level"], CONF.logger["format"],
     CONF.logger["backupCount"], CONF.logger["interval"])
    LOG.info(CONF.service)
    LOG.info(CONF.anno_conf)

    app = QApplication(sys.argv)
    app.setApplicationName("free")
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == "__main__":
    annos = []
    annos.append({'label':'a', 'x1':11, 'y1':12})
    annos.append({'label':'b', 'x1':21, 'y1':22})
    annos.append({'label':'c', 'x1':31, 'y1':32})

    labels = [anno['label'] for anno in annos] 
    main()
    LOG.info("exited.")
