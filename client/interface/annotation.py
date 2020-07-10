from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from log import LOG
from .draw_area import DrawArea
from .label_dlg import LabelDialog
from .annotation_config_dialog import AnnotationConfigDialog
from .shape import Shape
from utils.anno_file_io import AnnoFileIo
from utils.simple_dialog import WaringDlg
from config import CONF
import os

# 自定义列表项
class HashableQListWidgetItem(QListWidgetItem):
    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))

class Annotation(QWidget):
    def __init__(self):
        super(Annotation, self).__init__()

        self.labels = CONF.anno_conf["labels"].split(',')
        self.curr_anno_io = None
        self.image = QImage()
        self.labelDialog = LabelDialog(parent=self, listItem=self.labels)
        self.items2shapes = {}
        self.shapes2items = {}
        self._noSelectionSlot = False

        self.setStyleSheet("QPushButton:hover{background-color:rgb(160,200,240)}" #光标移动到上面后的前景色
                            "QPushButton:pressed{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                            "QPushButton:checked{background-color:rgb(160,200,240);color: red;}" #按下时的样式
                            )
        tool_bt_w = 60
        tool_bt_h = 20
        list_w = 200
        # 工具按钮设置
        config_bt = QPushButton("配置", self)
        config_bt.setFixedSize(200, tool_bt_h)
        config_bt.clicked.connect(self.openConfigDlg)
        prev_bt = QPushButton("上一个", self)
        prev_bt.setFixedSize(tool_bt_w, tool_bt_h)
        prev_bt.clicked.connect(self.prevCB)
        next_bt = QPushButton("下一个", self)
        next_bt.setFixedSize(tool_bt_w, tool_bt_h)
        next_bt.clicked.connect(self.nextCB)
        box_bt = QPushButton("框选", self)
        box_bt.setFixedSize(tool_bt_w, tool_bt_h)
        box_bt.clicked.connect(self.newBoxCB)
        box_del_bt = QPushButton("框删除", self)
        box_del_bt.setFixedSize(tool_bt_w, tool_bt_h)
        box_del_bt.clicked.connect(self.delSelShapeCB)
        del_action = QAction(self)
        #zoom_in_bt = QPushButton("放大", self)
        #zoom_in_bt.setFixedSize(tool_bt_w, tool_bt_h)
        self.zoom_value = QSpinBox()
        self.zoom_value.setRange(10, 500)
        self.zoom_value.setSingleStep(10)
        self.zoom_value.setSuffix(' %')
        self.zoom_value.setValue(100)
        self.zoom_value.setFixedSize(tool_bt_w, tool_bt_h)
        self.zoom_value.setStyleSheet("background-color:white")
        self.zoom_value.valueChanged.connect(self.zoomChange)

        #zoom_out_bt = QPushButton("缩小", self)
        #zoom_out_bt.setFixedSize(tool_bt_w, tool_bt_h)
        fin_win_bt = QPushButton("满窗", self)
        fin_win_bt.setFixedSize(tool_bt_w, tool_bt_h)
        fin_win_bt.clicked.connect(self.finWinCB)
        save_bt = QPushButton("保存", self)
        save_bt.setFixedSize(tool_bt_w, tool_bt_h)
        save_bt.clicked.connect(self.saveCB)
        self.defult_label_flg = QCheckBox("默认标签")
        self.defult_label_flg.setChecked(False)
        self.defult_label_flg.setFixedSize(100, tool_bt_h)
        self.defult_label_value = QComboBox()
        self.defult_label_value.setFixedSize(120, tool_bt_h)
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(config_bt)
        tool_layout.addWidget(prev_bt)
        tool_layout.addWidget(next_bt)
        tool_layout.addWidget(box_bt)
        tool_layout.addWidget(box_del_bt)
        #tool_layout.addWidget(zoom_in_bt)
        tool_layout.addWidget(self.zoom_value)
        #tool_layout.addWidget(zoom_out_bt)
        tool_layout.addWidget(fin_win_bt)
        tool_layout.addWidget(save_bt)
        tool_layout.addWidget(QLabel(""))
        tool_layout.addWidget(self.defult_label_value)
        tool_layout.addWidget(QLabel(""))
        tool_layout.addWidget(self.defult_label_flg)
        tool_layout.addStretch(1)
        tool_layout.setContentsMargins(0,0,0,0)
        tool_layout.setSpacing(0)

        # 列表栏设置
        label_list_lab = QLabel("标签列表")
        self.label_list = QListWidget()
        self.label_list.setFixedWidth(list_w-3)
        self.label_list.itemDoubleClicked.connect(self.labelListItemDbClick)
        self.label_list.itemSelectionChanged.connect(self.labelListItemSelected)
        self.label_list.itemActivated.connect(self.labelListItemSelected)
        file_list_lab = QLabel("文件列表")
        self.file_list = QListWidget()
        self.file_list.setFixedWidth(list_w-3)
        self.file_list.itemDoubleClicked.connect(self.fileListItemDbClick)

        list_layout = QVBoxLayout()
        list_layout.setContentsMargins(0,1,0,1)
        list_layout.setSpacing(2)
        list_layout.addWidget(label_list_lab)
        list_layout.addWidget(self.label_list)
        list_layout.addWidget(file_list_lab)
        list_layout.addWidget(self.file_list)

        list_frame = QFrame()
        list_frame.setFrameShape(QFrame.StyledPanel)
        list_frame.setLineWidth(1)
        list_frame.setFixedWidth(list_w)
        list_frame.setLayout(list_layout)

        # 绘图区域设置
        self.draw_area = DrawArea(parent=self)
        self.draw_area.zoomSignal.connect(self.zoomSignalCB)
        self.draw_area.scrollSignal.connect(self.scrollSignalCB)
        self.draw_area.newShapeSignal.connect(self.newShapeSignalCB)
        self.draw_area.selChangedSignal.connect(self.selChangedSignalCB)
        self.draw_area.drawingSignal.connect(self.drawingSignalCB)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.draw_area)
        self.scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: self.scroll.verticalScrollBar(),
            Qt.Horizontal: self.scroll.horizontalScrollBar()
        }

        win_layout = QHBoxLayout()
        win_layout.setContentsMargins(0,0,0,0)
        win_layout.setSpacing(0)
        win_layout.addWidget(list_frame)
        win_layout.addWidget(self.scroll)

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

        # 内容填充
        self.setDefultLabelItems()
        self.setFileListItems()

        # 快捷键
        del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        del_shortcut.activated.connect(self.delSelShapeCB)

        save_shortcut = QShortcut(QKeySequence('Ctrl+S'), self)
        save_shortcut.activated.connect(self.saveCB)

    # 缩放请求
    def zoomSignalCB(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and draw_area size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scroll.width()
        h = self.scroll.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def scrollSignalCB(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def newShapeSignalCB(self):
        """ 新框回调 """
        if not self.defult_label_flg.isChecked():
            text = self.labelDialog.popUp(labels = self.labels)
        else:
            text = self.defult_label_value.currentText()

        if text is not None:
            shape = self.draw_area.setLastLabel(text)
            self.addLabel(shape)
            self.draw_area.setEditing(True)
        else:
            self.draw_area.resetAllLines()

    # 绘图区域选中矩形信号
    def selChangedSignalCB(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.draw_area.selectedShape
            if shape:
                item = self.shapes2items[shape]
                item.setSelected(True)
            else:
                self.label_list.clearSelection()

    # 绘图区拖拽绘制信号
    def drawingSignalCB(self, drawing=True):
        if not drawing:
            self.draw_area.setEditing(True)
            self.draw_area.restoreCursor()

    def openConfigDlg(self):
        dlg = AnnotationConfigDialog(parent=self)
        if QDialog.Accepted == dlg.exec_():
            self.resetState()
            self.labels = dlg.label_list_te.toPlainText().split(',')
            self.setDefultLabelItems()
            self.setFileListItems()

    def prevCB(self):
        idx = self.file_list.row(self.file_list.currentItem())
        if idx < 0:
            return
        if idx == 0:
            WaringDlg(titile = '提示', text='已经是第一张图片', parent=self)
            return
        idx -= 1
        self.file_list.setCurrentRow(idx)
        file = self.file_list.currentItem().text()
        self.drawAreaLoadImg(file)

    def nextCB(self):
        idx = self.file_list.row(self.file_list.currentItem())
        if idx < 0:
            return
        if idx == self.file_list.count() - 1:
            WaringDlg(titile = '提示', text='已经是最后一张图片', parent=self)
            return
        idx += 1
        self.file_list.setCurrentRow(idx)
        file = self.file_list.currentItem().text()
        self.drawAreaLoadImg(file)

    def newBoxCB(self):
        self.draw_area.setEditing(False)

    def delSelShapeCB(self):
        shape = self.draw_area.deleteSelected()
        if shape is None:
            LOG.warning('rm empty label')
            return
        item = self.shapes2items[shape]
        self.label_list.takeItem(self.label_list.row(item))
        del self.shapes2items[shape]
        del self.items2shapes[item]

    def setDefultLabelItems(self):
        self.defult_label_flg.setChecked(False)
        self.defult_label_value.clear()
        self.defult_label_value.addItems(self.labels)

    def setFileListItems(self):
        # 获取配置目录下支持的图片文件
        supports = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []
        for root, dirs, files in os.walk(CONF.anno_conf["img_path"]):
            for file in files:
                if file.lower().endswith(tuple(supports)):
                    images.append(file)

        self.file_list.clear()
        self.file_list.addItems(images)

    def addLabel(self, shape):
        item = HashableQListWidgetItem(shape.label)
        self.items2shapes[item] = shape
        self.shapes2items[shape] = item
        self.label_list.addItem(item)

    def resetState(self):
        # 保存标注
        if self.curr_anno_io is not None:
            # 无论是否改变，这里都重置annos并保存
            self.curr_anno_io.clear()
            for shape in self.draw_area.shapes:
                anno = shape.toAnno()
                self.curr_anno_io.add(anno)
            self.curr_anno_io.save()
        # 状态重置
        self.draw_area.resetState()
        self.label_list.clear()
        self.zoom_value.setValue(100)
        self.items2shapes.clear()
        self.shapes2items.clear()

    def drawAreaLoadImg(self, file):
        img_file = CONF.anno_conf["img_path"] + '//' + file
        if not os.path.exists(img_file):
            LOG.error("{} non-existent".format(img_file))
            return

        # 重置状态
        self.resetState()
        self.draw_area.setEnabled(False)

        # 图片加载
        if not self.image.load(img_file):
            LOG.error("{} load failed.".format(img_file))
            return

        self.draw_area.loadPixmap(QPixmap.fromImage(self.image))
        zoom_v = self.scaleFitWindow()
        self.setZoom(100*zoom_v)
        self.draw_area.setEnabled(True)
        self.draw_area.setFocus(True)

        # label 添加
        anno_file = CONF.anno_conf["save_path"] + '//' + file.split('.')[0] + '.txt'
        self.curr_anno_io = AnnoFileIo(anno_file)
        shapes = []
        for anno in self.curr_anno_io.annos:
            tmp_shape = Shape(label= anno['label'])
            tmp_shape.addPoint(QPoint(anno['xmin'], anno['ymin']))
            tmp_shape.addPoint(QPoint(anno['xmax'], anno['ymin']))
            tmp_shape.addPoint(QPoint(anno['xmax'], anno['ymax']))
            tmp_shape.addPoint(QPoint(anno['xmin'], anno['ymax']))
            tmp_shape.close()
            shapes.append(tmp_shape)
            self.addLabel(tmp_shape)
        self.draw_area.loadShapes(shapes)

        self.parent().window().state_label1.setText(file)

    def fileListItemDbClick(self, item):
        file = item.text()
        self.drawAreaLoadImg(file)

    def zoomChange(self):
        if self.image.isNull():
            LOG.warning('image is null.')
            return
        self.draw_area.scale = 0.01 * self.zoom_value.value()
        self.draw_area.adjustSize()
        self.draw_area.update()

    def setZoom(self, value):
        self.zoom_value.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoom_value.value() + increment)

    def scaleFitWindow(self):
        """计算最适窗口比例，注意调用前请确保image存在"""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.scroll.width() - e
        h1 = self.scroll.height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.draw_area.pixmap.width() - 0.0
        h2 = self.draw_area.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def finWinCB(self):
        if self.image.isNull():
            LOG.warning('image is null.')
            return
        zoom_v = self.scaleFitWindow()
        self.setZoom(100*zoom_v)

    def saveCB(self):
        # 保存标注
        if self.curr_anno_io is not None:
            # 无论是否改变，这里都重置annos并保存
            self.curr_anno_io.clear()
            for shape in self.draw_area.shapes:
                anno = shape.toAnno()
                self.curr_anno_io.add(anno)
            self.curr_anno_io.save()

    # 标签列表项选中，关联到绘图区效果
    def labelListItemSelected(self):
        # 不支持多选，这里去选中的第一个
        items = self.label_list.selectedItems()
        item = items[0] if items else None
        if item and self.draw_area.editing():
            self._noSelectionSlot = True
            self.draw_area.selectShape(self.items2shapes[item])

    def labelListItemDbClick(self):
        if not self.draw_area.editing():
            return
        # 不支持多选，这里去选中的第一个
        items = self.label_list.selectedItems()
        item = items[0] if items else None
        if not item:
            return
        text = self.labelDialog.popUp(text=item.text(),labels = self.labels)
        if text is not None:
            item.setText(text)
            self.items2shapes[item].label = text
