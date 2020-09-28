import torch
import torch.utils.data as data
from pathlib import Path
import os
from config import LOG
import glob
import cv2
import numpy as np
from utils.utils import xyxy2xywh
from data.data_augment import FreeDatasetAugment

def detection_collate(batch):
    img, label, path = zip(*batch)  # transposed
    for i, l in enumerate(label):
        # 一批中的位置idx build_targets时用于定位目标
        l[:, 0] = i
    return torch.stack(img, 0), torch.cat(label, 0), path


# 支持的格式
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

class FreeDataset(data.Dataset):
    def __init__(self, path, class_names, hyp, img_size = 640, transform=None):
        super(FreeDataset, self).__init__()
        self.transform = transform if transform else FreeDatasetAugment(hyp)
        self.img_size = img_size
        # mosaic增强图片
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.root = str(Path(path)) + os.sep
        assert (os.path.isdir(self.root)), LOG.error("dataset path is not dir.")

        # 获取所有文件,并过滤不支持的格式
        files = glob.glob(self.root + 'imgs' + os.sep + '*.*')
        self.img_files = [x for x in files if os.path.splitext[-1].lower() in img_formats]
        label_path = self.root + 'labels' + os.sep
        self.label_files = [label_path + x.split(os.sep)[-1].replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files]

        assert (len(self.img_files) == len(self.label_files)), LOG.error("image annotation lose.")

        # 类别和idx对应
        self.classes = []
        self.classes.append('__background__')
        self.classes += [i.strip() for i in class_names.split(',')]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        LOG.info("class to idx: {}".format(self.class_to_idx))

    def __len__(self):
        return len(self.img_files)


    def classes_number(self):
        return len(self.classes)


    def __getitem__(self, index):
        img, labels = self.load_mosaic(index)
        img, labels = self.transform(img, labels, self.mosaic_border)

        nL = len(labels)
        labels_out = torch.zeros((nL, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, hwc to chw
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index]


    def load_label(self, index):
        labels = []
        try:
            with open(self.label_files[index], 'r') as f:
                for l in f.readlines():
                    target = l.split(' ')
                    labels.append([self.class_to_idx(target[0]), int(target[1]), int(target[2]), int(target[3]), int(target[4])])
        except:
            LOG.error('annotation file {} non-existent'.format(self.label_files[index]))
        return labels


    def load_mosaic(self, index):
        # mosaic增强
        labels4 = []
        s = self.img_size
        # mosaic增强四部分图片的相交点
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        # 随机出其他三张图片
        indices = [index] + [random.randint(0, len(self.img_files) - 1) for _ in range(3)]
        for i, index in enumerate(indices):
            # Load image
            img = cv2.imread(self.img_files[index])
            assert img is not None, LOG.error('Image not found' + self.img_files[index])
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = load_label(index)
            labels = x.copy()
            if x.size > 0:  # xyxy format
                labels[:, 1] = x[:, 1] + padw
                labels[:, 2] = x[:, 2] + padh
                labels[:, 3] = x[:, 3] + padw
                labels[:, 4] = x[:, 4] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        return img4, labels4




