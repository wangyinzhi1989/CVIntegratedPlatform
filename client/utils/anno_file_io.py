import sys
import os
from log import LOG

class AnnoFileIo:
    '''
    line format: label x1 y1 x2 y2
    '''
    def __init__(self, file):
        self.file = file
        self.annos = []
        self.load()

    def load(self):
        if not os.path.exists(self.file):
            return 

        try:
            with open(self.file, 'r', encoding='utf-8') as fp:
                for line_str in fp.readlines():
                    line = line_str.split(' ')
                    anno = {'label':line[0], 'xmin':int(line[1]), 'ymin':int(line[2]),
                     'xmax':int(line[3]), 'ymax':int(line[4])}
                    self.annos.append(anno)
        except:
            LOG.error("annotation file {} non-existent".format(self.file))

    def save(self):
        try:
            with open(self.file, 'w', encoding='utf-8') as fp:
                for anno in self.annos:
                    anno_str = '{} {} {} {} {}\n'.format(anno['label'], anno['xmin'], anno['ymin'], anno['xmax'], anno['ymax'])
                    fp.write(anno_str)
        except:
            LOG.error("annotation file {} non-existent".format(self.file))

    def clear(self):
        self.annos.clear()

    def add(self, anno):
        self.annos.append(anno)