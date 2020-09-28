from utils.utils import *

class FreeDatasetAugment():
    def __init__(self, hyp):
        self.hyp = hyp

    def __call__(self, img, labels, border):
        # 图片几何变换
        img, labels = random_geometric(img, labels, degrees=self.hyp['degrees'], translate=self.hyp['translate'],
                    scale=self.hyp['scale'], shear=self.hyp['shear'], perspective=self.hyp['perspective'],
                    border=border)
        # 光学变换
        random_photometric(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

        # 坐标转换
        img, labels = to_percent_coords(img, labels)

        # 随机翻转
        return flip_img(img, labels, self.hyp['flipud'], self.hyp['fliplr'])
