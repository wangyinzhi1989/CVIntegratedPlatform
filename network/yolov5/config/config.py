# 通过遗传算法获得的超参数

hyp = {
    'momentum' : 0.937,                 # 优化器动能
    'weight_decay' : 5e-4,              # 优化器中 权重衰减系数,pytorch将权重的衰减放到了权重优化器中
    'box':0.05,                         # box坐标回归损失比重系数
    'cls': 0.58,                        # 分类损失比重系数
    'cls_pw': 1.0,                      # 分类 BCELoss 正样本权重
    'obj': 1.0,                         # 目标损失比重系数
    'obj_pw': 1.0,                      # 目标 BCELoss 正样本权重
    'anchor_t': 4.0,                    # anchor-multiple threshold
    'fl_gamma': 0.0,                    # focal loss gamma (efficientDet default is gamma=1.5)
    'hsv_h': 0.014,                     # 色调增强
    'hsv_s': 0.68,                      # 饱和度增强 (fraction)
    'hsv_v': 0.36,                      # 亮度增强 (fraction)
    'degrees': 0.0,                     # 图像旋转 (+/- deg)
    'translate': 0.0,                   # 图像平移 (+/- fraction)
    'scale': 0.5,                       # 图像缩放 (+/- gain)
    'shear': 0.0,                       # 图片错切 (+/- deg)
    'perspective': 0.0,                 # 图片透视 (+/- fraction), range 0-0.001
    'flipud': 0.0,                      # 上下翻转 (probability)
    'fliplr': 0.5,                      # 左右翻转 (probability)
    'anchors' : [[10,13, 16,30, 33,23], [30,61, 62,45, 59,119],[116,90, 156,198, 373,326]]
}

dataset = {
    'img_size' : 640,                   # 图片尺寸
    # 类别名称，以‘,’分割
    'class_names': 'person, motor_vehicle'
}

logger = {
    "file": "./log/log.log",            # log文件路径
    "level": "DEBUG",                   # 日志等级
    # 格式
    "format": "%(levelname)s %(asctime)s %(thread)d %(filename)s:%(lineno)d %(funcName)s %(message)s",
    "backupCount": 7,                   # 保留文件数目
    "interval": 1                       # 日志文件切换间隔，单位天
}
