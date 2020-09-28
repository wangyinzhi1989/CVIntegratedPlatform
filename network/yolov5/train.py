import argparse
import torch
import time
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import config
from config import LOG
from data import FreeDataset, detection_collate
import torch.utils.data as data
from models import *
from torch.nn.parallel import DistributedDataParallel
from utils.utils import *
from utils.ema import *
import torch
from tqdm import tqdm
import os
from pathlib import Path
import test


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

# 混合精度标志
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# 参数枚举
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=None, help='Pretrained model')
parser.add_argument('--epochs', type=int, default=300, help = 'training epoch num')
parser.add_argument('--batch_size', type=int, default=16, help = 'training batch size')
parser.add_argument('--dataset_train', type=str, default='./data/', help = 'training dataset path')
parser.add_argument('--dataset_val', type=str, default='./data/', help = 'training dataset path')
parser.add_argument('--resume', type=str, default=None, help = 'resume training from file')
parser.add_argument('--start_iter', type=int, default=0, help = 'Resume training at this iter')
parser.add_argument('--save_interval', type=int, default=10, help = 'save checkpoint interval epoch')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
# 初始学习率 sgd-0.01 adam-0.001
parser.add_argument('--lr',  default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--save_folder', default='data/', help='Directory for saving checkpoint models')
parser.add_argument('--multi_scale', action='store_true', help='vary img-size +/- 50%')
parser.add_argument('--single_cls', action='store_true', help='train as single-class dataset')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
parser.add_argument('--save_path', type=str, default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--no_test', action='store_true', help='only test final epoch')
args = parser.parse_args()
args.world_size = 1


def train():
    # cuda时 使用DDP进行单机多卡或多机多卡训练
    if args.cuda:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        args.world_size = dist.get_world_size()
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.batch_size // args.world_size
    else:
        device = torch.device('cpu')

    hyp = config.hyp
    start_epoch = 1 if args.resume == None else args.start_iter + 1
    best_fitness = 0.0
    best = Path(args.save_path) / 'best.pth'

    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    
    # dataset
    with torch_distributed_zero_first(args.local_rank):
        dataset = FreeDataset(args.dataset_train, config.dataset['class_names'], 
                hyp, config.dataset['img_size'])
    batch_size = min(args.batch_size, len(dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1 else None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers, sampler=train_sampler,
        shuffle=True, collate_fn=detection_collate, pin_memory=True)
    # Testloader
    if args.local_rank in [-1, 0]:
        with torch_distributed_zero_first(args.local_rank):
            test_dataset = FreeDataset(args.dataset_val, config.dataset['class_names'], 
                    hyp, config.dataset['img_size'])
        test_batch_size = min(args.batch_size, len(test_dataset))
        test_train_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if args.local_rank != -1 else None
        test_data_loader = data.DataLoader(test_dataset, test_batch_size, num_workers=args.num_workers,
            sampler=test_train_sampler, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    # net
    net = FreeYoloV5().to(device)

    # cmbn归一化 更新权重总batch大小
    nominal_batch_size = 64
    ''' 积累损失次数 CmBN的实现
        BN是在每个batch中计算均值和方差，但这个均值和方差是对应这个batch样本的，这就导致每个batch的均值和方差不一样，尤其batch较
        小时，更容易导致均值和方差波动，进而导致准确率下降。。故提出了CBN，它通过收集最近几次迭代信息来更新当前的均值和方差，变相
        的扩大了batch。。CmBN是参考CBN的思想，其把大batch内部的4个mini batch当做一个整体，对外隔离。CBN在第t时刻，也会考虑前3个
        时刻的统计量进行汇合，而CmBN操作不会，不再滑动cross,其仅仅在mini batch内部进行汇合操作，保持BN一个batch更新一次可训练参
        数。。
    ''' 
    accumulate = max(round(nominal_batch_size / args.batch_size), 1)
    # 计算优化时的权重衰减超参，原始衰减超参的对应于一个cmbn batch
    hyp['weight_decay'] *= args.batch_size * accumulate / nominal_batch_size
    # 分类损失比重系数调整
    hyp['cls'] *= dataset.classes_number() / 80.

    # optim 按照 weight、bias、bn等分组进行优化
    other_group, weight_group, biass_group = [], [], []
    for k, v in net.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                biass_group.append(v)
            elif '.weight' in k and '.bn' not in k:
                weight_group.append(v)
            else:
                other_group.append(v)

    if args.adam:
        optimizer = optim.Adam(other_group, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(other_group, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': weight_group, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': biass_group})  # add pg2 (biases)
    LOG.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(biass_group), len(weight_group), len(other_group)))
    del other_group, weight_group, biass_group

    # 学习率衰减 伪余弦退火法，
    # optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    cos_fun = lambda x : (((1 + math.cos(x * math.pi / args.epochs)) / 2) ** 1.0) * 0.8 + 0.2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = cos_fun)
    scheduler.last_epoch = start_epoch - 1

    ''' 梯度scaler，用于在混合精度训练时，放到loss
        混合精度训练：pytorch 中使用amp模块来实现混合精度训练。在前向和反向是使用的Tensor有torch.FloatTensor和torch.HalfTensor两种。
            因为torch.HalfTensor的优势就是存储小、计算快、更好的利用CUDA设备的Tensor Core。因此训练的时候可以减少显存的占用（可以增加
            batchsize了），同时训练速度更快；但torch.HalfTensor的劣势就是：数值范围小（更容易Overflow / Underflow）、舍入误差（
            Rounding Error，导致一些微小的梯度信息达不到16bit精度的最低分辨率，从而丢失）。
            由以上可见，在有优势的时候使用HalfTensor，有劣势的时候用FloatTensor。如何调整呢，pytorch给出了如下两种方案：
            1、梯度scale，这正是上一小节中提到的torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度的underflow（这只是BP的时候传递
            梯度信息使用，真正更新权重的时候还是要把放大的梯度再unscale回去）；
            2、回落到torch.FloatTensor，这就是混合一词的由来。那怎么知道什么时候用torch.FloatTensor，什么时候用半精度浮点型呢？这是
            PyTorch框架决定的，在PyTorch 1.6的AMP上下文中，如下操作中tensor会被自动转化为半精度浮点型的torch.HalfTensor：
                __matmul__、addbmm、addmm、addmv、addr、baddbmm、bmm、chain_matmul、conv1d、conv2d、conv3d、conv_transpose1d、
                conv_transpose2d、conv_transpose3d、linear、matmul、mm、mv、prelu

        通常使用autocast + GradScaler来实现混合精度训练

        scaler的大小在每次迭代中动态的估计，为了尽可能的减少梯度underflow，scaler应该更大；但是如果太大的话，半精度浮点型的tensor又容
        易overflow（变成inf或者NaN）。所以动态估计的原理就是在不出现inf或者NaN梯度值的情况下尽可能的增大scaler的值——在每次
        scaler.step(optimizer)中，都会检查是否又inf或NaN的梯度出现：
            1、如果出现了inf或者NaN，scaler.step(optimizer)会忽略此次的权重更新（optimizer.step())，并且将scaler的大小缩小
            （乘上backoff_factor）；
            2、如果没有出现inf或者NaN，那么权重正常更新，并且当连续多次（growth_interval指定）没有出现inf或者NaN，则scaler.update()会
            将scaler的大小增加（乘上growth_factor）。
    '''
    scaler = amp.GradScaler(enabled=args.cuda)

    # load weight
    if args.resume:
        net.load_weights(args.resume)
    else:
        net.load_pretrained(args.pretrained)

    # 最大跨度、图片尺寸
    max_stride = int(max(net.stride))
    imgsz, imgsz_test = [check_img_size(x, max_stride) for x in config.dataset['img_size']]

    # 使用混合精度训练
    if mixed_precision:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1', verbosity=0)
    # 分布式数据并行
    if args.cuda and args.local_rank != -1:
        net = DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.cuda and args.local_rank == -1 and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # 指数加权平均，以增加模型鲁棒性
    ema = ModelEMA(net) if args.local_rank in [-1, 0] else None

    # loss
    criterion = FreeLoss(dataset.classes_number()-1)

    batch_size = len(data_loader)
    # 前3个epoch预热
    warmup_size = 3 * batch_size
    mAPs = np.zeros(dataset.classes_number())
    # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    results = (0, 0, 0, 0, 0, 0, 0)
    if args.local_rank in [0, -1]:
        print('Starting training for [%g, %g] epochs...' % (start_epoch, args.epochs))

    ## giou loss ratio (obj_loss = 1.0 or giou)
    net.gr = 1.0
    # train body
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        # 训练模式
        net.train()

        mean_loss = torch.zeros(4, device=device)
        if args.local_rank != -1:
            # 设置采样器epoch
            data_loader.sampler.set_epoch(epoch)

        # 进度条
        pbar = enumerate(data_loader)
        if args.local_rank in [-1, 0]:
            pbar = tqdm(pbar, totall = batch_size)
        LOG.info(('\n' + '%10s   ' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size', 
                                            'precision', 'recall', 'mAP_0.5', 'mAP_0.5:0.95'))

        # 优化器清理
        optimizer.zero_grad()

        # every batch
        for i, (imgs, targets, paths) in pbar:
            ni = i + batch_size * epoch

            # 数据打印，打印前三个batch的图片，一遍核对数据是否正确
            if args.local_rank in [-1, 0] and ni < 3:
                file_name = Path(args.save_path) / ('train_batch%g.jpg' % ni)
                train_img = plot_images(images=imgs, targets=targets, paths=paths, fname=file_name)

            # 预热
            if ni <= warmup_size:
                xi = [0, warmup_size]
                # 预热过程中的，累计损失 np.interp一维的线性插值
                accumulate = max(1, np.interp(ni, xi, [1, nominal_batch_size / args.batch_size]).round())
                # 对优化器的每个参数组进行lr和momentum设置
                for j, x in enumerate(optimizer.param_groups):
                    # 注意这里对bias（偏移量）的lr做了限制 bias_lr [0.1, lr0]; other lr
                    x['lr'] = np.intrep(ni, xi, [0.1 if j ==2 else 0.0, x['initial_lr'] * cos_fun(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # 多尺度训练
            if args.multi_scale:
                sz = random.randrange(config.dataset['img_size'] * 0.5, config.dataset['img_size'] * 1.5 + max_stride) // max_stride * max_stride
                # 尺寸比例
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    # 新的shape
                    ns = [math.ceil(x * sf / max_stride) * max_stride for x in imgs.shape[2:]]
                    # 使用插值法来进行图片尺寸的变换，可以用cv2的方法
                    imgs = F.interpolate(imgs, size = ns, mode='blinear', align_corners = False)

            # 前向
            with amp.autocast(enabled = args.cuda):
                out = net(imgs)
                loss, loss_items = criterion(out, targets, args.anchors, config.hyp['anchors'], config.hyp['anchors_t'], net.gr)
                if args.local_rank != -1:
                    loss *= world_size

            # 反向
            scaler.scale(loss).backward()

            # 优化
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(net)

            if args.local_rank in [-1, 0]:
                mean_loss = (mean_loss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                '''eg:
                    epoch/epochs mem   giou_loss obj_loss  cls_loss   all_loss  target_class imgs
                    41/299     4.14G   0.05477   0.07183   0.03054    0.1571        16       640 
                '''
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, args.epochs - 1), mem, *mean_loss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        if args.local_rank in [-1, 0]:
            if ema:
                ema.update_attr(net, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            if not args.no_test:
                results, maps, times = test.test(args,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 dataloader=test_data_loader,
                                                 save_dir=args.save_path)
             # s precision, recall, mAP, F1, test_losses=(GIoU, obj, cls)
            LOG.info(s + '%10.4g' * 4 % results)

        # 模型保存
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            epoch_model = Path(args.save_path) / 'yolov5_ep%g.pth' % (epoch+1)
            torch.save(ema.ema, epoch_model)
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
            torch.save(ema.ema, best)

    if args.local_rank in [-1, 0]:
        LOG.info('%g epochs completed in %.3f hours = %i.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if args.local_rank not in [-1, 0] else None
    torch.cuda.empty_cache()


if __name__ == '__main__':
    config.log_setting(config.logger["file"], config.logger["level"], config.logger["format"], 
        config.logger["backupCount"], config.logger["interval"])
    LOG.info("*******************train start.*******************")
    LOG.info(args)
    LOG.info(config.hyp)
    LOG.info(config.dataset)
    LOG.info(config.logger)
    if args.labels is None:
        LOG.error("labels is none.")
    LOG.info("***************training***************")
    train()
    LOG.info("*******************exited.*******************")