import argparse
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
训练自己的语义分割模型一定需要注意以下几点:
1. 训练前仔细检查自己的格式是否满足要求, 该库要求数据集格式为VOC格式, 需要准备好的内容有输入图片和标签
   输入图片为.jpg图片, 无需固定大小, 传入训练前会自动进行resize.
   灰度图会自动转成RGB图片进行训练, 无需自己修改.
   输入图片如果后缀非jpg, 需要自己批量转成jpg后再开始训练.

   标签为png图片, 无需固定大小, 传入训练前会自动进行resize.
   由于许多同学的数据集是网络上下载的, 标签格式并不符合, 需要再度处理. 一定要注意！标签的每个像素点的值就是这个像素点所属的种类.
   网上常见的数据集总共对输入图片分两类, 背景的像素点值为0, 目标的像素点值为255. 这样的数据集可以正常运行但是预测是没有效果的！
   需要改成, 背景的像素点值为0, 目标的像素点值为1.
   如果格式有误, 参考: https://github.com/bubbliiiing/segmentation-format-fix

2. 损失值的大小用于判断是否收敛, 比较重要的是有收敛的趋势, 即验证集损失不断下降, 如果验证集损失基本上不改变的话, 模型基本上就收敛了.
   损失值的具体大小并没有什么意义, 大和小只在于损失的计算方式, 并不是接近于0才好. 如果想要让损失好看点, 可以直接到对应的损失函数里面除上10000.
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

3. 训练好的权值文件保存在logs文件夹中, 每一轮迭代(epoch)包含若干训练步长(step), 每个训练步长(step)进行一次梯度下降.
   如果只是训练了几个step是不会保存的, epoch和step的概念要捋清楚一下.
'''
def get_args() -> argparse.Namespace:
    def str2bool(value):
      if isinstance(value, bool):
          return value
      if value.lower() in ("yes", "true", "t", "y", "1"):
          return True
      elif value.lower() in ("no", "false", "f", "n", "0"):
          return False
      else:
          raise argparse.ArgumentTypeError("Boolean value expected")

    def str2ints(value):
        if isinstance(value, int):
            return value
        return list(map(int, value.replace(',', ' ').split()))

    parser = argparse.ArgumentParser(description='Train the UNet on images and masks')
    #---------------------------------#
    #   cuda    是否使用CUDA
    #           没有GPU可以设置成False
    #---------------------------------#
    parser.add_argument('--cuda', type=str2bool, default=True, help='True for CUDA, False for CPU')
    #----------------------------------------------#
    #   seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    #----------------------------------------------#
    parser.add_argument('--seed', type=int, default=11, help='Sets the random seed for training')
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu. CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡.
    #                   Windows系统下默认使用DP模式调用所有显卡, 不支持DDP.
    #   DP模式:
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式:
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    parser.add_argument('--distributed', type=str2bool, default=False, help='For multiple card distribute')
    #---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn, DDP模式多卡可用
    #---------------------------------------------------------------------#
    parser.add_argument('--sync-bn', type=str2bool, default=False, help='sync_bn mode for DDP')
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存. 需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    parser.add_argument('--fp16', type=str2bool, default=False, help='Enables Automatic Mixed Precision (AMP) training')
    #-----------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1, 如2+1
    #-----------------------------------------------------#
    parser.add_argument('--num-classes', type=int, default=21, help='Number of instances class')
    #-----------------------------------------------------#
    #   主干网络选择
    #   vgg
    #   resnet50
    #-----------------------------------------------------#
    parser.add_argument('--backbone', type=str, default='vgg', help='backbone network(vgg/resnet50)')
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重, 此处使用的是主干的权重, 因此是在模型构建的时候进行加载的.
    #                   如果设置了model_path, 则主干的权值无需加载, pretrained的值无意义.
    #                   如果不设置model_path, pretrained = True, 此时仅加载主干开始训练.
    #                   如果不设置model_path, pretrained = False, freeze_train = Fasle, 此时从0开始训练, 且没有冻结主干的过程.
    #----------------------------------------------------------------------------------------------------------------------------#
    parser.add_argument('--pretrained', type=str2bool, default=False, help='Training from a pretrained model')
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README, 可以通过网盘下载. 模型的 预训练权重 对不同数据集是通用的, 因为特征是通用的.
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分, 用于进行特征提取.
    #   预训练权重对于99%的情况都必须要用, 不用的话主干部分的权值太过随机, 特征提取效果不明显, 网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常, 预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作, 可以将model_path设置成logs文件夹下的权值文件, 将已经训练了一部分的权值再次载入.
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数, 来保证模型epoch的连续性.
    #
    #   当model_path = ''的时候不加载整个模型的权值.
    #
    #   此处使用的是整个模型的权重, 因此是在train.py进行加载的, pretrain不影响此处的权值加载.
    #   如果想要让模型从主干的预训练权值开始训练, 则设置model_path = '', pretrain = True, 此时仅加载主干.
    #   如果想要让模型从0开始训练, 则设置model_path = '', pretrain = Fasle, freeze_train = Fasle, 此时从0开始训练, 且没有冻结主干的过程.
    #
    #   一般来讲, 网络从0开始的训练效果会很差, 因为权值太过随机, 特征提取效果不明显, 因此非常. 非常. 非常不建议大家从0开始训练！
    #   如果一定要从0开始, 可以了解imagenet数据集, 首先训练分类模型, 获得网络的主干部分权值, 分类模型的 主干部分 和该模型通用, 基于此进行训练.
    #----------------------------------------------------------------------------------------------------------------------------#
    parser.add_argument('--model-path', type=str, default='model_data/unet_vgg_voc.pth', help='pretrained model path')
    #-----------------------------------------------------#
    #   input_shape     输入图片的大小, 32的倍数
    #-----------------------------------------------------#
    parser.add_argument('--input-shape', type=str2ints, default=[512,512], help='Target image size for training.')

    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段, 分别是冻结阶段和解冻阶段. 设置冻结阶段是为了满足机器性能不足的同学的训练需求.
    #   冻结训练需要的显存较小, 显卡非常差的情况下, 可设置Freeze_Epoch等于UnFreeze_Epoch, 此时仅仅进行冻结训练.
    #
    #   在此提供若干参数设置建议, 各位训练者根据自己的需求进行灵活调整:
    #   (一)从整个模型的预训练权重开始训练:
    #       Adam:
    #           init_epoch = 0, freeze_epoch = 50, unfreeze_epoch = 100, freeze_train = true, optimizer_type = 'adam', init_lr = 1e-4, weight_decay = 0. (冻结)
    #           init_epoch = 0, unfreeze_epoch = 100, freeze_train = false, optimizer_type = 'adam', init_lr = 1e-4, weight_decay = 0. (不冻结)
    #       SGD:
    #           init_epoch = 0, freeze_epoch = 50, unfreeze_epoch = 100, freeze_train = true, optimizer_type = 'sgd', init_lr = 1e-2, weight_decay = 1e-4. (冻结)
    #           init_epoch = 0, unfreeze_epoch = 100, freeze_train = false, optimizer_type = 'sgd', init_lr = 1e-2, weight_decay = 1e-4. (不冻结)
    #       其中: unfreeze_epoch可以在100-300之间调整.
    #   (二)从主干网络的预训练权重开始训练:
    #       Adam:
    #           init_epoch = 0, freeze_epoch = 50, unfreeze_epoch = 100, freeze_train = true, optimizer_type = 'adam', init_lr = 1e-4, weight_decay = 0. (冻结)
    #           init_epoch = 0, unfreeze_epoch = 100, freeze_train = false, optimizer_type = 'adam', init_lr = 1e-4, weight_decay = 0. (不冻结)
    #       SGD:
    #           init_epoch = 0, freeze_epoch = 50, unfreeze_epoch = 120, freeze_train = true, optimizer_type = 'sgd', init_lr = 1e-2, weight_decay = 1e-4. (冻结)
    #           init_epoch = 0, unfreeze_epoch = 120, freeze_train = false, optimizer_type = 'sgd', init_lr = 1e-2, weight_decay = 1e-4. (不冻结)
    #       其中: 由于从主干网络的预训练权重开始训练, 主干的权值不一定适合语义分割, 需要更多的训练跳出局部最优解.
    #             unfreeze_epoch可以在120-300之间调整.
    #             Adam相较于SGD收敛的快一些. 因此unfreeze_epoch理论上可以小一点, 但依然推荐更多的epoch.
    #   (三)batch_size的设置:
    #       在显卡能够接受的范围内, 以大为好. 显存不足与数据集大小无关, 提示显存不足(OOM或者CUDA out of memory)请调小batch_size.
    #       由于resnet50中有BatchNormalization层
    #       当主干为resnet50的时候batch_size不可为1
    #       正常情况下freeze_batch_size建议为unfreeze_batch_size的1-2倍. 不建议设置的差距过大, 因为关系到学习率的自动调整.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了, 特征提取网络不发生改变
    #   占用的显存较小, 仅对网络进行微调
    #   init_epoch          模型当前开始的训练世代, 其值可以大于freeze_epoch, 如设置:
    #                       init_epoch = 60. freeze_epoch = 50. unfreeze_epoch = 100
    #                       会跳过冻结阶段, 直接从60代开始, 并调整对应的学习率.
    #                       (断点续练时使用)
    #   freeze_epoch        模型冻结训练的freeze_epoch
    #                       (当freeze_train=False时失效)
    #   freeze_batch_size   模型冻结训练的batch_size
    #                       (当freeze_train=False时失效)
    #------------------------------------------------------------------#
    parser.add_argument('--init-epoch', type=int, default=0, help='Init epoch')
    parser.add_argument('--freeze-epoch', type=int, default=50, help='Freeze epoch')
    parser.add_argument('--freeze-batch-size', type=int, default=2, help='Freeze batch size')
    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了, 特征提取网络会发生改变
    #   占用的显存较大, 网络所有的参数都会发生改变
    #   unfreeze_epoch          模型总共训练的epoch
    #   unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    parser.add_argument('--unfreeze-epoch', type=int, default=100, help='Unfreeze epoch')
    parser.add_argument('--unfreeze-batch-size', type=int, default=2, help='Unfreeze batch size')
    #------------------------------------------------------------------#
    #   freeze_train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练.
    #------------------------------------------------------------------#
    parser.add_argument('--freeze-train', type=str2bool, default=True, help='Freeze train')

    #------------------------------------------------------------------#
    #   其它训练参数: 学习率. 优化器. 学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  init_lr=1e-4
    #                   当使用SGD优化器时建议设置   init_lr=1e-2
    #   Min_lr          模型的最小学习率, 默认为最大学习率的0.01
    #------------------------------------------------------------------#
    parser.add_argument('--init-lr', type=float, default=1e-4, help='Init learning rate')
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类, 可选的有adam, sgd
    #                   当使用Adam优化器时建议设置  init_lr=1e-4
    #                   当使用SGD优化器时建议设置   init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减, 可防止过拟合
    #                   adam会导致weight_decay错误, 使用adam时建议设置为0.
    #------------------------------------------------------------------#
    parser.add_argument('--optimizer-type', type=str, default='adam', help='optimizer type (adam/sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0, help='weight decay')
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式, 可选的有'step', 'cos'
    #------------------------------------------------------------------#
    parser.add_argument('--lr-decay-type', type=str, default='cos', help='learning rate decay type (step/cos)')
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    parser.add_argument('--save-period', type=int, default=5, help='save period')
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    parser.add_argument('--save-dir', type=str, default='logs', help='save dir')
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估, 评估对象为验证集
    #   eval_period     代表多少个epoch评估一次, 不建议频繁的评估
    #                   评估需要消耗较多的时间, 频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同, 原因有二:
    #   (一)此处获得的mAP为验证集的mAP.
    #   (二)此处设置评估参数较为保守, 目的是加快评估速度.
    #------------------------------------------------------------------#
    parser.add_argument('--eval-flag', type=str2bool, default=True, help='eval flag')
    parser.add_argument('--eval-period', type=int, default=5, help='eval period')

    #------------------------------#
    #   数据集路径
    #------------------------------#
    parser.add_argument('--datasets-path', type=str, default='VOCdevkit/VOC2007', help='VOC datasets path')
    #------------------------------------------------------------------#
    #   建议选项:
    #   种类少(几类)时, 设置为True
    #   种类多(十几类)时, 如果batch_size比较大(10以上), 那么设置为True
    #   种类多(十几类)时, 如果batch_size比较小(10以下), 那么设置为False
    #------------------------------------------------------------------#
    parser.add_argument('--dice-loss', type=str2bool, default=False, help='dice loss')
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #------------------------------------------------------------------#
    parser.add_argument('--focal-loss', type=str2bool, default=False, help='focal loss')
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据, 1代表关闭多线程
    #                   开启后会加快数据读取速度, 但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程, 即GPU运算速度远大于读取图片的速度.
    #------------------------------------------------------------------#
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"args: {vars(args)}\n")

    #------------------------------------------------------------------#
    #   init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  init_lr=1e-4
    #                   当使用SGD优化器时建议设置   init_lr=1e-2
    #   Min_lr          模型的最小学习率, 默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Min_lr              = args.init_lr * 0.01
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值, 默认是平衡的.
    #   设置的话, 注意设置成numpy形式的, 长度和num_classes一样.
    #   如:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([args.num_classes], np.float32)

    seed_everything(args.seed)

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   下载预训练权重
    #----------------------------------------------------#
    if args.pretrained:
        if args.distributed:
            if local_rank == 0:
                download_weights(args.backbone)
            dist.barrier()
        else:
            download_weights(args.backbone)

    model = Unet(num_classes=args.num_classes, pretrained=args.pretrained, backbone=args.backbone).train()
    if not args.pretrained:
        weights_init(model)
    if args.model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README, 百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(args.model_path))

        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        if len(no_load_key) > 0:
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示: head部分没有载入是正常现象, Backbone部分没有载入是错误的.\033[0m")

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(args.save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=args.input_shape)
    else:
        log_dir         = args.save_dir
        loss_history    = None

    #------------------------------------------------------------------#
    #   torch 1.2不支持amp, 建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif args.sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if args.cuda:
        if args.distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(args.datasets_path, "ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.datasets_path, "ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = args.num_classes, backbone = args.backbone, model_path = args.model_path, input_shape = args.input_shape,
            Init_Epoch = args.init_epoch, Freeze_Epoch = args.freeze_epoch, UnFreeze_Epoch = args.unfreeze_epoch, Freeze_batch_size = args.freeze_batch_size, Unfreeze_batch_size = args.unfreeze_batch_size, Freeze_Train = args.freeze_train,
            Init_lr = args.init_lr, Min_lr = Min_lr, optimizer_type = args.optimizer_type, momentum = args.momentum, lr_decay_type = args.lr_decay_type,
            save_period = args.save_period, save_dir = args.save_dir, num_workers = args.num_workers, num_train = num_train, num_val = num_val
        )
    #------------------------------------------------------#
    #   主干特征提取网络特征通用, 冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏.
    #   init_epoch为起始世代
    #   interval_epoch为冻结训练的世代
    #   epoch总训练世代
    #   提示OOM或者显存不足请调小batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if args.freeze_train:
            model.freeze_backbone()

        #-------------------------------------------------------------------#
        #   如果不冻结训练的话, 直接设置batch_size为unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = args.freeze_batch_size if args.freeze_train else args.unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size, 自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 1e-4 if args.optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if args.optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (args.momentum, 0.999), weight_decay = args.weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = args.momentum, nesterov=True, weight_decay = args.weight_decay)
        }[args.optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)

        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小, 无法继续进行训练, 请扩充数据集.")

        train_dataset   = UnetDataset(train_lines, args.input_shape, args.num_classes, True, args.datasets_path)
        val_dataset     = UnetDataset(val_lines, args.input_shape, args.num_classes, False, args.datasets_path)

        if args.distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, pin_memory=True,
                                     drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, pin_memory=True,
                                     drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.datasets_path, log_dir, args.cuda,
                                           eval_flag=args.eval_flag, period=args.eval_period)
        else:
            eval_callback   = None

        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        for epoch in range(args.init_epoch, args.unfreeze_epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻, 并设置参数
            #---------------------------------------#
            if epoch >= args.freeze_epoch and not UnFreeze_flag and args.freeze_train:
                batch_size = args.unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   判断当前batch_size, 自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 1e-4 if args.optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if args.optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.unfreeze_epoch)

                model.unfreeze_backbone()

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小, 无法继续进行训练, 请扩充数据集.")

                if args.distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, pin_memory=True,
                                             drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler,
                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))
                gen_val         = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = args.num_workers, pin_memory=True,
                                             drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler,
                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=args.seed))

                UnFreeze_flag = True

            if args.distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, args.unfreeze_epoch, args.cuda, args.dice_loss, args.focal_loss, cls_weights, args.num_classes, args.fp16, scaler, args.save_period, args.save_dir, local_rank)

            if args.distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
