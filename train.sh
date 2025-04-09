#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py     \
    --cuda=True                                 \
    --seed=11                                   \
    --distributed=False                         \
    --sync-bn=False                             \
    --fp16=False                                \
    --num-classes=21                            \
    --backbone=vgg                              \
    --pretrained=False                          \
    --model-path=model_data/unet_vgg_voc.pth    \
    --input-shape=512,512                       \
    --init-epoch=0                              \
    --freeze-epoch=50                           \
    --freeze-batch-size=2                       \
    --unfreeze-epoch=100                        \
    --unfreeze-batch-size=2                     \
    --freeze-train=True                         \
    --init-lr=1e-4                              \
    --optimizer-type=adam                       \
    --momentum=0.9                              \
    --weight-decay=0                            \
    --lr-decay-type=cos                         \
    --save-period=5                             \
    --save-dir=logs                             \
    --eval-flag=True                            \
    --eval-period=5                             \
    --datasets-path=VOCdevkit/VOC2007           \
    --dice-loss=False                           \
    --focal-loss=False                          \
    --num-workers=4 | tee ${DATE_TIME}.log &
