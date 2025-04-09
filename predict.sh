#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python predict.py      \
    --mode=predict                              \
    --cuda=True                                 \
    --model-path=model_data/unet_vgg_voc.pth    \
    --num-classes=21                            \
    --backbone=vgg                              \
    --input-shape=512,512                       \
    --mix-type=0                                \
    --onnx-path=model_data/models.onnx          \
    --dir-input-path=img                        \
    --dir-output-path=output                    \
    --simplify=True                             \
    --onnx-save-path=model_data/models.onnx
