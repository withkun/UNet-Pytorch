#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python my_predict.py     \
    --mode=predict                              \
    --cuda=True                                 \
    --model-path=logs/best_epoch_weights.pth    \
    --num-classes=2                             \
    --backbone=vgg                              \
    --input-shape=1440,768                      \
    --mix-type=0                                \
    --onnx-path=model_data/models.onnx          \
    --dir-input-path=input                      \
    --dir-output-path=output                    \
    --simplify=True                             \
    --onnx-save-path=model_data/models.onnx
