#!/bin/bash

# 기본 설정 파일 (default: config.yaml)
CONFIG_FILE="../config/non_augmentation/ViT_pretrained_finetuning.yaml"

export CUDA_VISIBLE_DEVICES=1


# Python 학습 스크립트 실행
python ../train.py --config "$CONFIG_FILE"

# Python Inference 실행
python ../inference.py --config "$CONFIG_FILE"
