#!/bin/bash

# 실행 옵션 설정
GPU=1 # 사용할 GPU (0 또는 1)
PRETRAINED=false  # true: Pretrained 모델 사용, false: Scratch 모델 사용
PATIENCE=5  # Early Stopping patience

# 명령어 실행
if [ "$PRETRAINED" = true ]; then
    python train_augmentation.py --pretrained --gpu $GPU --patience $PATIENCE
else
    python train_augmentation.py --gpu $GPU --patience $PATIENCE
fi


# ./train.sh

## 학습:

# Augmentation:
## pretrained model + finetuning:
## non-pretrained model + finetuning

# Non-Augmentation:
## pretrained model + finetuning:
## non-pretrained model + finetuning