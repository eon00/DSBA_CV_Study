#!/bin/bash

# 기본 설정
GPU=0
SAVE_PATH="predictions.csv"
MODEL_PATH=""

echo "🔹 Running inference..."

# Pretrained 모델 실행
python inference.py --pretrained --gpu $GPU --save_path "./result/results_pretrained.csv"

# Non-Pretrained (Scratch) 모델 실행
python inference.py --gpu $GPU --save_path "./result/results_scratch.csv"

# Fine-Tuned 모델 실행 (모델 경로가 존재하는 경우)
if [ -f "$MODEL_PATH" ]; then
    echo "🔹 Running inference with fine-tuned model..."
    python inference.py --gpu $GPU --model_path $MODEL_PATH --save_path "results_finetuned.csv"
else
    echo "⚠️ Fine-tuned model not found. Skipping fine-tuned inference."
fi

echo "✅ Inference completed! Results saved."