#!/bin/bash

# 실행할 YAML 설정 파일 지정 (기본값: config.yaml)
CONFIG_FILE="config.yaml"
# Python Inference 실행
python inference.py --config "$CONFIG_FILE"
echo "Inference completed!"
