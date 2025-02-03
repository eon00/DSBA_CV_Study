import torch
from torchvision.models import resnet50

# 사전 학습된 가중치 다운로드 (torchvision 모델에서 가져오기)
pretrained_weights = resnet50(pretrained=True).state_dict()

# 본인이 구현한 ResNet50 모델 생성
my_resnet50 = MyResNet50()  # 사용자가 직접 정의한 ResNet50 클래스

# 사전 학습된 가중치 로드
my_resnet50.load_state_dict(pretrained_weights, strict=False)
