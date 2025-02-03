# Basic building block

# ResNet-18, ResNet-34
## import packages
import torch
from torch import Tensor
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion_factor = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        ## in_channels: 들어가는 이미지의 차원
        ## out_channels: 출력되는 이미지의 차원
        ## kernel: 기본 3으로 연산
        ## padding: 이미지 주변에 추가 -> 1을 기본값으로
        ## stride: window가 몇 단위로 움직이는지 -> 특징 추출 속도를 조절한다 -> 1을 기본값으로
        
        # 1번 신경망
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels) ## batch normalization 진행
        self.relu1 = nn.ReLU() ## relu activation function
        
        ## 2번 신경망 -> 1과 같은 형태인데, out_channel 사이즈의 input, output으로 동일하고, stride는 1이라는 특징
        ## Residual의 특징 상, 결과 값의 이전의 output값과 더해져야 하기 때문에 필수로 같아야 한다.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion_factor))
    
    ## input channel 만 넣어줘도, 자동으로 output channel은 인지한다.
    def forward(self, x: Tensor) -> Tensor:
        out = x # 64D -> out : 64dim
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x -> 32
        x += self.residual(out) # 32D + 64D/2
        x = self.relu2(x)
        return x
    



# BottleNeck
# ResNet-50, ResNet-101, ResNet-152
class BottleNeck(nn.Module):
    expansion_factor = 4 ## 원래 채널 수보다 확장된 채널 수로 변환.
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)  # ?, 64
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU() # 64
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # 64
        self.bn2 = nn.BatchNorm2d(out_channels) # 64
        self.relu2 = nn.ReLU()
        
        ## 원래 channel 보다 확장된 수로 변환 -> 기본 출력 채널 수의 4배 (expansion_factor=4)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size=1, stride=1, bias=False) # 64, 64 * e(4)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion_factor) 
        
        self.relu3 = nn.ReLU()
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion_factor))
        
    def forward(self, x:Tensor) -> Tensor:
        out = x  # ?
        x = self.conv1(x)  # 64
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x) # 64
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x) # 256
        x = self.bn3(x)
        
        x += self.residual(out)
        x = self.relu3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64  
# 	•	입력 채널: 3 (RGB 이미지의 3채널)
# 	•	출력 채널: 64 (초기 필터 개수)
# 	•	커널 크기: 7x7 (큰 수용 영역을 갖도록 설정)
# 	•	스트라이드: 2 (특성 맵 크기를 절반으로 감소)
# 	•	패딩: 3 (크기를 유지하도록 설정)
# 	•	바이어스: False (BatchNorm 적용 시 바이어스를 제거)
        # 목적:
# 	•	입력 이미지(예: 224x224x3)에서 큰 크기의 필터(7x7)를 적용하여 전역적인 특징을 추출
# 	•	stride=2로 크기를 감소시켜 계산량을 줄이고, 특성을 축소
# 	•	패딩을 3으로 설정하여 출력 크기 유지(224 → 112).

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) ## 맵 크기를 추가로 감소

        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) ## 최종적으로 출력 크기를 (1,1)로 변환하여 fully connected layer에 전달할 준비를 함.
        self.fc = nn.Linear(512 * block.expansion_factor, num_classes) ## Fully Connected Layer -> num_classes 크기의 출력 생성

        self._init_layer()
    ## 주어진 Bottleneck Block(block)을 여러 개 쌓는 역할
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1) # 첫 번째 블록에서는 stride 값을 사용하여 다운샘플링 수행.
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers) ## list들의 값들을 순차적으로 

    def _init_layer(self): ## 가중치 초기화
        for m in self.modules(): ## 합성곱 레이어의 가중치를 ReLU 활성화 함수에 맞춰 초기화하여 학습 안정성 향상.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1) ## (N(batch), 512 * expansion_factor)
        x = self.fc(x)
        return x
    


class Model:
    def resnet18(self):
        return ResNet(BasicBlock, [2, 2, 2, 2])

    def resnet34(self):
        return ResNet(BasicBlock, [3, 4, 6, 3])

    def resnet50(self):
        return ResNet(BottleNeck, [3, 4, 6, 3])

    def resnet101(self):
        return ResNet(BottleNeck, [3, 4, 23, 3])

    def resnet152(self):
        return ResNet(BottleNeck, [3, 8, 36, 3])
    


# model = Model().resnet50()
# y = model(torch.randn(1, 3, 224, 224))
# print (y.size()) # torch.Size([1, 10])






