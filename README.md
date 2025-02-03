# DSBA_CV_Study


DSBA CV 사전학습 스터디


# 1. Overview
- 목표 : 이미지 모델 학습을 위한 코드를 작성하고 실험을 진행 후 결과물 정리
- 사용 모델 : ResNet, ViT
- Task : 
    이미지 데이터를 이미지 모델을 통해 10개의 class로 분류하기
    모델 성능 높이기 위한 다양한 증강 기법 적용 및 하이퍼파라미터 수정 후 비교

# 2. Data
- Numpy 배열 형식으로 저장된 이미지 데이터셋
- Train_data_npy : (20431, 32, 32, 3)
- Train_target_npy : (20431, )

- Test_ data_npy : (1000, 32, 32, 3)
- Test_target.npy : (1000, )

- 10개의 class 지님


# 3. Models
- ResNet50 : 잔차 학습(Residual Learning)을 활용한 심층 신경망(Deep Neural Network)으로, 50개의 층을 가진 모델 (Scatch 모델 구현)
- ViT-S/16 : 이미지를 16×16 크기의 패치로 나누고, Transformer 모델에 입력하는 구조를 가진 비전 모델(hugging face 참조)



# 4. Experiments

## Setup

### Python Environment
Please setup your python environment and install the required dependencies as:
```bash
# Clone the repository
git clone https://github.com/eon00/DSBA_CV_Study.git
cd DSBA_CV_Study


# if you set up for Resnet:
conda create --name Resnet-dev python=3.10
conda activate Resnet-dev
pip install -r requirements.txt

# if you set up for ViT:
conda create --name ViT-dev python=3.10
conda activate ViT-dev
pip install -r requirements.txt
```

## Resnet & ViT

### Resnet Source Code:
- 먼저 Resnet Directory인 `./Resnet` 폴더로 이동
    - `./Resnet/renet.py`: scratch로 작성된 Resnet 모델이 구현됨
    - `./Resnet/code/train`
        1. `train_{augmentation}.py`: train을 구현한 python 코드, augumentation 여부에 따라 구별되어 있으며(데이터 처리 과정이 다름), pretrained 인자가 false 인 경우 scratch 모델 그 자체를 학습하거나, true인 경우 weight를 가져와서 학습을 진행함.
        2. `train_{augmentation}.sh`: train 코드를 실행하기 위한 bash code로, 3가지의 인자를 입력 가능: 1.PATIENCE: Early Stopping patience 2.PRETRAINED: pretrained 모델 weight를 불러올지 여부, 3.GPU: GPU 설정
    - `./Resnet/code/inference`: Resnet 모델을 inference하기 위한 코드가 구현됨
        1. `inference_{augmentation}.py`: inference를 구현한 python 코드
        2. `inference_{augmentation}.sh`: inference 코드를 실행하기 위한 bash code로, `model_path`를 입력해주지 않을 경우, 학습되지 않은 모델로 학습되며, 모델 경로명을 입력했을 경우에, 학습된 모델을 불러옴
- train을 진행하기 위해서 아래의 코드를 실행한
    ```bash
    cd ./Resnet/code/train
    chmod +x train_augmentation.sh # 내부 option은 설정 필요
    ./train_augmentation.sh
    ```
- inference을 진행하기 위해서 아래의 코드를 실행
    ```bash
    cd ./Resnet/code/inference
    chmod +x inference_augmentation.sh # 내부 option은 설정 필요
    ./inference_augmentation.sh
    ```

### ViT Source Code:
- ViT Directory인 `./ViT`폴더로 이동
    - `./ViT/models.py` : ViT 모델이 구현됨
    - `./ViT/train.py` 
        1. `train.py`: train을 구현한 python 코드, augumentation 여부를 직접 수정함
        2. `{file_name}.sh`: train 코드를 실행하기 위한 bash code
        3. `{config_name}.yaml`: configuration의 option이 담겨있는 yaml파일, Hyper Parameter 설정 가능
    - `./ViT/inference`: ViT 모델을 inference하기 위한 코드가 구현됨
        1. `inference.py`: inference를 구현한 python 코드
        2. `inference.sh`: inference 코드를 실행하기 위한 bash code로, `model_path`를 입력해주지 않을 경우, 학습되지 않은 모델로 학습되며, 모델 경로명을 입력했을 경우에, 학습된 모델을 불러옴
- train을 진행하기 위해서 아래의 코드를 실행
    ```bash
    cd ./Resnet/code/train
    chmod +x train_augmentation.sh # 내부 option은 설정 필요
    ./train_augmentation.sh
    ```
- inference을 진행하기 위해서 아래의 코드를 실행
    ```bash
    cd ./Resnet/code/inference
    chmod +x inference_augmentation.sh # 내부 option은 설정 필요
    ./inference_augmentation.sh
    ```

### Metric Source Code:
- metric을 진행하기 위해서`./{model_name}/metric` 폴더로 이동
- metric을 진행하기 위해서 아래의 코드를 실행
    ```bash
    cd ./Resnet/metric ## 예시
    python metric.py    
    ```
- metric은 총 4가지의 Metric이 구현됨
    1. Accuracy: 전체 샘플 중에서 정답을 맞춘 비율
    2. Precision: 모델이 Positive(예측한 정답)라고 예측한 것 중에서, 실제로 정답인 비율
    3. Recall: 모델이 정답(Positive)인 것 중에서, 얼마나 많이 맞췄는지에 대한 비율
    4. F1_Score: Precision과 Recall의 조화를 고려한 지표로, 두 값을 균형 있게 평가


## 🌲 File Structure
This section provides an overview of the core directories and organizational structure of this study's source code.
```
DSBA_CV_Study
├── README.md
├── Resnet
│   ├── code
│   │   ├── inference
│   │   │   ├── inference.py
│   │   │   ├── inference.sh
│   │   │   ├── inference_augmetation.py
│   │   │   ├── inference_augmetation.sh
│   │   │   ├── inference_non_augmetation.py
│   │   │   └── inference_non_augmetation.sh
│   │   ├── resnet.py
│   │   ├── resnet_pretrained.py
│   │   └── train
│   │       ├── train_augmentation.py
│   │       ├── train_augmentation.sh
│   │       ├── train_non_augmentation.py
│   │       └── train_non_augmetation.sh
│   ├── metric
│   │   └── metric.py
│   ├── requirements.txt
│   ├── result
│   └── saved_model
├── ViT
│   ├── bash
│   │   ├── inference.sh
│   │   ├── inference_results.csv
│   │   ├── run_training.sh
│   │   ├── run_training_not_finetuning.sh
│   │   ├── run_training_not_pretrained.sh
│   │   ├── run_training_not_pretrained_not_finetuning.sh
│   │   └── wandb
│   ├── checkpoints
│   │   ├── vit-training
│   │   │   ├── non_augmentation
│   │   │   ├── not_pretrained
│   │   │   └── pretrained
│   │   └── vit-training-augmentation
│   │       ├── not_pretrained
│   │       └── pretrained
│   ├── config
│   │   ├── augmentation
│   │   │   ├── ViT_not_pretrained_finetuning.yaml
│   │   │   ├── ViT_not_pretrained_not_finetuning.yaml
│   │   │   ├── ViT_pretrained_finetuning.yaml
│   │   │   └── ViT_pretrained_not_finetuning.yaml
│   │   └── non_augmentation
│   │       ├── ViT_not_pretrained_finetuning.yaml
│   │       ├── ViT_not_pretrained_not_finetuning.yaml
│   │       ├── ViT_pretrained_finetuning.yaml
│   │       └── ViT_pretrained_not_finetuning.yaml
│   ├── dataloader.py
│   ├── inference.py
│   ├── metric
│   │   └── metric.py
│   ├── models.py
│   ├── requirements.txt
│   ├── results
│   ├── saved_model
│   │   ├── not_pretrained
│   │   └── pretrained
│   └── train.py
└── data
    ├── data
    ├── data_EDA.ipynb
    └── data_augmentation.ipynb
```


# 📊 Results
| Model                         | Accuracy | Precision | Recall | F1 Score |
|--------------------------------|----------|-----------|--------|----------|
| ResNet50 (w/o pre-trained)     | 0.68   | 0.7200    | 0.6815 | 0.7002   |
| ResNet50 (w/ pre-trained on ImageNet 1k) | 0.68   | 0.7261    | 0.6809 | 0.7028   |

| Model                         | Accuracy | Precision | Recall | F1 Score |
|--------------------------------|----------|-----------|--------|----------|
| ViT-S/16 (w/o pre-trained)     | 0.49   | 0.5777    | 0.4932 | 0.5321   |
| ViT-S/16 (w/ pre-trained on ImageNet 1k) | 0.98   | 0.9824    | 0.9821 | 0.9822   |











