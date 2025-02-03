import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os


# ✅ Argument Parser 설정
parser = argparse.ArgumentParser(description="ViT Inference Script")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
args = parser.parse_args()

# ✅ config 로드 함수
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# ✅ config 로드 및 DEVICE 설정
config = load_config(args.config)

# ✅ DEVICE 설정 (기본값: CPU -> GPU 가능 시 설정)
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()  # 현재 GPU ID 가져오기
    DEVICE = torch.device(f"cuda:{device_id}")
    print(f"Using GPU: cuda:{device_id}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# ✅ 데이터 로더 & 모델 불러오기
from dataloader import get_dataloaders
from models import get_model
from transformers import Trainer, TrainingArguments

# ✅ 모델 로드
model = get_model(config, DEVICE, is_train=False)  # Inference에서는 저장된 모델 불러옴
model.eval()

# ✅ 데이터 로드 (테스트 데이터만 사용)
_, test_loader = get_dataloaders(config)

# ✅ 저장 디렉토리 생성
output_dir = config["inference"].get("output_path", "results")  # ✅ 저장할 디렉토리 설정
os.makedirs(output_dir, exist_ok=True)  # ✅ 디렉토리 없으면 생성

# ✅ Trainer 설정
trainer = Trainer(
    model=model,
    args=TrainingArguments(**config["training_args"]),
    eval_dataset=test_loader.dataset
)

# ✅ Inference 실행
def run_inference():
    print("🔹 Running Inference...")
    predictions = trainer.predict(test_loader.dataset).predictions
    predicted_labels = np.argmax(predictions, axis=1)  # ✅ 확률값에서 가장 높은 클래스로 변환

    # ✅ 결과 CSV 저장
    save_results_to_csv(predicted_labels, test_loader.dataset.targets)

# ✅ 결과를 CSV 파일로 저장
def save_results_to_csv(predicted_labels, true_labels):
    df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_labels
    })
    output_csv_path = f"{output_dir}/inference_results.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Inference results saved to {output_csv_path}")

if __name__ == "__main__":
    run_inference()