import argparse
import yaml
import torch
import wandb
import os


parser = argparse.ArgumentParser(description="ViT Training Script")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config(args.config)

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()  # 현재 GPU ID 가져오기
    DEVICE = torch.device(f"cuda:{device_id}")
    print(f"Using GPU: cuda:{device_id}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

from dataloader import get_dataloaders
from models import get_model

# ✅ 모델 & 데이터 로더
model = get_model(config, DEVICE, is_train=True)  # Training에서는 학습된 모델 불러오지 않음

train_loader, test_loader = get_dataloaders(config)

model_save_path = config["inference"]["model_path"]

model_save_dir = os.path.dirname(model_save_path)
os.makedirs(model_save_dir, exist_ok=True)

from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(**config["training_args"]),
    train_dataset=train_loader.dataset,
    eval_dataset=test_loader.dataset
)

if __name__ == "__main__":
    trainer.train()
    
    print(f"✅ Saving trained model to {model_save_path}...")
    model.save_pretrained(model_save_dir)  # 모델 저장
    torch.save(model.state_dict(), model_save_path)  # 가중치 저장
    print("✅ Model saved successfully!")