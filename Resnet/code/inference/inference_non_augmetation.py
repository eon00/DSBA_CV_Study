import os
import torch
import numpy as np
import argparse
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from resnet import Model  

# Argument 설정
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, choices=[0, 1], help="Choose GPU (0 or 1)")
parser.add_argument("--pretrained", action="store_true", help="Use pretrained model for inference")
parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
args = parser.parse_args()

# GPU 설정
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    print(f"🔹 Using GPU: cuda:{args.gpu}")
else:
    device = torch.device("cpu")
    print("🔹 CUDA not available, using CPU")

# 데이터 로드 (numpy)
data_folder = "../data/data"
test_data = np.load(f"{data_folder}/test_data.npy")  
test_target = np.load(f"{data_folder}/test_target.npy")  

# PyTorch Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  
        label = self.targets[idx]  
        
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# 데이터 변환 정의
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Dataset 및 DataLoader 생성
test_dataset = CustomDataset(test_data, test_target, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 로드
print("🔹 Loading Model...")
model = Model().resnet50()

# 모델 가중치 로드
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()
print(f"✅ Model loaded from {args.model_path}")

# 🔹 Inference 시작
correct, total = 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 최종 Accuracy 출력
accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")

# 결과를 DataFrame으로 저장
df = pd.DataFrame({"True Label": all_labels, "Predicted Label": all_preds})
csv_filename = "inference_results_non_augmentation_True.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ Inference results saved to '{csv_filename}'")

# 결과 반환
def get_predictions():
    return df["True Label"].values, df["Predicted Label"].values  # 실제값, 예측값 반환