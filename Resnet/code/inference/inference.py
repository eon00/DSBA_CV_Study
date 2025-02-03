import os
import torch
import numpy as np
import pandas as pd
import argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from resnet import Model  # 🔹 기존 ResNet 사용
from torchvision.transforms.functional import to_pil_image

# 🔹 Argument 설정
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet50")
parser.add_argument("--gpu", type=int, default=0, choices=[0, 1], help="Choose GPU (0 or 1)")
parser.add_argument("--model_path", type=str, required=False, help="Path to fine-tuned model")
parser.add_argument("--save_path", type=str, default="predictions.csv", help="Path to save predictions")
args = parser.parse_args()

# 🔹 Device 설정
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# 🔹 데이터 로드
data_folder = "../data/data"
test_data = np.load(f"{data_folder}/test_data.npy")  
test_target = np.load(f"{data_folder}/test_target.npy")  

# 🔹 Custom Dataset 클래스
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

# 🔹 데이터 변환 설정
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 🔹 Dataset 및 DataLoader 생성
test_dataset = CustomDataset(test_data, test_target, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 🔹 모델 로드 함수
def load_model(pretrained=True, model_path=None):
    model = Model().resnet50()  # 🔹 기존 ResNet50 사용

    if pretrained:
        print("✅ Using Pretrained ResNet50")
        pretrained_weights = models.resnet50(weights='IMAGENET1K_V1').state_dict()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 🔹 fc 레이어를 10개 클래스로 변경
        pretrained_weights.pop('fc.weight')
        pretrained_weights.pop('fc.bias')
        model.load_state_dict(pretrained_weights, strict=False)  

    if model_path:
        print(f"✅ Loading fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    return model

# 🔹 Inference 함수
def inference(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predictions

if __name__ == "__main__":
    # 모델 로드
    model = load_model(pretrained=args.pretrained, model_path=args.model_path)

    # Inference 실행
    true_labels, predicted_classes = inference(model, test_loader)
    print(f"🔹 Predicted Classes: {predicted_classes}")

    # 결과를 CSV로 저장
    output_df = pd.DataFrame({
        "True Label": true_labels,
        "Predicted Label": predicted_classes
    })

    output_df.to_csv(args.save_path, index=False)
    print(f"✅ Predictions saved to {args.save_path}")