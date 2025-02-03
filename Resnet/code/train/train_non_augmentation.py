import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from resnet import Model  

# 1️⃣ Argument 설정
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet50")
parser.add_argument("--gpu", type=int, default=0, choices=[0, 1], help="Choose GPU (0 or 1)")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")


args = parser.parse_args()

# 2️⃣ 모델 저장 폴더 설정
save_dir = "./saved_model"
os.makedirs(save_dir, exist_ok=True)

# 3️⃣ GPU 설정
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    print(f"🔹 Using GPU: cuda:{args.gpu}")
else:
    device = torch.device("cpu")
    print("🔹 CUDA not available, using CPU")

# 4️⃣ 데이터 로드 (numpy)
data_folder = "../data/data"
test_data = np.load(f"{data_folder}/test_data.npy")  
test_target = np.load(f"{data_folder}/test_target.npy")  
train_data = np.load(f"{data_folder}/train_data.npy")  
train_target = np.load(f"{data_folder}/train_target.npy")  

# 5️⃣ PyTorch Dataset 클래스 정의
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

# 6️⃣ 데이터 변환 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 7️⃣ Dataset 및 DataLoader 생성
train_dataset = CustomDataset(train_data, train_target, transform=train_transform)
test_dataset = CustomDataset(test_data, test_target, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 8️⃣ 모델 설정
if args.pretrained:
    print("🔹 Using pretrained ResNet50 model")
    model = Model().resnet50()
    pretrained_weights = models.resnet50(weights='IMAGENET1K_V1').state_dict()
    # ResNet의 fc 레이어를 10개의 클래스로 수정
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # fc 레이어의 가중치만 제외하고 나머지 로드
    pretrained_weights.pop('fc.weight')
    pretrained_weights.pop('fc.bias')

    # 나머지 가중치만 로드
    model.load_state_dict(pretrained_weights, strict=False)  
    model.to(device)

else:
    print("🔹 Using scratch ResNet50 model")
    model = Model().resnet50().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

# ✅ TensorBoard 설정
writer = SummaryWriter(f"runs/ResNet50_Training_non_augmentation_{args.pretrained}_gpu_{args.gpu}_01_30")

# 9️⃣ Early Stopping 설정
best_test_acc = 0
early_stop_counter = 0
patience = args.patience

# 🔥 학습 루프 + 모델 저장 + Early Stopping
num_epochs = 40
step = 0  
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        writer.add_scalar("Train/Loss", loss.item(), step)
        step += 1

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_accuracy = 100 * correct / total
    avg_train_loss = epoch_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    writer.add_scalar("Train/Accuracy", train_accuracy, epoch)

    # 🔹  평가
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")
    writer.add_scalar("Test/Accuracy", test_accuracy, epoch)

    # 🔹 Early Stopping 체크
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_model_wts = model.state_dict()  # Save best model weights
        early_stop_counter = 0  # Reset counter
        print("🔹 New best model found! Reset early stopping counter.")
    else:
        early_stop_counter += 1
        print(f"🔹 No improvement. Early Stopping Counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("🚀 Early Stopping Triggered! Training Stopped.")
        break

# 🔹 Save the best model after training is complete (including early stopping)
if best_model_wts is not None:
    model.load_state_dict(best_model_wts)
    save_path = os.path.join(save_dir, f"best_resnet50_non_augmentation_{args.pretrained}_gpu_{args.gpu}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Best model saved at: {save_path}")

# ✅ TensorBoard 닫기
writer.close()