import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # ✅ NumPy 배열 (H, W, C)
        label = self.targets[idx]

        # ✅ NumPy 배열을 PIL 이미지로 변환
        image = transforms.ToPILImage()(image)

        # ✅ 변환 적용
        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,  # ✅ PyTorch Tensor 반환
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ✅ DataLoader 생성 함수
def get_dataloaders(config):
    """config를 인자로 받아 데이터 로드 및 DataLoader 반환"""

    # ✅ 데이터 로드
    data_folder = config["data"]["path"]
    train_data = np.load(f"{data_folder}/train_data.npy")
    train_target = np.load(f"{data_folder}/train_target.npy")
    test_data = np.load(f"{data_folder}/test_data.npy")
    test_target = np.load(f"{data_folder}/test_target.npy")

    # ✅ transform 설정 -> 
    train_transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.2),
        # transforms.RandomRotation(40),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),  # 🔹 PIL → Tensor 변환을 가장 마지막에 배치
        # transforms.RandomErasing(p=0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),  # 🔹 PIL → Tensor 변환을 가장 마지막에 배치
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CustomDataset(train_data, train_target, transform=train_transform)
    test_dataset = CustomDataset(test_data, test_target, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_args"]["per_device_train_batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training_args"]["per_device_eval_batch_size"],
        shuffle=False
    )

    return train_loader, test_loader