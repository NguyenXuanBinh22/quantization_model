import torch
import torch.nn as nn
from torch.utils.data import Subset
from model.resnet import resnet18, ResNet18_Weights
from data_loader import get_dataloader
from torch.utils.data import random_split, DataLoader
import os


def train_with_val_ratio(total_samples=3000, train_ratio=0.8, epochs=100, lr=1e-4, device='cuda',seed=42):
    g = torch.Generator().manual_seed(seed)

    # 1. Lấy dataloader và subset ban đầu
    full_loader = get_dataloader(is_train=True, limit_samples=total_samples)
    subset_data = full_loader.dataset 
    
    # 2. KHẮC PHỤC LỖI INDEX: 
    # Truy cập vào dataset gốc và tính toán chỉ số tuyệt đối
    # Điều này giúp loại bỏ việc lồng Subset(Subset(...))
    base_dataset = subset_data.dataset
    actual_indices = subset_data.indices
    
    actual_size = len(actual_indices)
    print(f"Dữ liệu thực tế: {actual_size} ảnh")

    # Chia chỉ số theo tỷ lệ
    train_size = int(train_ratio * actual_size)
    val_size = actual_size - train_size
    
    # Chia ngẫu nhiên các chỉ số hiện có
    train_indices, val_indices = random_split(
        actual_indices, 
        [train_size, val_size]
        , generator=g)

    # Tạo các Subset mới trực tiếp từ dataset gốc để tránh lỗi index lồng nhau
    train_set = Subset(base_dataset, train_indices)
    val_set = Subset(base_dataset, val_indices)

    # 3. Tạo DataLoader (Để num_workers=0 để an toàn tuyệt đối trên Windows)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # chạy trên local
    save_path = 'weights/resnet18_fp32.pth'
    os.makedirs('weights', exist_ok=True)

    # chạy trên Kaggle
    # save_path = 'kaggle/working/weights/resnet18_fp32.pth'
    # os.makedirs('kaggle/working/weights', exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                total_val_loss += criterion(model(images), labels).item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"--> Đã lưu model tốt nhất")

if __name__ == '__main__':
    # Bắt buộc phải có khối này trên Windows khi dùng multiprocessing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_with_val_ratio(total_samples=15000, epochs=20, train_ratio=0.8, device=device)

