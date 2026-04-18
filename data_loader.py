import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset # Thêm Subset

def get_dataloader(root='./data', batch_size=32, is_train=True, use_cifar=True, limit_samples=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if use_cifar:
        dataset = CIFAR10(root=root, train=is_train, download=True, transform=transform)
    else:
        dataset = ImageFolder(root=root, transform=transform)

    # Nếu có yêu cầu giới hạn số lượng ảnh (ví dụ: 100 ảnh)
    if limit_samples is not None and limit_samples < len(dataset):
        # Lấy ra limit_samples chỉ số đầu tiên
        # indices = torch.arange(limit_samples)

        # Lấy ra limit_samples chỉ số ngẫu nhiên
        indices = torch.randperm(len(dataset))[:limit_samples]

        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)