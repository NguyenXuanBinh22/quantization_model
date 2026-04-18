import torch
import torch.nn as nn
from model.resnet import resnet18
from data_loader import get_dataloader
from evaluate import predict_sample, run_benchmark
import os 

def test_model_complete(model_path='weights/resnet18_fp32.pth', device_str=None):
    # 1. Cấu hình thiết bị
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Sử dụng thiết bị: {device}")

    # 2. Khởi tạo cấu trúc model (Phải khớp với file train)
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10) # 10 lớp cho CIFAR-10
    
    # 3. Load trọng số đã train từ File 1
    if not os.path.exists(model_path):
        print(f" Không tìm thấy file trọng số tại {model_path}!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" Đã load thành công model từ {model_path}")

    # 4. Chuẩn bị dữ liệu Test (1000 mẫu để kết quả benchmark ổn định)
    # Chúng ta dùng is_train=False để lấy tập test gốc của CIFAR-10
    test_loader = get_dataloader(is_train=False, limit_samples=500)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n--- BẮT ĐẦU ĐÁNH GIÁ CHI TIẾT ---")
    
    # Sử dụng hàm run_benchmark cũ của bạn để lấy Precision/Recall/F1
    run_benchmark(model, test_loader, device=device)

    print("\n--- CHẠY THỬ TRÊN ẢNH INTERNET ---")
    # Sử dụng hàm predict_sample cũ của bạn để xem kết quả thực tế
    # Lưu ý: Vì model đã train trên CIFAR-10, ảnh con chó trên mạng sẽ được phân loại vào 1 trong 10 lớp này.
    predict_sample(model, str(device), classes=cifar10_classes)

if __name__ == '__main__':
    # Đường dẫn đến file .pth bạn đã lưu từ file train_fp32.py
    WEIGHTS = 'weights/resnet18_fp32.pth'
    
    test_model_complete(model_path=WEIGHTS)