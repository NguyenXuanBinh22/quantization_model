import os
import torch
import sys

# Import các hàm từ các file bạn đã tách
from train_fp32 import train_with_val_ratio
from test_fp32 import test_model_complete
# from main import main as run_qat_logic

def run_full_pipeline():
    # 1. Khởi tạo tham số hệ thống
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHTS_DIR = 'weights'
    FP32_PATH = os.path.join(WEIGHTS_DIR, 'resnet18_fp32.pth')
    
    print("="*60)
    print("      RESNET18 QUANTIZATION PIPELINE - HUST PROJECT")
    print("="*60)
    print(f"Thiết bị sử dụng: {device.upper()}")

    # Tạo thư mục lưu weights nếu chưa có
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # 2. BƯỚC 1: HUẤN LUYỆN FP32 (BASE MODEL)
    # Kiểm tra xem đã có file weights chưa để tránh train lại lãng phí thời gian

    train_with_val_ratio(
        total_samples=3000, 
        train_ratio=0.8, 
        epochs=100, 
        lr=1e-4, 
        device=device
    )
    print("✅ Huấn luyện FP32 hoàn tất.")


    # 3. BƯỚC 2: KIỂM THỬ FP32 (BASELINE)
    print("\n[BƯỚC 2] Đánh giá chất lượng mô hình Floating Point...")
    try:
        test_model_complete(model_path=FP32_PATH, device_str=device)
    except Exception as e:
        print(f"❌ Lỗi khi test FP32: {e}")
        # Vẫn tiếp tục QAT nếu file weights tồn tại

if __name__ == '__main__':
    # Đảm bảo multiprocessing hoạt động ổn định trên Windows
    # và bọc trong khối main để tránh lỗi lặp vô hạn
    run_full_pipeline()