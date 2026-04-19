import os
import torch
from model.resnet import resnet18
# from ipdb_hook import ipdb_sys_excepthook
from evaluate import run_benchmark, print_model_information, predict_sample, compare_inference_speed
from data_loader import get_dataloader

# Import các thư viện Quantization
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as LearnableFakeQuantize
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.quantization import QConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization import get_default_qat_qconfig_mapping

# ipdb_sys_excepthook()

def main():
    # 1. Cấu hình & Data Loader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHT_PATH = 'weights/resnet18_fp32.pth'
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Sử dụng num_workers=0 để tránh lỗi multiprocessing trên Windows
    train_loader = get_dataloader(is_train=True, limit_samples=2000)
    test_loader = get_dataloader(is_train=False, limit_samples=200)

    # 2. Load Model FP32 đã huấn luyện từ file
    print(f"--- Đang nạp mô hình FP32 từ {WEIGHT_PATH} ---")
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"Lỗi: Không tìm thấy file {WEIGHT_PATH}. Vui lòng chạy train_fp32.py trước!")
        return
        
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("Đã chuẩn bị xong Floating Point Model. Bắt đầu cấu hình QAT...")

    # 3. Cấu hình Learnable Quantization (QConfig)
    act_learn_qconfig = LearnableFakeQuantize.with_args(
        observer=HistogramObserver,
        quant_min=0, quant_max=255,
        dtype=torch.quint8, qscheme=torch.per_tensor_affine,
        scale=0.1, zero_point=0.0, use_grad_scaling=True,
    )

    wgt_learn_qconfig = lambda channels : LearnableFakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
        scale=0.1, zero_point=0.0, use_grad_scaling=True,
        channel_len=channels,
    )

    act_static_qconfig = FakeQuantize.with_args(
        observer=HistogramObserver.with_args(
            quant_min=0, quant_max=255, 
            dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    )

    # Map QConfig vào Model
    global_qconfig = QConfig(activation=act_static_qconfig, weight=torch.ao.quantization.default_fused_per_channel_wt_fake_quant)
    qconfig_map = QConfigMapping().set_global(global_qconfig)

    for name, module in model.named_modules():
        if hasattr(module, 'out_channels'):
            qconfig = QConfig(activation=act_learn_qconfig, weight=wgt_learn_qconfig(module.out_channels))
            qconfig_map.set_module_name(name, qconfig)

    # 4. Chuẩn bị mô hình QAT (FX Graph)
    example_input = torch.rand(1, 3, 224, 224).to(device) # Resize chuẩn ResNet là 224
    fx_model = prepare_qat_fx(model, qconfig_map, example_input)
    fx_model.to(device)

    # 5. Fine-tuning QAT
    print(f"\n--- Bắt đầu QAT Fine-tuning trên {device} ---")
    optimizer_qat = torch.optim.Adam(fx_model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    fx_model.train()
    num_epochs_qat = 10 # Số epoch QAT
    for epoch in range(num_epochs_qat): 
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_qat.zero_grad()
            outputs = fx_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_qat.step()
            running_loss += loss.item()
        
        print(f"QAT Epoch {epoch+1}/{num_epochs_qat} - Avg Loss: {running_loss/len(train_loader):.4f}")

    # 6. Chuyển đổi sang Quantized Model (INT8)
    print("\n--- Đang chuyển đổi sang INT8 Model ---")
    fx_model.eval()
    fx_model.to('cpu') # Chuyển về CPU để convert và chạy inference INT8
    quantized_model = convert_fx(fx_model)

    # 7. Đánh giá cuối cùng
    print("\nKẾT QUẢ SAU QUANTIZATION (INT8):")
    run_benchmark(quantized_model, test_loader, device='cpu')
    
    # Lưu lại model quantized nếu cần
    torch.save(quantized_model.state_dict(), "weights/resnet18_quantized_int8.pth")
    print("Đã lưu model quantized tại weights/resnet18_quantized_int8.pth")

    # print thông tin về model gốc (FP32)
    print("\nThông tin về mô hình gốc (FP32):")
    print_model_information(model, 'cpu')

    # print thông tin về quantized model
    print("\nThông tin về mô hình sau khi Quantization:")
    print_model_information(quantized_model, 'cpu')
    

    print("\n--- So sánh tốc độ inference giữa FP32 và INT8 ---")
    compare_inference_speed(model, quantized_model, input_size=(1, 3, 224, 224), n_iters=100)

if __name__ == '__main__':
    main()
