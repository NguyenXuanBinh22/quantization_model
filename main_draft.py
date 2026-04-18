from model.resnet import ResNet18_Weights, resnet18
from ipdb_hook import ipdb_sys_excepthook
from evaluate import predict_sample, run_benchmark
from data_loader import get_dataloader
import torch
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as LearnableFakeQuantize
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.quantization import QConfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

ipdb_sys_excepthook()

if __name__ == '__main__':
    # Giới hạn tập train 2000 ảnh, tập test 200 ảnh để chạy nhanh
    train_loader = get_dataloader(is_train=True, limit_samples=2000)
    test_loader = get_dataloader(is_train=False, limit_samples=200)

    # nhãn 10 lớp của CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Các bước load model và thay đổi lớp fc, giữ nguyên phần còn lại...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # Fine-tuning 10 epoch
    print(f"Training nhanh trên {len(train_loader.dataset)} ảnh mẫu...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10): # Giảm xuống 10 epoch cho nhanh
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} hoàn tất")

    print("\nFloating point model:")

    #evaluate model trước khi quantization
    model.eval()
    predict_sample(model, 'cuda' if torch.cuda.is_available() else 'cpu', classes=cifar10_classes)
    run_benchmark(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')  

    # quantization

    act_learn_qconfig = LearnableFakeQuantize.with_args(
        observer = HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        scale=0.1,
        zero_point=0.0,
        use_grad_scaling=True,
    )

    wgt_learn_qconfig = lambda channels : LearnableFakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        scale=0.1,
        zero_point=0.0,
        use_grad_scaling=True,
        channel_len = channels,
    )

    act_static_qconfig = FakeQuantize.with_args(
        observer = HistogramObserver.with_args(
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        )
    )

    global_qconfig = QConfig(activation=act_static_qconfig, weight=torch.ao.quantization.default_fused_per_channel_wt_fake_quant)
    qconfig_map = QConfigMapping().set_global(global_qconfig)


    # assign learnable qconfig to each module with weight tensors
    for name, module in model.named_modules():
        if hasattr(module, 'out_channels'):
            qconfig = QConfig(activation=act_learn_qconfig, weight=wgt_learn_qconfig(module.out_channels))
            qconfig_map.set_module_name(name, qconfig)

    example_input = torch.rand(1,3,256,256)
    fx_model = prepare_qat_fx(model, qconfig_map, example_input)

    # for name, module in fx_model.named_modules():
    #     if type(module) is FakeQuantize:
    #         print(name)


    fx_model.to(device) # Đảm bảo model mới ở đúng device
    optimizer_qat = torch.optim.Adam(fx_model.parameters(), lr=1e-5)

    print(f"\nStarting QAT Fine-tuning on {device}...")
    fx_model.train()
    for epoch in range(6): # Giảm xuống 10 epoch cho nhanh
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_qat.zero_grad()
            outputs = fx_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_qat.step()

            running_loss += loss.item()
        print(f"QAT Epoch {epoch+1} hoàn tất. Avg Loss: {running_loss/len(train_loader):.4f}")

    print("\nFx graph model:")
    fx_model.eval()
    predict_sample(fx_model, device, classes=cifar10_classes)
    run_benchmark(fx_model, test_loader, device=device)

    # Chuyển đổi sang model đã quantization
    fx_model.eval()
    fx_model.to('cpu') # Chuyển về CPU trước khi convert
    quantized_model = convert_fx(fx_model)

    print("\nConverted Quantized Model:")
    run_benchmark(quantized_model, test_loader, device='cpu')
    xxx