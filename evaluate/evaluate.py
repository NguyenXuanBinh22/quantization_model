from pathlib import Path
import torch
import ssl
from sklearn.metrics import precision_recall_fscore_support
import urllib
from PIL import Image
from torchvision import transforms

def predict_sample(model, device_str: str, classes=None):
    """
    Chạy thử mô hình trên một ảnh duy nhất.
    classes: List tên các lớp (ví dụ: cifar10_classes). 
             Nếu None, hàm sẽ mặc định load từ imagenet_classes.txt.
    """


    # Tải ảnh mẫu (giữ nguyên logic cũ)
    ssl._create_default_https_context = ssl._create_unverified_context
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try: urllib.request.urlretrieve(url, filename)
    except AttributeError: urllib.urlretrieve(url, filename)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_image = Image.open(filename)
    input_batch = preprocess(input_image).unsqueeze(0)

    # Chuyển thiết bị
    if device_str == 'cuda' and torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    else:
        input_batch = input_batch.to('cpu')
        model.to('cpu')

    model.eval()

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Xử lý danh sách nhãn
    if classes is not None:
        categories = classes
    else:
        # Dự phòng load ImageNet nếu không truyền classes
        with open(Path("evaluate/imagenet_classes.txt"), "r") as f:
            categories = [s.strip() for s in f.readlines()]

    # Giới hạn Top-K không vượt quá số lượng lớp hiện có
    top_k = min(5, len(categories))
    topk_prob, topk_catid = torch.topk(probabilities, top_k)

    print(f"\nTop {top_k} Predictions:")
    for i in range(topk_prob.size(0)):
        label = categories[topk_catid[i]]
        prob = topk_prob[i].item()
        print(f"{label}: {prob:.4f}")


def run_benchmark(model, dataloader, device='cpu'):
    # Giữ nguyên logic cũ của bạn nhưng đảm bảo model ở eval mode
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")