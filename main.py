import torch
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime  # ONNX 推論
import numpy as np

# 參數設定
image_path = "test.jpg"         # 測試圖片路徑
model_pt_path = "../models/smoke_model.pt"  # PyTorch 模型權重
model_onnx_path = "../models/smoke_model.onnx"  # ONNX 模型權重
image_size = 224

# 圖片前處理（要跟訓練時一樣）
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def predict_pt(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 載入模型結構與權重（同樣ResNet18）
    model = torch.hub.load('pytorch/vision:v0.14.0', 'resnet18', weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2類
    model.load_state_dict(torch.load(model_pt_path, map_location=device))
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    return pred

def predict_onnx(image_path):
    session = onnxruntime.InferenceSession(model_onnx_path)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)
    inputs = {session.get_inputs()[0].name: input_tensor}
    output = session.run(None, inputs)
    pred = np.argmax(output[0])
    return pred

if __name__ == "__main__":
    # 用 PyTorch 推論
    pred_pt = predict_pt(image_path)
    print(f"PyTorch 預測結果：{pred_pt} (0=non_smoking, 1=smoking)")

    # 用 ONNX 推論
    pred_onnx = predict_onnx(image_path)
    print(f"ONNX 預測結果：{pred_onnx} (0=non_smoking, 1=smoking)")
