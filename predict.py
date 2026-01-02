import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 导入模型结构
from model import Net as GTSRBModel

# --- 配置 ---
MODEL_PATH = './model/best_stn_model.pth'  # 确保模型路径正确
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GTSRB 类别映射表 (部分常见类别)
CLASSES = {
    0: '限速 20 km/h',
    1: '限速 30 km/h',
    2: '限速 50 km/h',
    3: '限速 60 km/h',
    4: '限速 70 km/h',
    5: '限速 80 km/h',
    7: '限速 100 km/h',
    8: '限速 120 km/h',
    9: '禁止超车',
    11: '优先权',
    12: '先行权',
    13: '让行 (Yield)',
    14: '停车 (Stop)',
    17: '禁止进入',
    18: '一般危险',
    25: '施工',
    33: '向右转',
    34: '向左转',
    35: '直行',
    # ... 总共43类，这里列举了常见的
}


def predict_image(image_path):
    # 1. 加载模型
    model = GTSRBModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("模型加载成功！")
    else:
        print("错误：找不到模型文件！请先运行 train.py")
        return

    model.eval()  # 切换到评估模式

    # 2. 图片预处理 (必须和训练时完全一样！)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 缩放
        transforms.ToTensor(),  # 转 Tensor
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))  # 归一化
    ])

    # 3. 读取图片
    try:
        original_img = Image.open(image_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)  # 增加 Batch 维度 [1, 3, 32, 32]
    except Exception as e:
        print(f"图片读取失败: {e}")
        return

    # 4. 推理
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output)  # 因为模型输出是 LogSoftmax，取指数变回概率
        prob, predicted_class = torch.max(probabilities, 1)

        class_id = predicted_class.item()
        confidence = prob.item() * 100

    # 5. 显示结果
    class_name = CLASSES.get(class_id, f"未知类别 ID: {class_id}")

    print(f"=============================")
    print(f"预测结果: {class_name}")
    print(f"置信度: {confidence:.2f}%")
    print(f"=============================")

    # 可视化
    plt.imshow(original_img)
    plt.title(f"Pred: {class_name} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    import os

    # --- 在这里修改你要测试的图片路径 ---
    # 你可以把图片放在项目文件夹里，然后改名字
    test_image = './gtsrb-german-traffic-sign/Test/00000.png'

    predict_image(test_image)