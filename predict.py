import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 导入模型结构
from model import Net as GTSRBModel

# --- 配置 ---
MODEL_PATH = './model/best_stn_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GTSRB 类别映射表 (改为英文，避免 Matplotlib 显示方框)
CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    11: 'Right-of-way',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    17: 'No entry',
    18: 'General caution',
    25: 'Road work',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    # 如果预测出其他类别，会显示 Class ID
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

    model.eval()

    # 2. 图片预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # 3. 读取图片
    try:
        original_img = Image.open(image_path).convert('RGB')
        img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"图片读取失败: {e}")
        return

    # 4. 推理
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.exp(output)
        prob, predicted_class = torch.max(probabilities, 1)

        class_id = predicted_class.item()
        confidence = prob.item() * 100

    # 5. 显示结果
    class_name = CLASSES.get(class_id, f"Class ID: {class_id}")

    print(f"=============================")
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"=============================")

    # 可视化
    plt.imshow(original_img)
    # 标题改为英文格式
    plt.title(f"Pred: {class_name}\n({confidence:.1f}%)")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 默认测试路径
    test_image = './gtsrb-german-traffic-sign/Test/00000.png'

    # 检查图片是否存在，防止报错
    if os.path.exists(test_image):
        predict_image(test_image)
    else:
        print(f"提示：找不到测试图片 {test_image}")
        print("请手动修改代码中的 'test_image' 变量为你电脑上存在的图片路径。")