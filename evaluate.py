import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms

# --- 关键修正：导入我们自己写的类，而不是原项目的函数 ---
from model import Net as GTSRBModel
from data import GTSRBDataset

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
DATA_DIR = './gtsrb-german-traffic-sign'

# 检查模型文件是否存在，防止路径错误
# 如果你之前的 train.py 保存的是 best_model.pth，这里会自动适配
if os.path.exists('./model/best_stn_model.pth'):
    MODEL_PATH = './model/best_stn_model.pth'
else:
    MODEL_PATH = './model/best_model.pth'


def evaluate():
    # 1. 准备测试数据 (必须与训练时的预处理完全一致)
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    # 加载测试集 (train=False)
    # 注意：确保 GTSRBDataset 类在 data.py 中已正确定义
    test_dataset = GTSRBDataset(root_dir=DATA_DIR, train=False, transform=data_transforms)

    # Windows 下 num_workers=0 以防死锁
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"测试集加载完成，共 {len(test_dataset)} 张图片")
    print(f"正在加载模型: {MODEL_PATH}")

    # 2. 加载模型
    model = GTSRBModel().to(DEVICE)

    # 加载训练好的权重
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    model.eval()  # 切换到评估模式 (关闭 Dropout 等)

    # 3. 预测循环
    all_preds = []
    all_labels = []

    print("正在进行推理评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # 获取概率最高的类别

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"================================")
    print(f"最终测试集准确率: {acc * 100:.2f}%")
    print(f"================================")

    # 5. 生成混淆矩阵
    print("正在绘制混淆矩阵...")
    cm = confusion_matrix(all_labels, all_preds)

    # 设置大图，确保43个类别都能看清
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.title(f"Confusion Matrix (Test Acc: {acc * 100:.2f}%)")
    plt.savefig('confusion_matrix.png')
    print("成功！混淆矩阵已保存为 'confusion_matrix.png'")


if __name__ == '__main__':
    evaluate()
"""
from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from data import initialize_data # data.py in the same folder
from model import Net
import torchvision


parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='pred.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()

state_dict = torch.load(args.model)
model = Net()
model.load_state_dict(state_dict)
model.eval()

from data import data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale


test_dir = args.data + '/test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transforms = [data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center]
output_file = open(args.outfile, "w")
output_file.write("Filename,ClassId\n")

for f in tqdm(os.listdir(test_dir)):
    if 'ppm' in f:
        output = torch.zeros([1, 43], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0,len(transforms)):
                data = transforms[i](pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data)
                output = output.add(model(data))
            pred = output.data.max(1, keepdim=True)[1]
            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

            

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle '
      'competition at https://www.kaggle.com/c/nyu-cv-fall-2017/')"""


