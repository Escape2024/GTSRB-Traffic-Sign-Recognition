import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 确保导入正确的类名
from data import GTSRBDataset
from model import Net as GTSRBModel

# --- 全局参数配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 5  # 早停策略阈值
DATA_DIR = './gtsrb-german-traffic-sign'
CHECKPOINT_DIR = './model'
LOG_DIR = './runs/experiment_stn'  # 修改日志名以便区分
# 确保保存模型的文件夹存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#---随机种子--

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 早停策略 (Early Stopping) ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss


# --- 主程序 ---
def main():
    # 1. 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # 严格匹配模型 STN 和 CNN 的输入维度
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    full_dataset = GTSRBDataset(root_dir=DATA_DIR, train=True, transform=data_transforms)

    # 划分验证集 (80% 训练 / 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Dataset Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 2. 模型初始化
    model = GTSRBModel().to(DEVICE)

    # [关键修改] 因为模型最后输出了 log_softmax，这里必须用 NLLLoss
    criterion = nn.NLLLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学习率衰减 (可选，但在STN训练中很有用，防止震荡)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    writer = SummaryWriter(LOG_DIR)
    save_path = os.path.join(CHECKPOINT_DIR, 'best_stn_model.pth')
    early_stopping = EarlyStopping(patience=PATIENCE, path=save_path)

    # 3. 训练循环
    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()  # 启用 BatchNorm 和 Dropout
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', leave=False)
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)  # 输出已经是 LogSoftmax
            loss = criterion(outputs, labels)  # NLLLoss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- Validation ---
        model.eval()  # 关闭 BatchNorm 和 Dropout
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # 学习率调整
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

        # TensorBoard 记录
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    writer.close()
    print("Training Complete.")


if __name__ == '__main__':
    main()