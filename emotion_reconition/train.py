import os
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import matplotlib.pyplot as plt  # ← 用于绘图

# --- 参数读取 ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
LR         = float(os.getenv("LR", 1e-3))
EPOCHS     = int(os.getenv("EPOCHS", 30))
CSV_PATH   = os.getenv("DATA_CSV", "./data/fer2013.csv")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 7))
MODEL_OUT   = os.getenv("MODEL_OUT", "/home/sriensty/face_training/model/MobileNetv2/mobilenetv2_emotion.pth")

# --- 自定义 Dataset 读取 FER2013 CSV ---
class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.transform = transform
        self.pixels   = df['pixels'].values
        self.emotions = df['emotion'].values.astype(int)

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        vals = np.fromstring(self.pixels[idx], sep=' ', dtype=np.uint8)
        img  = Image.fromarray(vals.reshape(48,48)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.emotions[idx]
        return img, label

# --- 数据增强与预处理 ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --- 加载并拆分数据 ---
dataset = FER2013Dataset(CSV_PATH, transform=transform)
n_total = len(dataset)
n_val   = int(0.1 * n_total)
n_train = n_total - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 构建模型 ---
model = models.mobilenet_v2(pretrained=True)
in_f  = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# --- 损失与优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# --- 用于绘图的数据结构 ---
train_losses = []
val_accs     = []

# --- 训练循环 ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss  = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 验证
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            correct += (preds.argmax(dim=1) == labels).sum().item()
    val_acc = correct / n_val
    val_accs.append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

# --- 保存模型 ---
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# --- 绘制并保存结果曲线 ---
epochs = list(range(1, EPOCHS + 1))
fig, ax1 = plt.subplots(figsize=(8,4))

ax1.plot(epochs, train_losses, label='train_loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(epochs, val_accs, color='orange', label='val_acc')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='upper right')

plt.title('Training Loss & Validation Accuracy')
plt.tight_layout()
plt.savefig('results_mobilenetv2.png', dpi=200)
print("Curve saved to results_mobilenetv2.png")
