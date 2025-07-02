# train_rafdb.py
import os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

# ---------------- 参数读取 ----------------
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", 32))
LR          = float(os.getenv("LR", 1e-4))
EPOCHS      = int(os.getenv("EPOCHS", 20))
PRETRAINED  = os.getenv("PRETRAINED", "/home/sriensty/face_training/model/MobileNetv2/mobilenetv2_emotion.pth")
DATA_ROOT   = Path(os.getenv("DATA_DIR", "/home/sriensty/face_training/data/RAF-DB_data/RAF-DB/basic"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 7))
MODEL_OUT   = os.getenv("MODEL_OUT", "/home/sriensty/face_training/model/MobileNetv2/rafdb_mobilenetv2.pth")
VAL_SPLIT   = float(os.getenv("VAL_SPLIT", 0.1))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- RAF‑DB Dataset ----------------
class RAFDB(Dataset):
    """RAF‑DB 读取器：把 train_00001.jpg → train_00001_aligned.jpg"""
    def __init__(self, root: Path, transform=None):
        img_root   = root / "Image" / "aligned"
        label_file = root / "EmoLabel" / "list_patition_label.txt"
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                fname, label = parts
                # 加上 "_aligned" 后缀
                fname_aligned = fname.replace('.jpg', '_aligned.jpg')
                img_path = img_root / fname_aligned
                if img_path.exists():
                    self.samples.append((img_path, int(label)-1))
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {label_file}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------- 数据预处理 & 增强 ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------------- 构建数据集 & 随机划分 ----------------
full_ds = RAFDB(DATA_ROOT, transform=transform)
n_total = len(full_ds)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- 模型构建 ----------------
model = models.mobilenet_v2(pretrained=False)
checkpoint = torch.load(PRETRAINED, map_location=DEVICE)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(checkpoint, strict=False)
model = model.to(DEVICE)

# ---------------- 损失与优化 ----------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------- 训练循环 ----------------
best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    # 训练
    print(f"Epoch {epoch}/{EPOCHS}")sudo apt install git-lfs
git lfs install

    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs)
        loss  = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            correct += (preds.argmax(dim=1) == labels).sum().item()
    val_acc = correct / len(val_ds)

    print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"→ Saved best model (val_acc={best_acc:.4f}) to {MODEL_OUT}")

print("Training complete. Best val_acc =", best_acc)
