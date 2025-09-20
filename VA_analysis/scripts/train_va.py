import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image

# ====================
# Config
# ====================
class Config:
    train_dir = "/home/ml/Desktop/shubham/sensors/VA_analysis/va_data/YOLO_format/train"
    val_dir = "/home/ml/Desktop/shubham/sensors/VA_analysis/va_data/YOLO_format/valid"
    test_dir = "/home/ml/Desktop/shubham/sensors/VA_analysis/va_data/YOLO_format/test"
    save_path = "/home/ml/Desktop/shubham/sensors/VA_analysis/models/best_model_mobilenetv3.pth"
    img_size = 48
    batch_size = 128
    epochs = 100
    lr = 3e-5
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ====================
# CCC Loss
# ====================
def ccc_loss(y_true, y_pred):
    y_true_mean = torch.mean(y_true, dim=0)
    y_pred_mean = torch.mean(y_pred, dim=0)

    cov = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=0)
    var_true = torch.var(y_true, dim=0)
    var_pred = torch.var(y_pred, dim=0)

    ccc = (2 * cov) / (var_true + var_pred + (y_true_mean - y_pred_mean) ** 2 + 1e-8)
    return 1 - torch.mean(ccc)


# ====================
# Model: MobileNetV3 Small
# ====================
class LightweightVA(nn.Module):
    def __init__(self, n_outputs=2, img_size=48):
        super(LightweightVA, self).__init__()
        self.base = models.mobilenet_v3_small(weights=None)
        # Grayscale first conv
        self.base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # Dynamically compute in_features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            x = self.base.features(dummy)
            x = x.mean([2, 3])
            in_features = x.shape[1]

        # Replace classifier
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_outputs),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.base.features(x)
        x = x.mean([2, 3])
        x = self.base.classifier(x)
        return x


# ====================
# Dataset Loader
# ====================
class AffectNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")  # YOLO-style labels
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith(".png")]
        self.transform = transform

        # Keep only files that have corresponding label
        self.files = [f for f in self.files if os.path.exists(
            os.path.join(self.label_dir, f.replace(".png", ".txt"))
        )]

        print(f"✅ Found {len(self.files)} samples in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))

        image = Image.open(img_path).convert("L")

        with open(label_path, "r") as f:
            vals = f.readline().strip().split()
            # YOLO format: class x_center y_center width height
            valence, arousal = float(vals[1]), float(vals[2])  

        # Normalize to [-1,1]
        valence = (valence - 0.5) * 2
        arousal = (arousal - 0.5) * 2

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([valence, arousal], dtype=torch.float32)


# ====================
# Training
# ====================
def train():
    cfg = Config()

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = AffectNetDataset(cfg.train_dir, transform)
    val_dataset = AffectNetDataset(cfg.val_dir, transform)
    test_dataset = AffectNetDataset(cfg.test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    model = LightweightVA(img_size=cfg.img_size).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse = nn.MSELoss()
    scaler = GradScaler(device=cfg.device)

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()

            with autocast(device_type=cfg.device):
                outputs = model(imgs)
                loss = mse(outputs, labels) + 0.5 * ccc_loss(labels, outputs)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                outputs = model(imgs)
                loss = mse(outputs, labels) + 0.5 * ccc_loss(labels, outputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{cfg.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            print(f"✅ Best model updated at epoch {epoch+1}")

    # Final Test evaluation
    best_model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            outputs = best_model(imgs)
            loss = mse(outputs, labels) + 0.5 * ccc_loss(labels, outputs)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"📊 Final Test Loss: {avg_test_loss:.4f}")

    # Save full model
    torch.save({
        "model": best_model,
        "config": cfg.__dict__,
        "test_loss": avg_test_loss
    }, cfg.save_path)
    print(f"💾 Full model saved to {cfg.save_path}")


if __name__ == "__main__":
    train()
