import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ---------------- MODEL ----------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv2d_input = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_normalization = nn.BatchNorm2d(32)
        self.conv2d_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_normalization_1 = nn.BatchNorm2d(64)
        self.max_pooling2d = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_normalization_2 = nn.BatchNorm2d(128)
        self.conv2d_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_normalization_3 = nn.BatchNorm2d(128)
        self.max_pooling2d_1 = nn.MaxPool2d(2, 2)
        self.dropout_1 = nn.Dropout(0.25)

        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_normalization_4 = nn.BatchNorm2d(256)
        self.conv2d_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_normalization_5 = nn.BatchNorm2d(256)
        self.max_pooling2d_2 = nn.MaxPool2d(2, 2)
        self.dropout_2 = nn.Dropout(0.25)

        self.conv2d_6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_normalization_6 = nn.BatchNorm2d(512)
        self.conv2d_7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_normalization_7 = nn.BatchNorm2d(512)
        self.max_pooling2d_3 = nn.MaxPool2d(2, 2)
        self.dropout_3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(512*3*3, 512)
        self.batch_normalization_8 = nn.BatchNorm1d(512)
        self.dropout_4 = nn.Dropout(0.5)
        self.dense_1 = nn.Linear(512, 256)
        self.dense_2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_normalization(self.conv2d_input(x)))
        x = self.relu(self.batch_normalization_1(self.conv2d_1(x)))
        x = self.max_pooling2d(x)
        x = self.dropout(x)
        
        x = self.relu(self.batch_normalization_2(self.conv2d_2(x)))
        x = self.relu(self.batch_normalization_3(self.conv2d_3(x)))
        x = self.max_pooling2d_1(x)
        x = self.dropout_1(x)
        
        x = self.relu(self.batch_normalization_4(self.conv2d_4(x)))
        x = self.relu(self.batch_normalization_5(self.conv2d_5(x)))
        x = self.max_pooling2d_2(x)
        x = self.dropout_2(x)
        
        x = self.relu(self.batch_normalization_6(self.conv2d_6(x)))
        x = self.relu(self.batch_normalization_7(self.conv2d_7(x)))
        x = self.max_pooling2d_3(x)
        x = self.dropout_3(x)
        
        x = self.flatten(x)
        x = self.relu(self.batch_normalization_8(self.dense(x)))
        x = self.dropout_4(x)
        x = self.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x


if __name__ == "__main__":
    num_classes = 7
    model = EmotionCNN(num_classes=num_classes)

    # Load trained weights
    checkpoint = torch.load("/home/ml/Desktop/shubham/sensors/RS/models/best_emotion_model_fp32.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Save TorchScript versions
    scripted_model = torch.jit.script(model)
    scripted_model.save("/home/ml/Desktop/shubham/sensors/RS/models/emotion_cnn_scripted_fp32.pt")

    traced_model = torch.jit.trace(model, torch.randn(1, 3, 48, 48))
    traced_model.save("/home/ml/Desktop/shubham/sensors/RS/models/emotion_cnn_traced_fp32.pt")

    print("✅ TorchScript models saved.")

    # ---------------- Test with unseen image ----------------
    img_path = "RS/samples/happy.jpg"   # <--- change path to your test image
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])

    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # add batch dim

    # Load back scripted model for inference
    ts_model = torch.jit.load("/home/ml/Desktop/shubham/sensors/RS/models/emotion_cnn_scripted_fp32.pt")
    ts_model.eval()

    with torch.no_grad():
        output = ts_model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    print(f"Predicted class index: {pred_class}")
