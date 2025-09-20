# utils.py
import torch
import torch.nn as nn
import json

# ---------------- CONFIG ----------------
class Config:
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

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


# ---------------- EMOTION MAPPING ----------------
emotion_mapping = {
    "Happy":    {"valence": 0.9, "arousal": 0.9},
    "Sad":      {"valence": 0.2, "arousal": 0.2},
    "Angry":    {"valence": 0.1, "arousal": 0.8},
    "Fear":     {"valence": 0.2, "arousal": 0.8},
    "Disgust":  {"valence": 0.1, "arousal": 0.7},
    "Neutral":  {"valence": 0.5, "arousal": 0.5},
    "Surprise": {"valence": 0.7, "arousal": 0.9}
}

# ---------------- HELPERS ----------------
def load_emotion_model(model_path: str, device: str = "cpu"):
    """
    Load EmotionCNN with pretrained weights.
    """
    model = EmotionCNN(num_classes=len(Config.class_names))
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def save_emotion_json(prediction: str, output_path: str):
    """
    Save prediction + valence/arousal mapping to JSON.
    """
    data = {
        "emotion": prediction,
        "valence": emotion_mapping[prediction]["valence"],
        "arousal": emotion_mapping[prediction]["arousal"]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Saved emotion inference to {output_path}")
