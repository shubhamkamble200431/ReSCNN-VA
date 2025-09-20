# cpu_inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import cv2
from config import cpu_config
from utils import save_emotion_json

# ------------------------------
# Define EmotionCNN (3-channel input to match trained model)
# ------------------------------
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

# ------------------------------
# Profiling Start
# ------------------------------
start_total = time.perf_counter()
device = torch.device(cpu_config["device"])

model_path = cpu_config["model_path"]
print(f"\n[CPU Inference] Using model path: {model_path}")

# --- Model Loading ---
start_load = time.perf_counter()
checkpoint = torch.load(model_path, map_location=device)
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
end_load = time.perf_counter()

# --- Preprocessing ---
start_preprocess = time.perf_counter()
# Downsample to 48x48 for model
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

img = Image.open(cpu_config["image_path"])
img_tensor = transform(img).unsqueeze(0).to(device)

# Expand grayscale to 3 channels if needed
if img_tensor.shape[1] == 1:
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
end_preprocess = time.perf_counter()

# --- Inference ---
start_inference = time.perf_counter()
with torch.no_grad():
    output = model(img_tensor)
    pred = torch.argmax(output, dim=1).item()
end_inference = time.perf_counter()

# --- Interpretation ---
start_interpret = time.perf_counter()
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_emotion = classes[pred]
end_interpret = time.perf_counter()

# --- Save JSON ---
start_save = time.perf_counter()
save_emotion_json(predicted_emotion, cpu_config["output_json"])
end_save = time.perf_counter()

end_total = time.perf_counter()

# ------------------------------
# Resize original image to 1280x960 for annotation
# ------------------------------
img_cv = cv2.imread(cpu_config["image_path"])
img_cv = cv2.resize(img_cv, (960, 1280))

# Haarcascade Face Detection
face_cascade = cv2.CascadeClassifier(cpu_config["haar_cascade_path"])
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw bounding boxes + label (red)
for (x, y, w, h) in faces:
    cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 0, 255), 6)
    cv2.putText(img_cv, predicted_emotion, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

# Prepare fullform timing text (each on a new line)
timing_lines = [
    f"Model Load Time     : {end_load - start_load:.3f} s",
    f"Preprocessing Time  : {end_preprocess - start_preprocess:.3f} s",
    f"Inference Time      : {end_inference - start_inference:.3f} s",
    f"Total Execution Time: {end_total - start_total:.3f} s"
]

# Draw timings at bottom-left, one line per timing (green)
font_scale = 1.5
thickness = 5
line_height = 45
text_x = 10
text_y = img_cv.shape[0] - 10 - (len(timing_lines)-1)*line_height

for i, line in enumerate(timing_lines):
    y = text_y + i * line_height
    cv2.putText(img_cv, line, (text_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

cv2.imwrite(cpu_config["output_image"], img_cv)
print(f"[CPU Inference] Annotated image with timings saved to {cpu_config['output_image']}")
