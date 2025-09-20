# rpi_inference.py
import torch
from torchvision import transforms
from PIL import Image
import time
import cv2
from config import rpi_config
from utils import save_emotion_json

# --- Select model ---
active_variant = rpi_config["active_model"]
model_path = rpi_config["model_path"]
device = torch.device(rpi_config["device"])

print(f"\n[RPI Inference] Using model variant: {active_variant.upper()} -> {model_path}")

# --- Timing setup ---
start_total = time.perf_counter()

# Load TorchScript model (only TorchScript supported on Pi)
start_load = time.perf_counter()
model = torch.jit.load(model_path, map_location=device)

if active_variant == "fp16":
    model = model.half()

model.eval()
end_load = time.perf_counter()

# --- Preprocess ---
start_preprocess = time.perf_counter()
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
img = Image.open(rpi_config["image_path"])
img = transform(img).unsqueeze(0).to(device)

if active_variant == "fp16":
    img = img.half()

end_preprocess = time.perf_counter()

# --- Inference ---
start_inference = time.perf_counter()
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()
end_inference = time.perf_counter()

# --- Interpretation ---
start_interpret = time.perf_counter()
classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
predicted_emotion = classes[pred]
end_interpret = time.perf_counter()

# --- Save JSON ---
start_save = time.perf_counter()
save_emotion_json(predicted_emotion, rpi_config["output_json"])
end_save = time.perf_counter()

end_total = time.perf_counter()

# --- Print timings ---
print("\n[Profiling Results - Raspberry Pi Inference]")
print(f"Model Load Time     : {end_load - start_load:.4f} sec")
print(f"Preprocessing Time  : {end_preprocess - start_preprocess:.4f} sec")
print(f"Inference Time      : {end_inference - start_inference:.4f} sec")
print(f"Interpretation Time : {end_interpret - start_interpret:.4f} sec")
print(f"Save JSON Time      : {end_save - start_save:.4f} sec")
print(f"Total Execution Time: {end_total - start_total:.4f} sec\n")

# --- Haarcascade Face Detection + Annotation ---
face_cascade = cv2.CascadeClassifier(rpi_config["haar_cascade_path"])
img_cv = cv2.imread(rpi_config["image_path"])
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(img_cv, predicted_emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv2.imwrite(rpi_config["output_image"], img_cv)

print(f"[RPI Inference] Predicted Emotion: {predicted_emotion}")
print(f"[RPI Inference] Annotated image saved to {rpi_config['output_image']}")
