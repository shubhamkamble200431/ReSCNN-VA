# convert_to_torchscript.py
import torch
from entire_train_cpu_gpu import EmotionCNN, Config  # Import your CNN + config

# ---------------- CONFIG ----------------
# Choose which model you want to export (FP32 / PTQ / QAT)
model_path = Config.fp32_model_path       # Baseline FP32
# model_path = Config.ptq_model_path      # Uncomment for PTQ
# model_path = Config.qat_model_path      # Uncomment for QAT

# ---------------- LOAD MODEL ----------------
model = EmotionCNN(num_classes=len(Config.class_names))
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ---------------- CONVERT TO TORCHSCRIPT ----------------
dummy_input = torch.randn(1, 3, 48, 48)   # FER2013 images are RGB (3 channels)
traced = torch.jit.trace(model, dummy_input)

# Save TorchScript model
out_path = model_path.replace(".pth", "_ts.pt")
traced.save(out_path)

print(f"[INFO] TorchScript model saved as {out_path}")
