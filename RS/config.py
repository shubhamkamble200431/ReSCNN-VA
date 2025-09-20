# config.py

# ---------------- ACTIVE MODEL ----------------
# Choose one of: ["fp32", "fp16", "torchscript"]
active_model = "fp32"

# Run benchmarking mode (True = evaluate all variants sequentially)
benchmark_mode = False

# ---------------- MODEL PATHS ----------------
# Only supported variants (no int8)
model_variants = {
    "fp32": "RS/models/best_emotion_model_fp32.pth",     # baseline full precision
    "fp16": "models/best_emotion_model_fp16.pth",     # half precision
    "torchscript": "models/emotion_cnn_scripted_fp32.pt"  # TorchScript for Raspberry Pi / edge
}

# ---------------- GPU CONFIG ----------------
gpu_config = {
    "active_model": active_model,  
    "model_variants": model_variants, 
    "model_path": model_variants[active_model],
    "image_path": "RS/samples/happy.jpg",  # example input for GPU run
    "output_json": f"RS/outputs/gpu_inference_{active_model}.json",
    "output_image": f"RS/outputs/gpu_inference_{active_model}.png",
    "device": "cuda",  # enforce GPU run
    "haar_cascade_path": "RS/models/haarcascade_frontalface_default.xml"
}

# ---------------- CPU CONFIG ----------------
cpu_config = {
    "active_model": active_model,  
    "model_path": model_variants[active_model],
    "image_path": r"D:\PROJECTS\sensors\preprocess\preprocess\cases\25-0.05-1\neutral.jpg",
    "output_json": f"RS/outputs/cpu_inference_{active_model}_imp_0.05.json",
    "output_image": f"RS/outputs/cpu_inference_{active_model}_imp_0.05.png",
    "device": "cpu",
    "haar_cascade_path": "RS/models/haarcascade_frontalface_default.xml"
}

# ---------------- RPI CONFIG ----------------
rpi_config = {
    "active_model": "torchscript",   # ✅ fixed explicitly
    "model_path": model_variants["torchscript"],  # TorchScript only
    "image_path": "samples/test_rpi.jpg",
    "output_json": "outputs/rpi_inference.json",
    "output_image": "outputs/rpi_inference.png",
    "device": "cpu",  # Pi doesn’t have CUDA
    "haar_cascade_path": "models/haarcascade_frontalface_default.xml"
}

# ---------------- SPOTIFY CONFIG ----------------
spotify_config = {
    "client_id": "75761f28a5a94634959f14a89d8c7563",
    "client_secret": "2e6458aea5fe43589181dcc9f098e0ab",
    "redirect_uri": "http://localhost:8888/callback/",
    "playlist_id": "5rjfioSxWkmzlsCdgNeePc",  # extracted from your playlist URL
    "spotify_space_csv": "/home/ml/Desktop/shubham/sensors/RS/Spotify_changed.csv",
    "user_history_csv": "RS/outputs/user_history.csv",
    "recommendations_csv": "RS/outputs/recommendations.csv",
    "top_k": 10
}

# ---------------- EMOTION MAPPING ----------------
# Valence = positivity, Arousal = energy
emotion_mapping = {
    "Happy":    {"valence": 0.9, "arousal": 0.9},
    "Sad":      {"valence": 0.2, "arousal": 0.2},
    "Angry":    {"valence": 0.1, "arousal": 0.8},
    "Fear":     {"valence": 0.2, "arousal": 0.8},
    "Disgust":  {"valence": 0.1, "arousal": 0.7},
    "Neutral":  {"valence": 0.5, "arousal": 0.5},
    "Surprise": {"valence": 0.7, "arousal": 0.9}
}
