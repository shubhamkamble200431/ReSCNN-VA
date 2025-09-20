import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# ====================
# Config
# ====================
class Config:
    data_dir = "/home/ml/Desktop/shubham/sensors/dataset/train"  # Folder with 7 subfolders (emotions) containing PNGs
    model_path = "/home/ml/Desktop/shubham/sensors/VA_analysis/models/best_model_va.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 48
    batch_size = 64
    num_workers = 4
    output_dir = "./plots_train"

os.makedirs(Config.output_dir, exist_ok=True)

# ====================
# Model Definition
# ====================
class LightweightVA(nn.Module):
    def __init__(self, n_outputs=2):
        super(LightweightVA, self).__init__()
        self.base = models.mobilenet_v3_small(weights=None)
        self.base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.base.classifier[3].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_outputs),
            nn.Tanh()
        )

    def forward(self, x):
        return self.base(x)

# ====================
# Dataset Loader (FER subfolders)
# ====================
class FERVA_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.emotion_map = {}  # Map folder name → class id
        
        # Check if root directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Data directory not found: {root_dir}")
        
        print(f"Scanning directory: {root_dir}")
        print(f"Directory contents: {os.listdir(root_dir)}")
        
        folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        if not folders:
            raise ValueError(f"No subdirectories found in {root_dir}")
        
        print(f"Found subdirectories: {folders}")
        
        # Supported image extensions (case-insensitive)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        
        for idx, folder in enumerate(sorted(folders)):
            folder_path = os.path.join(root_dir, folder)
            self.emotion_map[folder] = idx
            
            # Get all files in the folder
            all_files = os.listdir(folder_path)
            print(f"Folder '{folder}' contains {len(all_files)} files")
            
            # Find image files (check multiple extensions)
            image_files = []
            for f in all_files:
                file_ext = os.path.splitext(f.lower())[1]
                if file_ext in image_extensions:
                    image_files.append(f)
            
            print(f"  - Found {len(image_files)} image files in '{folder}'")
            if len(image_files) > 0:
                print(f"    Sample files: {image_files[:3]}{'...' if len(image_files) > 3 else ''}")
            
            # Add to samples
            for f in image_files:
                full_path = os.path.join(folder_path, f)
                if os.path.isfile(full_path):  # Double-check file exists
                    self.samples.append((full_path, idx))
        
        print(f"Total samples loaded: {len(self.samples)}")
        
        if not self.samples:
            # Provide more detailed error information
            print("DEBUG INFORMATION:")
            for folder in folders:
                folder_path = os.path.join(root_dir, folder)
                files = os.listdir(folder_path)
                print(f"  {folder}/: {files[:10]}{'...' if len(files) > 10 else ''}")
            
            raise ValueError("No image files found in any subdirectory. Supported formats: PNG, JPG, JPEG, BMP, TIFF, GIF")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, emotion = self.samples[idx]
        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', (48, 48), 0)
            
        if self.transform:
            image = self.transform(image)
        
        # For demonstration: fake VA values as zeros (since no label txts in this structure)
        va = np.array([0.0, 0.0], dtype=np.float32)
        return image, emotion, va

# ====================
# Transforms & Dataloader
# ====================
transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

try:
    dataset = FERVA_Dataset(Config.data_dir, transform)
    print(f"Dataset loaded successfully: {len(dataset)} samples")
    print(f"Emotion mapping: {dataset.emotion_map}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers)

# ====================
# Load Model
# ====================
try:
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(f"Model file not found: {Config.model_path}")
    
    print("Loading model checkpoint...")
    
    # Load checkpoint
    checkpoint = torch.load(Config.model_path, map_location=Config.device, weights_only=False)
    
    # Debug: print checkpoint structure
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check if config is available to understand model architecture
        if 'config' in checkpoint:
            print(f"Model config: {checkpoint['config']}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # The checkpoint contains the actual model object
        print("Loading complete model from 'model' key...")
        model = checkpoint['model']
        
        # Ensure it's the right type
        if not isinstance(model, nn.Module):
            raise ValueError(f"'model' key contains {type(model)}, not a PyTorch model")
            
    elif isinstance(checkpoint, dict):
        # Try to create model and load state dict
        model = LightweightVA(n_outputs=2)
        
        if 'model_state_dict' in checkpoint:
            print("Loading from 'model_state_dict' key...")
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            print("Loading from 'state_dict' key...")
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the entire checkpoint is the state_dict
            print("Loading checkpoint as state_dict...")
            model.load_state_dict(checkpoint)
            
    elif hasattr(checkpoint, 'state_dict'):
        # checkpoint is a model object
        print("Loading from model object...")
        model = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    
    # Move to device and set to eval mode
    model.to(Config.device)
    model.eval()
    print(f"Model loaded successfully on {Config.device}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print model architecture summary
    print(f"Model type: {type(model).__name__}")
    print(f"Model architecture preview:")
    print(f"  Input: Grayscale images (1 channel)")
    print(f"  Output: VA coordinates (2 values)")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\n=== DEBUGGING INFORMATION ===")
    try:
        checkpoint = torch.load(Config.model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dict with keys: {list(checkpoint.keys())}")
            for key, value in checkpoint.items():
                if key == 'model' and hasattr(value, '__class__'):
                    print(f"  {key}: {value.__class__.__name__}")
                    if hasattr(value, 'base') and hasattr(value.base, 'classifier'):
                        try:
                            classifier_shape = value.base.classifier[0].weight.shape if len(value.base.classifier) > 0 else "Unknown"
                            print(f"    Classifier input shape: {classifier_shape}")
                        except:
                            print(f"    Could not determine classifier shape")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print(f"Checkpoint type: {type(checkpoint)}")
    except Exception as debug_e:
        print(f"Could not load checkpoint for debugging: {debug_e}")
    
    print("\n=== SUGGESTED SOLUTIONS ===")
    print("1. The saved model has a different architecture than LightweightVA")
    print("2. Try using the exact same model definition used during training")
    print("3. Or create a new LightweightVA class matching the saved model dimensions")
    exit(1)

# ====================
# Predict VA
# ====================
all_va = []
all_emotions = []

print("Predicting VA values...")
with torch.no_grad():
    for batch_idx, (imgs, emotions, vas) in enumerate(loader):
        imgs = imgs.to(Config.device)
        try:
            outputs = model(imgs).cpu().numpy()
            all_va.append(outputs)
            all_emotions.append(emotions.numpy())
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
        
        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx * Config.batch_size} samples")

if not all_va:
    print("No predictions were made successfully")
    exit(1)

all_va = np.vstack(all_va)
all_emotions = np.hstack(all_emotions)

print(f"Predictions complete: {len(all_va)} samples")
print(f"VA range - Valence: [{all_va[:,0].min():.3f}, {all_va[:,0].max():.3f}]")
print(f"VA range - Arousal: [{all_va[:,1].min():.3f}, {all_va[:,1].max():.3f}]")

# ====================
# 1️⃣ Plot 7 Emotions
# ====================
emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
colors7 = ['red', 'green', 'purple', 'yellow', 'blue', 'orange', 'gray']

# Adjust emotion_names based on actual dataset structure
unique_emotions = np.unique(all_emotions)
if len(unique_emotions) != 7:
    print(f"Warning: Expected 7 emotions, found {len(unique_emotions)}: {unique_emotions}")
    # Use actual folder names if available
    if hasattr(dataset, 'emotion_map'):
        emotion_names = [k for k, v in sorted(dataset.emotion_map.items(), key=lambda x: x[1])]

plt.figure(figsize=(10, 8))
for i in unique_emotions:
    if i < len(emotion_names) and i < len(colors7):
        idxs = np.where(all_emotions == i)[0]
        if len(idxs) > 0:
            plt.scatter(all_va[idxs, 0], all_va[idxs, 1], 
                       label=f"{emotion_names[i]} ({len(idxs)})", 
                       alpha=0.6, s=30, color=colors7[i])

plt.title("FER2013 VA Space - 7 Emotions")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.output_dir, "7_emotions_va.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ====================
# 2️⃣ KMeans Clustering - 4 Moods
# ====================
print("Performing K-means clustering...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_va)
mood_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']  # Generic names
colors4 = ['yellow', 'blue', 'red', 'green']

plt.figure(figsize=(10, 8))
for i in range(4):
    idxs = np.where(cluster_labels == i)[0]
    if len(idxs) > 0:
        plt.scatter(all_va[idxs, 0], all_va[idxs, 1], 
                   label=f"{mood_names[i]} ({len(idxs)})", 
                   alpha=0.6, s=30, color=colors4[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=150, label='Cluster Centers', edgecolors='white', linewidth=2)

plt.title("K-Means Clustering - 4 Moods")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.output_dir, "kmeans_4moods_va.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ====================
# 3️⃣ Predicted Mood Based on Max Count (Happy as demo)
# ====================
# Use the most common emotion as demo instead of hardcoding
emotion_counts = np.bincount(all_emotions)
predicted_emotion = np.argmax(emotion_counts)
print(f"Using most common emotion as demo: {emotion_names[predicted_emotion] if predicted_emotion < len(emotion_names) else f'Emotion {predicted_emotion}'}")

emotion_idxs = np.where(all_emotions == predicted_emotion)[0]
if len(emotion_idxs) == 0:
    print("No samples found for predicted emotion, using all samples")
    emotion_idxs = np.arange(len(all_emotions))

vas_pred_class = all_va[emotion_idxs]
cluster_counts = np.bincount(cluster_labels[emotion_idxs], minlength=4)
selected_cluster = np.argmax(cluster_counts)

print(f"Predicted Emotion: {emotion_names[predicted_emotion] if predicted_emotion < len(emotion_names) else f'Emotion {predicted_emotion}'}")
print(f"Cluster distribution: {cluster_counts}")
print(f"Selected Cluster: {selected_cluster} ({mood_names[selected_cluster]})")

plt.figure(figsize=(10, 8))
# Plot all points with low alpha
for i in range(4):
    idxs = np.where(cluster_labels == i)[0]
    if len(idxs) > 0:
        plt.scatter(all_va[idxs, 0], all_va[idxs, 1], 
                   alpha=0.3, s=20, color=colors4[i])

# Highlight predicted emotion points
plt.scatter(vas_pred_class[:, 0], vas_pred_class[:, 1], 
           c='black', s=50, label=f'Predicted Emotion Points ({len(vas_pred_class)})', 
           alpha=0.8, edgecolors='white', linewidth=1)

# Highlight selected cluster center
plt.scatter(kmeans.cluster_centers_[selected_cluster, 0],
            kmeans.cluster_centers_[selected_cluster, 1],
            c='red', marker='X', s=200, 
            label=f"Selected Cluster: {mood_names[selected_cluster]}", 
            edgecolors='white', linewidth=2)

plt.title(f"Predicted Mood in VA Space\n({emotion_names[predicted_emotion] if predicted_emotion < len(emotion_names) else f'Emotion {predicted_emotion}'} → {mood_names[selected_cluster]})")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.output_dir, "predicted_mood_va.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"All plots saved to: {Config.output_dir}")
print("Analysis complete!")