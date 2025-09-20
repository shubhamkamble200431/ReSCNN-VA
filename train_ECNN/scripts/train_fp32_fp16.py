"""
emotion_pipeline_enhanced.py

Enhanced pipeline with:
 - FP32, FP16 evaluation
 - Class imbalance handling with upsampling
 - Better plots and visualizations
 - Progress bars with tqdm
 - Best model saving
"""

import os
import time
import random
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
    # Import tqdm with fallback
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
class Config:
    train_dir = r"/home/ml/Desktop/shubham/sensors/dataset/train"
    test_dir = r"/home/ml/Desktop/shubham/sensors/dataset/test"
    outputs_dir = "/home/ml/Desktop/shubham/sensors/new_approach_cnn/gpt_2.0/outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    batch_size = 128
    learning_rate = 0.001
    num_epochs = 100
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    seed = 42

    train_from_scratch = True
    fp32_model_path = os.path.join(outputs_dir, "best_emotion_model_fp32.pth")
    fp16_model_path = os.path.join(outputs_dir, "best_emotion_model_fp16.pth")
    
    # Class balancing
    balance_classes = True
    early_stopping_patience = 15
    
    unseen_image_path = "/home/ml/Desktop/shubham/sensors/test_images/happy.jpg"

# Set plot style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style
sns.set_palette("husl")

# ---------------- SEED ----------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(Config.seed)

# ---------------- DATASET ----------------
class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None, balance_classes=False):
        self.root_dir = root_dir
        self.transform = transform
        self.balance_classes = balance_classes
        self.classes = Config.class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples: List[Tuple[str,int]] = []
        self.class_counts = Counter()

        if not os.path.exists(root_dir):
            print(f"ERROR: Directory {root_dir} does not exist!")
            return

        # Load original samples
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
                        self.class_counts[class_name] += 1

        print("\nOriginal class distribution:")
        for cls, count in self.class_counts.items():
            print(f"  {cls}: {count}")

        # Balance classes if requested
        if balance_classes and self.class_counts:
            self._balance_classes()

    def _balance_classes(self):
        """Balance classes by upsampling to max class count"""
        max_count = max(self.class_counts.values())
        print(f"\nBalancing classes to {max_count} samples each...")
        
        balanced_samples = []
        
        for class_name in self.classes:
            class_samples = [s for s in self.samples if s[1] == self.class_to_idx[class_name]]
            current_count = len(class_samples)
            
            if current_count == 0:
                continue
                
            # Add original samples
            balanced_samples.extend(class_samples)
            
            # Upsample by repeating existing samples
            needed = max_count - current_count
            if needed > 0:
                upsampled = random.choices(class_samples, k=needed)
                balanced_samples.extend(upsampled)
                print(f"  {class_name}: {current_count} -> {max_count} (+{needed})")
        
        self.samples = balanced_samples
        
        # Update counts
        self.class_counts = Counter()
        for _, label in self.samples:
            class_name = self.classes[label]
            self.class_counts[class_name] += 1
            
        print("\nBalanced class distribution:")
        for cls, count in self.class_counts.items():
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

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

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- EVALUATION HELPERS ----------------
def evaluate_and_confmat(model, loader, cm_path, device, model_name="Model", use_amp=False, half=False):
    """Evaluate model and generate confusion matrix with enhanced visualization"""
    model.eval()
    preds, labs = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    model = model.to(device)
    if half and device.type == "cuda":
        model = model.half()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {model_name}", ncols=100)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            if half and device.type == "cuda":
                imgs = imgs.half()
                
            if use_amp and device.type == "cuda":
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                except:
                    # Fallback if AMP is not available
                    outputs = model(imgs)
            else:
                outputs = model(imgs)
                
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds.extend(outputs.argmax(1).cpu().tolist())
            labs.extend(labels.cpu().tolist())
            
            # Update progress bar
            current_acc = 100.0 * (np.array(preds) == np.array(labs)).mean()
            pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    accuracy = 100.0 * (np.array(preds) == np.array(labs)).mean()
    avg_loss = total_loss / len(loader)
    
    # Generate enhanced confusion matrix
    cm = confusion_matrix(labs, preds)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Performance Analysis', fontsize=16, fontweight='bold')
    
    # Confusion Matrix (Raw counts)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=Config.class_names, 
                yticklabels=Config.class_names, ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix (Raw Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Confusion Matrix (Normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=Config.class_names,
                yticklabels=Config.class_names, ax=ax2, cmap='Reds')
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    ax3.bar(Config.class_names, per_class_acc)
    ax3.set_title('Per-Class Accuracy')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    for i, acc in enumerate(per_class_acc):
        ax3.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    ax3.tick_params(axis='x', rotation=45)
    
    # Class distribution in predictions vs actual
    actual_counts = np.bincount(labs, minlength=len(Config.class_names))
    pred_counts = np.bincount(preds, minlength=len(Config.class_names))
    
    x = np.arange(len(Config.class_names))
    width = 0.35
    ax4.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
    ax4.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    ax4.set_title('Class Distribution: Actual vs Predicted')
    ax4.set_xlabel('Classes')
    ax4.set_ylabel('Count')
    ax4.set_xticks(x)
    ax4.set_xticklabels(Config.class_names, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(labs, preds, target_names=Config.class_names))
    
    return accuracy, avg_loss, preds, labs

def measure_latency_and_throughput(model, loader, device, model_name="Model", repeat_batches=10, half=False, warmup=3):
    """Measure latency and throughput with detailed statistics"""
    model = model.to(device).eval()
    if half and device.type == "cuda":
        model = model.half()
    
    batch_times = []
    sample_times = []
    
    with torch.no_grad():
        pbar = tqdm(range(repeat_batches), desc=f"Measuring {model_name} Performance", ncols=100)
        
        for batch_idx in pbar:
            if batch_idx >= len(loader):
                break
                
            imgs, _ = next(iter(loader))
            imgs = imgs.to(device)
            if half and device.type == "cuda":
                imgs = imgs.half()
            
            # Warmup
            for _ in range(warmup):
                _ = model(imgs)
            
            # Synchronize before timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            outputs = model(imgs)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            sample_time = batch_time / imgs.size(0)
            
            batch_times.append(batch_time)
            sample_times.append(sample_time)
            
            # Update progress bar
            avg_batch_time = np.mean(batch_times)
            fps = imgs.size(0) / avg_batch_time
            pbar.set_postfix({
                'Batch Time': f'{avg_batch_time*1000:.2f}ms',
                'FPS': f'{fps:.1f}'
            })
    
    # Calculate statistics
    stats = {
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'avg_sample_time': np.mean(sample_times),
        'fps': imgs.size(0) / np.mean(batch_times),
        'throughput': len(loader.dataset) / (np.mean(batch_times) * len(loader))
    }
    
    return stats

def plot_class_distribution(train_dataset, test_dataset):
    """Plot class distribution comparison"""
    train_counts = train_dataset.class_counts
    test_counts = Counter()
    
    # Count test dataset classes
    for _, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        test_counts[class_name] += 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training set distribution
    classes = list(train_counts.keys())
    train_values = [train_counts[cls] for cls in classes]
    test_values = [test_counts[cls] for cls in classes]
    
    ax1.bar(classes, train_values, alpha=0.8, color='skyblue')
    ax1.set_title('Training Set Class Distribution')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(train_values):
        ax1.text(i, v + max(train_values)*0.01, str(v), ha='center')
    
    ax2.bar(classes, test_values, alpha=0.8, color='lightcoral')
    ax2.set_title('Test Set Class Distribution')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(test_values):
        ax2.text(i, v + max(test_values)*0.01, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.outputs_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def train_baseline_model(model, train_loader, val_loader, device):
    """Train baseline model with early stopping and best model saving"""
    print("\n" + "="*50)
    print("Training Baseline FP32 Model")
    print("="*50)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}", ncols=120)
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, LR: {current_lr:.2e}")
        
        # Early stopping and best model saving
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss
            }, Config.fp32_model_path)
            print(f"✓ New best model saved! Val Acc: {epoch_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= Config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training history
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History - Baseline FP32 Model', fontsize=16)
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss plots
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plots
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate history
    lr_history = []
    # Get learning rate history from scheduler if available
    try:
        # For ReduceLROnPlateau, we need to track manually
        initial_lr = optimizer.param_groups[0]['lr']
        lr_history = [initial_lr] * len(train_losses)  # Simplified for display
    except:
        lr_history = [Config.learning_rate] * len(train_losses)
    
    if len(lr_history) > 1:
        ax3.plot(epochs_range, lr_history, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        ax3.set_yscale('log')
    else:
        ax3.axis('off')
    
    # Best metrics summary
    ax4.axis('off')
    summary_text = f"""
    Training Summary:
    
    Best Validation Accuracy: {best_val_acc:.2f}%
    Total Epochs: {len(train_losses)}
    Final Train Accuracy: {train_accuracies[-1]:.2f}%
    Final Validation Accuracy: {val_accuracies[-1]:.2f}%
    
    Early Stopping: {'Yes' if patience_counter >= Config.early_stopping_patience else 'No'}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.outputs_dir, "training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Load best model
    checkpoint = torch.load(Config.fp32_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model loaded with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model

def create_benchmark_summary(results_dict):
    """Create comprehensive benchmark summary with all metrics"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Create summary DataFrame
    summary_data = []
    for model_name, metrics in results_dict.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy (%)': f"{metrics['accuracy']:.2f}",
            'Avg Loss': f"{metrics.get('avg_loss', 0):.4f}",
            'Batch Time (ms)': f"{metrics.get('batch_time', 0)*1000:.2f}",
            'FPS': f"{metrics.get('fps', 0):.1f}",
            'Model Size': metrics.get('model_size', 'N/A')
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(Config.outputs_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Benchmark summary saved to: {csv_path}")
    
    # Print formatted table
    print("\n" + df.to_string(index=False))
    
    # Create visual comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = [item['Model'] for item in summary_data]
    accuracies = [float(item['Accuracy (%)']) for item in summary_data]
    fps_values = [float(item['FPS']) if item['FPS'] != 'N/A' else 0 for item in summary_data]
    batch_times = [float(item['Batch Time (ms)']) if item['Batch Time (ms)'] != 'N/A' else 0 for item in summary_data]
    
    # Accuracy comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]
    bars1 = ax1.bar(models, accuracies, color=colors)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # FPS comparison
    bars2 = ax2.bar(models, fps_values, color=colors)
    ax2.set_title('Inference Speed Comparison')
    ax2.set_ylabel('FPS (Frames Per Second)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, fps in zip(bars2, fps_values):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(fps_values)*0.01,
                    f'{fps:.1f}', ha='center', va='bottom')
    
    # Batch time comparison
    bars3 = ax3.bar(models, batch_times, color=colors)
    ax3.set_title('Batch Processing Time Comparison')
    ax3.set_ylabel('Batch Time (ms)')
    ax3.tick_params(axis='x', rotation=45)
    for bar, bt in zip(bars3, batch_times):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(batch_times)*0.01,
                    f'{bt:.2f}ms', ha='center', va='bottom')
    
    # Accuracy vs Speed scatter plot
    valid_fps = [fps for fps in fps_values if fps > 0]
    valid_acc = [acc for acc, fps in zip(accuracies, fps_values) if fps > 0]
    valid_models = [model for model, fps in zip(models, fps_values) if fps > 0]
    
    if valid_fps:
        scatter = ax4.scatter(valid_fps, valid_acc, s=100, alpha=0.7)
        ax4.set_xlabel('FPS (Frames Per Second)')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy vs Speed Trade-off')
        ax4.grid(True, alpha=0.3)
        
        for i, model in enumerate(valid_models):
            ax4.annotate(model, (valid_fps[i], valid_acc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.outputs_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return f"{size_mb:.2f} MB"

# ---------------- MAIN ----------------
def main():
    print("🚀 Starting Enhanced Emotion Recognition Pipeline")
    print("="*60)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = FER2013Dataset(Config.train_dir, transform=train_transform, 
                                  balance_classes=Config.balance_classes)
    val_dataset = FER2013Dataset(Config.test_dir, transform=eval_transform)
    
    # Plot class distributions
    plot_class_distribution(train_dataset, val_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                            shuffle=True, num_workers=Config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                          shuffle=False, num_workers=Config.num_workers)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Device: {Config.device}")
    
    # Initialize model
    model = EmotionCNN(num_classes=len(Config.class_names))
    
    # Dictionary to store all results
    results = {}
    
    # 1. CHECK FOR EXISTING FP32 MODEL AND ASK USER
    if os.path.exists(Config.fp32_model_path) and not Config.train_from_scratch:
        print(f"Found existing FP32 model at: {Config.fp32_model_path}")
        user_choice = input("Do you want to use this model? (y/n): ").lower().strip()
        if user_choice in ['y', 'yes']:
            print("Loading existing FP32 model...")
            checkpoint = torch.load(Config.fp32_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(Config.device)
            print(f"Loaded model with validation accuracy: {checkpoint.get('val_acc', 'Unknown'):.2f}%")
        else:
            print("Training new model from scratch...")
            model = train_baseline_model(model, train_loader, val_loader, Config.device)
    elif Config.train_from_scratch:
        model = train_baseline_model(model, train_loader, val_loader, Config.device)
    else:
        print(f"No existing model found at {Config.fp32_model_path}")
        print("Training new model from scratch...")
        model = train_baseline_model(model, train_loader, val_loader, Config.device)
    
    # 2. EVALUATE FP32 BASELINE
    print("\n" + "="*50)
    print("Evaluating FP32 Baseline Model")
    print("="*50)
    
    acc_fp32, loss_fp32, _, _ = evaluate_and_confmat(
        model, val_loader, 
        os.path.join(Config.outputs_dir, "confusion_matrix_fp32.png"),
        Config.device, "FP32 Baseline"
    )
    
    # Measure FP32 performance
    perf_fp32 = measure_latency_and_throughput(model, val_loader, Config.device, "FP32")
    
    results['FP32'] = {
        'accuracy': acc_fp32,
        'avg_loss': loss_fp32,
        'batch_time': perf_fp32['avg_batch_time'],
        'fps': perf_fp32['fps'],
        'model_size': get_model_size(model)
    }
    
    print(f"FP32 Results: Acc={acc_fp32:.2f}%, FPS={perf_fp32['fps']:.1f}")
    
    # 3. EVALUATE FP16 (GPU only)
    if Config.device.type == "cuda":
        print("\n" + "="*50)
        print("Evaluating FP16 Model")
        print("="*50)
        
        # Create FP16 model copy
        model_fp16 = EmotionCNN(num_classes=len(Config.class_names))
        model_fp16.load_state_dict(model.state_dict())
        
        acc_fp16, loss_fp16, _, _ = evaluate_and_confmat(
            model_fp16, val_loader,
            os.path.join(Config.outputs_dir, "confusion_matrix_fp16.png"),
            Config.device, "FP16", half=True
        )
        
        # Measure FP16 performance
        perf_fp16 = measure_latency_and_throughput(model_fp16, val_loader, Config.device, "FP16", half=True)
        
        # Save FP16 model
        model_fp16_cpu = model_fp16.to("cpu")
        torch.save({
            'model_state_dict': model_fp16_cpu.state_dict(),
            'val_acc': acc_fp16,
            'val_loss': loss_fp16,
            'model_type': 'FP16'
        }, Config.fp16_model_path)
        print(f"FP16 model saved to: {Config.fp16_model_path}")
        
        results['FP16'] = {
            'accuracy': acc_fp16,
            'avg_loss': loss_fp16,
            'batch_time': perf_fp16['avg_batch_time'],
            'fps': perf_fp16['fps'],
            'model_size': get_model_size(model_fp16_cpu)
        }
        
        print(f"FP16 Results: Acc={acc_fp16:.2f}%, FPS={perf_fp16['fps']:.1f}")
    else:
        print("\n⚠️  FP16 evaluation skipped (CUDA not available)")
    
    # 4. CREATE COMPREHENSIVE SUMMARY
    print("\n" + "="*50)
    print("Generating Final Report")
    print("="*50)
    
    # Create benchmark summary
    summary_df = create_benchmark_summary(results)
    
    # Print final summary
    print("\n🎯 FINAL RESULTS SUMMARY:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"{model_name:8} | Acc: {metrics['accuracy']:6.2f}% | "
              f"FPS: {metrics['fps']:6.1f} | Size: {metrics['model_size']}")
    
    print(f"\n✅ All outputs saved to: {Config.outputs_dir}")
    print("📊 Generated files:")
    print("  - confusion_matrix_*.png (Detailed confusion matrices)")
    print("  - training_history.png (Training progress)")  
    print("  - class_distribution.png (Dataset analysis)")
    print("  - performance_comparison.png (Model comparison)")
    print("  - benchmark_summary.csv (Detailed metrics)")
    print("  - best_emotion_model_fp32.pth (FP32 model checkpoint)")
    if Config.device.type == "cuda":
        print("  - best_emotion_model_fp16.pth (FP16 model checkpoint)")
    
    print("\n🚀 Pipeline completed successfully!")

if __name__ == "__main__":
    main()