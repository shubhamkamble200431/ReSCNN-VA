# ReSCNN-VA: Lightweight Sensor-Embedded CNN for Real-Time Valence-Arousal Regression and Emotion-Driven Music Recommendation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-3-red)](https://www.raspberrypi.org)

> **A lightweight CNN-based inference pipeline for real-time facial emotion recognition and emotion-aware music recommendation, optimized for embedded sensor deployment.**

![System Pipeline](images/ECNN_VA.drawio.png)
*Proposed ReSCNN-VA system pipeline for emotion-aware music recommendation*

## 🚀 Overview

ReSCNN-VA presents a comprehensive framework that combines facial emotion recognition (FER) with continuous valence-arousal (VA) regression to enable real-time, personalized music recommendations. The system is specifically designed for deployment on resource-constrained embedded platforms while maintaining robust performance across various sensing conditions.

### Key Features

- **🎯 Dual Recognition**: Discrete emotion classification (7 categories) + continuous valence-arousal regression
- **⚡ Lightweight Architecture**: Only 2.6M parameters for emotion recognition, 1.8M for VA regression
- **🔧 Edge Deployment**: Optimized for Raspberry Pi 3 with real-time inference capabilities
- **🎵 Music Recommendation**: Emotion-aware personalized music selection with user feedback
- **🏃‍♂️ Real-time Processing**: Low-latency inference suitable for interactive applications
- **🔋 Power Efficient**: 5W total system power consumption during inference

## 📊 Performance Metrics

| Platform | Precision | Accuracy | Model Size | Memory Usage | Power |
|----------|-----------|----------|------------|--------------|--------|
| CPU FP32 | FP32 | 69.69% | 27.35MB | ~1GB | ~15W |
| GPU FP16 | FP16 | 69.70% | 13.67MB | ~512MB | ~25W |
| RPi3 | FP32 | 69.69% | 27.35MB | ~800MB | ~5W |

## 🏗️ System Architecture

![System Architecture](images/system_architecture.png)
*Comprehensive system architecture from facial input to music recommendation*

The framework consists of three main components:

1. **EmotionCNN**: Lightweight CNN for discrete emotion classification (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
2. **VA Regressor**: MobileNetV3-Small based model for continuous valence-arousal mapping
3. **Music Recommender**: Emotion-aware recommendation system with collaborative filtering

## 📁 Repository Structure

```
ReSCNN-VA/
├── .dist/                          # Distribution files
├── cpu_outputs/                    # CPU inference results and visualizations
│   ├── cpu_inference_fp32*.json   # Performance metrics
│   └── cpu_inference_fp32*.png    # Confusion matrices
├── gpu_outputs/                    # GPU inference results
├── images/                         # Documentation images and diagrams
├── preprocessing/                  # Data preprocessing utilities
├── rpi_outputs/                    # Raspberry Pi deployment results
├── RS/                            # Main recommendation system
│   ├── cpu_inference.py          # CPU inference pipeline
│   ├── gpu_inference.py          # GPU inference pipeline
│   ├── rpi_inference.py          # Raspberry Pi deployment
│   ├── config.py                 # Configuration settings
│   ├── recommend.py              # Music recommendation engine
│   └── convert_to_torchscript.py # Model conversion utilities
├── train_ECNN/                   # Emotion CNN training
│   └── scripts/
│       └── train_fp32_fp16.py    # Training script with mixed precision
├── VA_analysis/                  # Valence-Arousal analysis
│   └── scripts/
│       └── train_va.py          # VA regression training
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9+
- OpenCV 4.5+
- NumPy, Pandas, Scikit-learn
- Spotify API credentials (for music features)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ReSCNN-VA.git
cd ReSCNN-VA

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Raspberry Pi Setup

```bash
# Install PyTorch for ARM
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Convert model to TorchScript
python RS/convert_to_torchscript.py

# Run inference on RPi
python RS/rpi_inference.py
```

## 🚀 Quick Start

### Basic Emotion Recognition

```python
from RS.cpu_inference import EmotionInference

# Initialize inference pipeline
inferencer = EmotionInference(model_path="models/emotion_cnn.pth")

# Recognize emotion from image
emotion, va_coords = inferencer.predict("path/to/image.jpg")
print(f"Emotion: {emotion}, Valence: {va_coords[0]:.2f}, Arousal: {va_coords[1]:.2f}")
```

### Music Recommendation

```python
from RS.recommend import MusicRecommender

# Initialize recommender
recommender = MusicRecommender()

# Get emotion-based recommendations
songs = recommender.recommend_by_emotion(
    emotion="happy", 
    valence=0.8, 
    arousal=0.6, 
    user_id="user123"
)

for song in songs:
    print(f"{song['name']} by {song['artist']} - Match: {song['similarity']:.2f}")
```

### Real-time Webcam Demo

```python
# Run real-time emotion recognition with webcam
python RS/realtime_demo.py --source webcam --recommend
```

## 📚 Training

### Emotion CNN Training

```bash
cd train_ECNN/scripts/
python train_fp32_fp16.py --dataset fer2013 --epochs 100 --batch_size 64
```

### VA Regression Training

```bash
cd VA_analysis/scripts/
python train_va.py --dataset affectnet --epochs 50 --lr 0.001
```

## 🎯 Datasets

- **FER2013**: 35,887 samples for emotion classification
- **AffectNet**: 287,651 samples for valence-arousal regression
- **Custom Music Dataset**: Audio features from Spotify API

### Data Preprocessing

The framework includes comprehensive preprocessing:
- Gaussian filtering for noise reduction
- Image standardization to 48×48 resolution
- Class balancing through minority upsampling
- Data augmentation (rotation, flipping, color jittering)

## 🔬 Experimental Results

### Robustness Analysis

![Robustness Results](cpu_outputs/cpu_inference_fp32_gaussian50.png)
*Performance under Gaussian noise (σ=50)*

The system demonstrates robust performance under various perturbations:
- **Gaussian Noise**: Minimal accuracy drop with σ up to 50
- **Impulse Noise**: Stable performance with noise ratios up to 0.05
- **Illumination Variation**: Consistent recognition across lighting conditions

### Cross-Platform Validation

| Test Condition | CPU Accuracy | GPU Accuracy | RPi3 Accuracy |
|----------------|--------------|--------------|---------------|
| Clean Images | 69.69% | 69.70% | 69.69% |
| Gaussian Noise (σ=25) | 67.2% | 67.3% | 67.1% |
| Impulse Noise (0.01) | 68.5% | 68.6% | 68.4% |
| Illumination (1.5x) | 66.8% | 67.0% | 66.7% |

## 🎵 Music Recommendation Features

### Mood Clustering

The system uses K-means clustering (k=4) to partition the VA space into distinct mood regions:
- **Happy** (High Valence, Moderate Arousal)
- **Energetic** (High Valence, High Arousal)  
- **Calm** (Moderate Valence, Low Arousal)
- **Sad** (Low Valence, Low Arousal)

### Recommendation Strategy

- **70% Mood-based**: Songs selected based on VA similarity
- **30% User History**: Collaborative filtering from user preferences
- **Adaptive Learning**: Continuous feedback integration
- **Multi-user Support**: Individual preference profiles

## 🤖 Edge Deployment

### Hardware Requirements

- **Minimum**: Raspberry Pi 3B+ (1GB RAM)
- **Recommended**: Raspberry Pi 4B (4GB RAM)
- **Camera**: Pi NoIR Camera Module or USB webcam
- **Storage**: 16GB microSD card

### Optimization Techniques

- **Model Quantization**: FP16 precision for GPU inference
- **TorchScript Conversion**: Optimized mobile execution
- **Memory Management**: Batch size optimization for constrained resources
- **Power Optimization**: Efficient inference scheduling

## 📈 Future Enhancements

- [ ] Temporal emotion modeling for video streams
- [ ] Hardware acceleration on Coral TPU/Jetson Nano
- [ ] Context-aware recommendations (time, location, activity)
- [ ] Large-scale user studies for demographic fairness
- [ ] Integration with additional sensor modalities

## 🤝 Applications

### Socially Assistive Technology

The framework is particularly designed for:
- **Healthcare Monitoring**: Real-time patient emotion tracking
- **Special Needs Support**: Adaptive interfaces for cognitive impairments
- **Educational Technology**: Engagement monitoring in learning environments
- **Driver Assistance**: Emotion-aware safety systems

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{kamble2025rescnn,
  title={ReSCNN-VA: A Lightweight Sensor-Embedded CNN for Real-Time Valence-Arousal Regression and Emotion-Driven Music Recommendation},
  author={Kamble, Shubham and Swati and Engineer, Pinalkumar},
  journal={Sensor Applications},
  volume={1},
  number={3},
  pages={0000000},
  year={2025},
  publisher={IEEE}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- **Shubham Kamble** - Department of Electronics Engineering, SVNIT Surat
- **Swati** - Student Member, IEEE
- **Pinalkumar Engineer** - Senior Member, IEEE

## 📞 Contact

For questions and collaborations, please reach out:
- 📧 Email: [contact-email]
- 🌐 Website: [project-website]
- 📚 Paper: [paper-link]

## 🙏 Acknowledgments

Special thanks to the FER2013 and AffectNet dataset creators, and the open-source computer vision community for their valuable contributions.

---

