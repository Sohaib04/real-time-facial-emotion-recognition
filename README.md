# 🎭 Real-Time Emotion Detection

A production-quality desktop application that detects facial emotions in real time using a webcam, a custom CNN trained on FER-2013, and OpenCV for face detection and rendering.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time inference** | Webcam-based face detection + emotion classification at ≥15 FPS |
| **7 emotions** | Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral |
| **Colored bounding boxes** | Each emotion has a unique color for instant visual feedback |
| **Confidence scores** | Displays prediction confidence as a percentage |
| **Emotion smoothing** | Moving-average over the last N predictions to reduce jitter |
| **Session logging** | All predictions stored in SQLite with timestamp + session ID |
| **Session reports** | JSON summary on exit: dominant emotion, frequencies, avg confidence |
| **FPS counter** | Real-time frames-per-second display |
| **Session timer** | Elapsed time shown in the video overlay |
| **Screenshots** | Press `S` to save the current frame |
| **CLI configuration** | Override all settings via command-line arguments |
| **Docker support** | Containerized training environment |

---

## 🏗️ Architecture

```
emotion-recognition/
│
├── model/
│   ├── model.py           # EmotionCNN architecture (3-block Conv2D)
│   ├── train.py           # Full training pipeline with augmentation
│   ├── inference.py        # Inference engine with preprocessing
│   └── weights/            # Saved model weights + training curves
│
├── app/
│   ├── main.py             # Application entry point
│   ├── video_capture.py    # Webcam manager + Haar Cascade face detection
│   ├── logger.py           # SQLite session logger
│   ├── display.py          # OpenCV overlay renderer
│   └── emotion_smoother.py # Prediction smoothing (sliding window)
│
├── data/                   # FER-2013 dataset (not tracked)
├── logs/                   # SQLite DB + JSON session reports
├── screenshots/            # Saved screenshots
├── tests/                  # Unit tests
├── config.py               # Central configuration + CLI parser
├── requirements.txt
├── Dockerfile
└── README.md
```

### Data Flow

```
Webcam Frame
    │
    ▼
┌─────────────────┐
│  Face Detection  │ ← Haar Cascade (OpenCV)
│  (video_capture) │
└────────┬────────┘
         │ face ROI
         ▼
┌─────────────────┐
│  Preprocessing   │ ← Grayscale → 48×48 → Normalize [0,1]
│   (inference)    │
└────────┬────────┘
         │ tensor (1,1,48,48)
         ▼
┌─────────────────┐
│   EmotionCNN     │ ← 3 Conv blocks + FC classifier
│    (model)       │
└────────┬────────┘
         │ 7-class probabilities
         ▼
┌─────────────────┐
│    Smoothing     │ ← Sliding window majority vote
│ (emotion_smoother)│
└────────┬────────┘
         │ smoothed (emotion, confidence)
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Display Overlay │     │  SQLite Logger   │
│   (display)      │     │   (logger)       │
└─────────────────┘     └─────────────────┘
```

---

## 🧠 Model Details

### Architecture: EmotionCNN

| Layer | Details | Output Shape |
|---|---|---|
| Conv Block 1 | Conv2d(1→32, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25) | 32 × 24 × 24 |
| Conv Block 2 | Conv2d(32→64, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25) | 64 × 12 × 12 |
| Conv Block 3 | Conv2d(64→128, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25) | 128 × 6 × 6 |
| Flatten | — | 4608 |
| Dense | Linear(4608→128) → ReLU → Dropout(0.5) | 128 |
| Output | Linear(128→7) | 7 |

- **Parameters**: ~620K (≈2.4 MB)
- **Input**: 1 × 48 × 48 grayscale
- **Output**: 7-class probability vector
- **Weight Init**: Kaiming Normal (optimized for ReLU)

### Training

- **Dataset**: FER-2013 (35,887 images, 7 classes)
- **Augmentation**: Random horizontal flip, ±10° rotation, random crop with padding
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: patience=7 epochs
- **Loss**: CrossEntropyLoss

### Expected Performance

| Metric | Value |
|---|---|
| FER-2013 Test Accuracy | ~63-66% |
| Inference Time (CPU) | ~15-25ms/frame |
| FPS (with frame skip=2) | ≥15 FPS |
| Model Size | ~2.4 MB |

### Why Custom CNN vs Transfer Learning?

| Model | Accuracy | CPU Speed | Size | Decision |
|---|---|---|---|---|
| **Custom CNN** | ~63-66% | **15-25ms** | **~2MB** | ✅ Chosen |
| MobileNetV2 | ~68-72% | 60-100ms | ~14MB | Too slow on CPU |
| EfficientNetB0 | ~70-74% | 80-150ms | ~20MB | Too slow on CPU |

The custom CNN was chosen to meet the **≥15 FPS target on CPU**. The slight accuracy trade-off is compensated by **emotion smoothing** which reduces noisy predictions.

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Webcam
- (Optional) CUDA-compatible GPU

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Training

### 1. Download FER-2013

Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the `data/` directory:

```
data/
└── fer2013.csv
```

### 2. Train the Model

```bash
python model/train.py
```

Training will:
- Load and augment the FER-2013 dataset
- Train for up to 50 epochs with early stopping
- Save the best model to `model/weights/emotion_cnn.pth`
- Generate training curves and confusion matrix in `model/weights/`

### 3. Training with GPU

```bash
python model/train.py  # Add --use-gpu flag when running the app
```

---

## 🎥 Running the Application

### Basic Usage

```bash
python app/main.py
```

### With Options

```bash
# Use a different camera
python app/main.py --camera-index 1

# Higher confidence threshold
python app/main.py --confidence-threshold 0.6

# Faster (skip more frames)
python app/main.py --frame-skip 3

# GPU inference
python app/main.py --use-gpu

# Wider smoothing window
python app/main.py --smoothing-window 10

# All options
python app/main.py --camera-index 0 --confidence-threshold 0.5 --frame-skip 2 --smoothing-window 5
```

### Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Save current frame as screenshot |

### Session End

When you press `Q`, the application will:
1. Print a session summary in the terminal
2. Save a JSON session report in `logs/`
3. Release the camera and close the window

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_labels.py -v
python -m pytest tests/test_logger.py -v

# Generate test face image
python tests/generate_test_image.py
```

---

## ⚡ Performance Optimizations

| Optimization | Impact |
|---|---|
| **Frame skipping** | Runs inference every N frames (configurable), reuses last prediction for display |
| **`model.eval()`** | Disables dropout and batch norm updates during inference |
| **`torch.no_grad()`** | Disables gradient computation, reduces memory and speeds up forward pass |
| **Haar Cascade** | Fast face detection without a neural network overhead |
| **Histogram equalization** | Improves face detection under varying lighting without extra compute |
| **Context manager** | Guarantees camera release on exit, prevents resource leaks |
| **Rolling FPS** | Uses deque for O(1) FPS tracking without accumulating data |
| **Frame resize** | Camera set to 640×480 to reduce face detection processing |

---

## 🔮 How to Improve Accuracy

1. **Use a larger dataset**: Combine FER-2013 with AffectNet or RAF-DB
2. **Transfer learning**: Fine-tune MobileNetV2 or EfficientNetB0 (trade FPS for accuracy)
3. **Better face detection**: Replace Haar Cascade with MTCNN or RetinaFace
4. **Face alignment**: Align faces using facial landmarks before classification
5. **Ensemble**: Average predictions from multiple models
6. **Label smoothing**: Reduce overconfidence on noisy labels
7. **Test-time augmentation**: Average predictions over flipped/rotated versions
8. **Larger input size**: Use 64×64 or 96×96 (requires architecture changes)

---

## 🐳 Docker

```bash
# Build
docker build -t emotion-detector .

# Training
docker run -v $(pwd)/data:/app/data -v $(pwd)/model/weights:/app/model/weights emotion-detector

# Inference (requires display forwarding)
docker run -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/video0 \
  emotion-detector python app/main.py
```

---

## 📌 Resume Bullet Points

- Developed a **real-time facial emotion detection** desktop application using **PyTorch** and **OpenCV**, achieving **≥15 FPS** on CPU
- Designed and trained a **custom 3-block CNN** on the **FER-2013** dataset (35K+ images, 7 emotion classes) with **data augmentation**, **learning-rate scheduling**, and **early stopping**
- Built a **modular inference pipeline** with configurable frame skipping, prediction smoothing (sliding-window majority vote), and confidence thresholding
- Implemented a **SQLite session logging system** with per-prediction timestamps, automatic session management, and **JSON summary reports**
- Created a polished **OpenCV display** with per-emotion colored bounding boxes, real-time FPS counter, session timer, and screenshot functionality
- Applied **production-grade optimizations**: `torch.no_grad()`, `model.eval()`, context-managed camera lifecycle, histogram equalization, and configurable CLI arguments

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
