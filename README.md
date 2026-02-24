# 🎭 Real-Time Emotion Detection

A production-quality desktop application that detects facial emotions in real time using a webcam, a custom CNN trained on FER-2013, and OpenCV for face detection and rendering.

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