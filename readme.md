# DrowsyGuard — Real-Time Drowsiness Detection System

A real-time driver drowsiness detection system that uses facial landmark analysis and convolutional neural networks to monitor eye closure and yawning behavior via webcam. Built with OpenCV, Dlib, TensorFlow/Keras, and Streamlit.

---

## Overview

DrowsyGuard monitors a driver's facial state in real time and raises an alert when signs of drowsiness are detected. The system relies on two complementary detection strategies:

- **Eye Aspect Ratio (EAR):** A geometric metric computed from 68 facial landmarks that tracks how open or closed the eyes are across consecutive frames. If the EAR remains below a configurable threshold for a sustained period (default: 3 seconds), a drowsiness alert is triggered.
- **CNN-based classification (optional):** Two lightweight convolutional neural networks — one for eye state (open/closed) and one for yawn detection — provide a second layer of confidence on top of the EAR signal. These models can be enabled or disabled from the UI at runtime.

The system distinguishes between normal blinks (~0.2 s) and actual eye closure by requiring that the closed state persist beyond a configurable duration before raising an alert.

---

## Features

- Real-time webcam feed with 68-point facial landmark overlay
- Eye Aspect Ratio (EAR) computation with temporal smoothing to filter blinks
- Optional CNN inference for eye state and yawn detection (enable in sidebar)
- Live EAR chart and alert log in the dashboard
- Configurable sensitivity, alert delay, and display options via sidebar
- Dark / light UI mode toggle
- Audio alert on macOS (`afplay`) when drowsiness or yawning is detected
- Alert history with timestamps

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Computer vision | OpenCV, imutils |
| Facial landmark detection | Dlib (68-point predictor) |
| Deep learning | TensorFlow / Keras |
| Data science | NumPy, SciPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard UI | Streamlit |

---

## Project Structure

```
Drowsiness_Detection/
├── app.py                          # Streamlit dashboard (main entry point)
├── Drowsiness_Detection.py         # Standalone OpenCV script (no UI)
├── train_model.py                  # CNN training script
├── kaggle_train_code.py            # Kaggle-compatible training variant
├── requirements.txt
├── models/
│   ├── shape_predictor_68_face_landmarks.dat   # Dlib landmark model
│   ├── eye_state_model.h5                      # Trained eye CNN
│   └── yawn_model.h5                           # Trained yawn CNN
├── train/
│   ├── Open/                       # Open eye images
│   ├── Closed/                     # Closed eye images
│   ├── Open_Eyes/                  # Alternate open eye set
│   ├── Closed_Eyes/                # Alternate closed eye set
│   ├── yawn/                       # Yawn mouth images
│   └── no_yawn/                    # Non-yawn mouth images
├── results/
│   ├── Eye_State_training.png      # Training curves (eye model)
│   ├── Eye_State_cm.png            # Confusion matrix (eye model)
│   ├── Yawn_Detection_training.png # Training curves (yawn model)
│   └── Yawn_Detection_cm.png       # Confusion matrix (yawn model)
└── .streamlit/
    └── config.toml                 # Streamlit theme configuration
```

---

## Installation

**1. Clone the repository and navigate into it:**

```bash
git clone https://github.com/<your-username>/Drowsiness_Detection.git
cd Drowsiness_Detection
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` requires CMake and a C++ compiler. On macOS, install Xcode Command Line Tools (`xcode-select --install`) and CMake (`brew install cmake`) before running `pip install`.

---

## Running the Application

**Streamlit dashboard (recommended):**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Use the sidebar to adjust detection parameters and toggle CNN inference. Press **Start** to begin monitoring.

**Standalone script (no UI):**

```bash
python Drowsiness_Detection.py
```

Opens a raw OpenCV window. Press `q` to quit.

---

## Training the CNN Models

The training script builds and trains two binary classifiers using data augmentation and early stopping.

**Dataset layout expected under `train/`:**

| Class | Folder |
|---|---|
| Open eye | `train/Open` |
| Closed eye | `train/Closed` |
| Yawning | `train/yawn` |
| Not yawning | `train/no_yawn` |

**Run training:**

```bash
python train_model.py
```

The script will:

1. Load and normalize all images to 64x64 grayscale
2. Split data into 80% train / 20% validation (stratified)
3. Apply data augmentation (rotation, shifts, zoom, horizontal flip, brightness variation)
4. Train a 3-block CNN with BatchNormalization and Dropout using early stopping and learning rate reduction
5. Save the best checkpoints to `models/eye_state_model.h5` and `models/yawn_model.h5`
6. Generate training curves and confusion matrix plots in `results/`

**CNN architecture summary:**

```
Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.25)
GlobalAveragePooling → Dense(128) → BN → Dropout(0.5)
Dense(1, sigmoid)
```

Training configuration: `IMG_SIZE=64`, `BATCH_SIZE=32`, `EPOCHS=30`, `Adam(lr=1e-3)`, binary cross-entropy loss.

---

## How It Works

```
Webcam frame
     |
     v
Dlib face detector  -->  68 facial landmarks
     |
     +---> Left eye + Right eye landmarks
     |          |
     |          v
     |      Eye Aspect Ratio (EAR) = (A + B) / (2 * C)
     |          |                    (vertical / horizontal)
     |          v
     |      Temporal smoothing (5-frame rolling average)
     |          |
     |          v
     |      EAR < threshold for >= N seconds?  -->  DROWSINESS ALERT
     |
     +---> Mouth landmarks
               |
               v  (if CNN enabled)
           Yawn CNN  -->  P(yawn) >= 0.75 for >= T seconds?  -->  YAWN ALERT
```

The EAR formula is defined as:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

where p1–p6 are the six landmark points around each eye. A fully open eye has EAR ~ 0.3; a closed eye has EAR ~ 0.0.

---

## Configuration

All detection parameters are adjustable from the Streamlit sidebar at runtime:

| Parameter | Default | Description |
|---|---|---|
| Eye sensitivity (EAR threshold) | 0.22 | Lower values require more eye closure to trigger |
| Alert after closed eyes | 3.0 s | Minimum sustained closure to raise a drowsiness alert |
| Yawn alert after | 2.0 s | Minimum yawn duration to raise a yawn alert |
| CNN inference | Off | Enable CNN models on top of EAR (requires trained models) |
| Camera width | 450 px | Frame resolution sent to the detector |

---

## Requirements

See [`requirements.txt`](requirements.txt) for the full pinned list. Main dependencies:

- Python >= 3.9
- opencv-python >= 4.8.0
- dlib >= 19.24.0
- tensorflow >= 2.13.0
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

---

## Notes

- The Dlib landmark model (`shape_predictor_68_face_landmarks.dat`) must be present in the `models/` directory before running the app. It is a large binary file managed via Git LFS or downloaded separately.
- The macOS audio alert uses `afplay` and runs as a background process (`&`). On Linux or Windows, this call is a no-op; replace it with an appropriate audio library (e.g., `playsound`, `pygame`) if needed.
- For best detection accuracy, ensure adequate frontal lighting and that the face is clearly visible to the webcam.
