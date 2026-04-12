"""
======================================================
  Drowsiness Detection — CNN Training Script
  Trains two binary classifiers:
    1. Eye State  : Open vs Closed
    2. Mouth State: yawn  vs no_yawn
======================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# ── TF / Keras ─────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ═══════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════
IMG_SIZE   = 64          # resize all images to 64×64
BATCH_SIZE = 32
EPOCHS     = 30
TRAIN_DIR  = "train"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════
#  HELPER: load images from a list of (class, folder) pairs
# ═══════════════════════════════════════════════════

def load_images(class_pairs):
    """
    class_pairs: list of (label_int, folder_path)
    Returns X (N, IMG_SIZE, IMG_SIZE, 1), y (N,)
    """
    X, y = [], []
    for label, folder in class_pairs:
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Loading {len(files):4d} images from '{os.path.basename(folder)}' → label={label}")
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
    X = np.array(X, dtype='float32') / 255.0
    X = X[..., np.newaxis]  # add channel dim
    y = np.array(y, dtype='int32')
    return X, y


# ═══════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=2):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Head
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),   # binary → sigmoid
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ═══════════════════════════════════════════════════
#  TRAIN FUNCTION
# ═══════════════════════════════════════════════════

def train_classifier(name, class_pairs, model_path):
    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    # ── Load data ─────────────────────────────────
    X, y = load_images(class_pairs)
    print(f"\n  Total samples : {len(X)}")
    print(f"  Class balance : {np.bincount(y)}")

    # ── Train / Val split ─────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train)} | Val : {len(X_val)}")

    # ── Data augmentation ─────────────────────────
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.10,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    # ── Build model ───────────────────────────────
    model = build_cnn()
    model.summary()

    # ── Callbacks ─────────────────────────────────
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # ── Fit ───────────────────────────────────────
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  ✅ Final Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # ── Reports ───────────────────────────────────
    y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    class_names = [p[2] if len(p) > 2 else str(p[0]) for p in class_pairs]
    print("\n" + classification_report(y_val, y_pred, target_names=class_names))

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'],    label='Train Acc')
    axes[0].plot(history.history['val_accuracy'],label='Val Acc')
    axes[0].set_title(f'{name} — Accuracy'); axes[0].legend()
    axes[1].plot(history.history['loss'],    label='Train Loss')
    axes[1].plot(history.history['val_loss'],label='Val Loss')
    axes[1].set_title(f'{name} — Loss'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name.replace(" ","_")}_training.png'), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name.replace(" ","_")}_cm.png'), dpi=150)
    plt.close()

    print(f"\n  📦 Model saved → {model_path}")
    print(f"  📊 Plots saved → {RESULTS_DIR}/")
    return model, history


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Eye State Classifier ───────────────────
    #     0 = Open  |  1 = Closed
    eye_pairs = [
        (0, os.path.join(TRAIN_DIR, "Open"),   "Open"),
        (1, os.path.join(TRAIN_DIR, "Closed"), "Closed"),
    ]
    train_classifier(
        name       = "Eye State",
        class_pairs= eye_pairs,
        model_path = os.path.join(MODELS_DIR, "eye_state_model.h5"),
    )

    # ── 2. Yawn Classifier ────────────────────────
    #     0 = no_yawn  |  1 = yawn
    yawn_pairs = [
        (0, os.path.join(TRAIN_DIR, "no_yawn"), "no_yawn"),
        (1, os.path.join(TRAIN_DIR, "yawn"),    "yawn"),
    ]
    train_classifier(
        name       = "Yawn Detection",
        class_pairs= yawn_pairs,
        model_path = os.path.join(MODELS_DIR, "yawn_model.h5"),
    )

    print("\n\n🎉 Training complete! All models saved to ./models/")
    print("   Run 'python app.py' to launch the Streamlit interface.")
