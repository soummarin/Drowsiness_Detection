"""
======================================================
  KAGGLE: Drowsiness Detection — CNN Training Script
======================================================
Copie et colle tout ce code dans ton Notebook Kaggle !
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2
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
#  CONFIGURATION KAGGLE
# ═══════════════════════════════════════════════════
IMG_SIZE   = 64 
BATCH_SIZE = 64   # Plus grand car on a un GPU Kaggle puissant !
EPOCHS     = 40   # Plus d'epochs pour mieux apprendre

MODELS_DIR = "/kaggle/working/models"
RESULTS_DIR = "/kaggle/working/results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── RECHERCHE MAGIQUE MULTI-DATASETS KAGGLE ──
def find_all_folders(folder_name):
    # Cherche TOUS les dossiers qui ont ce nodnc om exact, n'importe où !
    paths = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for d in dirs:
            # On ignore les espaces pour être très robuste
            if d.lower().replace(" ", "") == folder_name.lower().replace(" ", ""):
                paths.append(os.path.join(root, d))
    return paths

print("\n🔍 Collecte massive de toutes tes images sur Kaggle...")
OPEN_DIRS   = find_all_folders("Open") + find_all_folders("Open_Eyes") + find_all_folders("NonDrowsy")
CLOSED_DIRS = find_all_folders("Closed") + find_all_folders("Closed_Eyes") + find_all_folders("Drowsy")
NO_YAWN_DIRS= find_all_folders("no_yawn")
YAWN_DIRS   = find_all_folders("yawn")

print(f"✔️ {len(OPEN_DIRS)} dossiers trouvés pour 'Open/NonDrowsy'")
print(f"✔️ {len(CLOSED_DIRS)} dossiers trouvés pour 'Closed/Drowsy'")
print(f"✔️ {len(YAWN_DIRS)} dossiers trouvés pour 'Yawn'\n")

# ═══════════════════════════════════════════════════
#  HELPER: Charger les images
# ═══════════════════════════════════════════════════

def load_images(class_pairs):
    X, y = [], []
    for p in class_pairs:
        label, folder = p[0], p[1]
        
        if not folder or not os.path.isdir(folder):
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > 0:
            print(f"  > Loading {len(files):5d} images from '{folder}' → label={label}")
        
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
            
    X = np.array(X, dtype='float32') / 255.0
    X = X[..., np.newaxis]
    y = np.array(y, dtype='int32')
    return X, y

# ═══════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    # Architecture légèrement plus puissante pour grand Dataset
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ═══════════════════════════════════════════════════
#  TRAIN FUNCTION
# ═══════════════════════════════════════════════════

def train_classifier(name, class_pairs, model_path):
    print(f"\n{'='*55}\n  Training: {name}\n{'='*55}")

    X, y = load_images(class_pairs)
    if len(X) == 0:
        print("❌ AUCUNE IMAGE TROUVÉE ! Ne lance pas la suite.")
        return None, None

    print(f"\n  Total samples : {len(X)}")
    counts = np.bincount(y)
    print(f"  Class balance : Class 0: {counts[0] if len(counts)>0 else 0} | Class 1: {counts[1] if len(counts)>1 else 0}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train samples : {len(X_train)} | Val samples : {len(X_val)}")

    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.1, horizontal_flip=True, brightness_range=[0.8, 1.2],
    )

    model = build_cnn()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  ✅ Final Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    y_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
    
    class_names = []
    for p in class_pairs:
        nom = p[2] if len(p) > 2 else str(p[0])
        if nom not in class_names:
            class_names.append(nom)

    print("\n" + classification_report(y_val, y_pred, target_names=class_names))

    # Confusion matrix & Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(f'{name} — Accuracy'); axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{name} — Loss'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name.replace(" ","_")}_training.png'), dpi=150)
    plt.close()

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name.replace(" ","_")}_cm.png'), dpi=150)
    plt.close()

    print(f"\n  📦 Model saved → {model_path}")
    return model, history


# ═══════════════════════════════════════════════════
#  MAIN KAGGLE
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    eye_pairs, yawn_pairs = [], []
    
    # On ajoute absolument tous les dossiers trouvés dans les 3 datasets !
    for folder in OPEN_DIRS:   eye_pairs.append((0, folder, "Open"))
    for folder in CLOSED_DIRS: eye_pairs.append((1, folder, "Closed"))
    
    for folder in NO_YAWN_DIRS: yawn_pairs.append((0, folder, "no_yawn"))
    for folder in YAWN_DIRS:    yawn_pairs.append((1, folder, "yawn"))

    # ── 1. Eye State Classifier ───────────────────
    if len(eye_pairs) >= 2:
        train_classifier("Eye State", eye_pairs, os.path.join(MODELS_DIR, "eye_state_model.h5"))
    else:
        print("⏭️ Impossible d'entraîner Eye State (Dossiers Open/Closed manquants)")

    # ── 2. Yawn Classifier ────────────────────────
    if len(yawn_pairs) >= 2:
        train_classifier("Yawn Detection", yawn_pairs, os.path.join(MODELS_DIR, "yawn_model.h5"))
    else:
        print("⏭️ Impossible d'entraîner Yawn (Dossiers yawn/no_yawn manquants)")

    print("\n🎉 ENTRAÎNEMENT KAGGLE TERMINÉ ! 🎉")
    print("👉 Va dans le panneau de droite sur Kaggle.")
    print("👉 Cherche le dossier 'Output' ou 'working/models'.")
    print("👉 Télécharge 'eye_state_model.h5' et 'yawn_model.h5' !")
