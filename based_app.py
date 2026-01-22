import streamlit as st
import pickle
import numpy as np
import cv2
from utils.cnnHelper import ModerateCNN_GABOR_COMBINED, Conv2D, ReLU, MaxPool2x2, Flatten, Dense
from utils.preprocessing import resize_and_pad, grayscale


MODEL_PATH = 'model_cnn_gabor_5_dec.pkl'
CLASS_NAME = ['acantocyte', 'ellip', 'hypochromic', 'normal', 'sperobulat', 'stomachyocyte', 'targetsel', 'teardrop']

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print("success")
    return model


loaded_model = load_model(MODEL_PATH)

def predict_custom_image(model, image_path, class_names):
    """
    Fungsi untuk memprediksi 1 file gambar mentah.
    """
    try:
        # 1. Buka Gambar
        image_real = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # 2. Preprocessing (Wajib sama dengan Training!)
        img_gray = grayscale(image_real)
        img = resize_and_pad(img_gray)

        # 3. Normalisasi
        img_arr = np.asarray(img, dtype=np.float32) / 255.0

        # 4. Reshape: Tambah dimensi Batch & Channel
        # (H, W) -> (1, H, W, 1)
        # Angka 1 di depan artinya "Batch size = 1"
        input_data = img_arr.reshape(1, 64, 64, 1)

        # 5. Prediksi (Forward Pass)
        logits = model.forward(input_data)

        # 6. Hitung Confidence (Softmax)
        z = logits
        zmax = np.max(z, axis=1, keepdims=True)
        exps = np.exp(z - zmax)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        # 7. Ambil Hasil
        pred_idx = np.argmax(probs)
        confidence = probs[0][pred_idx] * 100
        pred_label = class_names[pred_idx]

        return img, pred_label, confidence

    except Exception as e:
        print(f"Error memproses {image_path}: {e}")
        return None, None, None

# Sesuaikan path
FILE_PATH = 'sample_pencil.png'


img, label, conf = predict_custom_image(loaded_model, FILE_PATH, CLASS_NAME)

print(f"Prediksi: {label}\nYakin: {conf:.2f}%")

