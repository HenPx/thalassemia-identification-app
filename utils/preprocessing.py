import cv2
import numpy as np

def grayscale(img):
    """Konversi gambar ke grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# target 64px x 64 px
TARGET_SIZE_RESIZE = 64
# fungsi resize custom
def resize_and_pad(img, target_size=TARGET_SIZE_RESIZE):
    """
    Mengubah ukuran gambar dengan padding hitam agar pas 256x256
    tanpa distorsi aspek rasio.
    """
    # Mencari nilai terbesar dan scaling
    h, w = img.shape[:2]
    max_dim = max(w, h)
    scale_factor = target_size / max_dim

    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)

    resized = cv2.resize(img, (new_width, new_height))

    # menambal kekurangan pixel dengan padding 0 atau hitam
    top_pad = (target_size - new_height) // 2
    bottom_pad = target_size - new_height - top_pad
    left_pad = (target_size - new_width) // 2
    right_pad = target_size - new_width - left_pad

    padded_img = cv2.copyMakeBorder(
        resized,
        top=top_pad,
        bottom=bottom_pad,
        left=left_pad,
        right=right_pad,
        borderType=cv2.BORDER_CONSTANT,
        value=0)
    return padded_img


# --- FUNGSI PROSES & PREDIKSI ---
def process_and_predict(image_real, model, class_names):
    try:
        img_gray = grayscale(image_real)
        img_pre = resize_and_pad(img_gray)
        img_arr = np.asarray(img_pre, dtype=np.float32) / 255.0
        input_data = img_arr.reshape(1, 64, 64, 1)
        
        logits = model.forward(input_data)
        
        z = logits
        zmax = np.max(z, axis=1, keepdims=True)
        exps = np.exp(z - zmax)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        pred_idx = np.argmax(probs)
        confidence = probs[0][pred_idx] * 100
        pred_label = class_names[pred_idx]

        return img_pre, pred_label, confidence
    except Exception as e:
        return None, None, None