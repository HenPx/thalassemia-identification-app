import streamlit as st
import pickle, cv2, zipfile, math
import numpy as np
import pandas as pd 
from utils.cnnHelper import ModerateCNN_GABOR_COMBINED, Conv2D, ReLU, MaxPool2x2, Flatten, Dense
from utils.preprocessing import process_and_predict

st.set_page_config(page_title="Prediksi Eritrosit Thalassemia")

MODEL_PATH = 'model_cnn_gabor_5_dec.pkl'
CLASS_NAME = ['Acanthocyte', 'Elliptocyte', 'Hypochromic', 'Normal', 'Spherocyte', 'Stomatocyte', 'Target Cell', 'Teardrop']

# --- INITIALIZE SESSION STATE ---
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = []
if 'page_number' not in st.session_state:
    st.session_state['page_number'] = 0

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model_cached(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

loaded_model = load_model_cached(MODEL_PATH)

st.title("🔬 Identifikasi Eritrosit Thalassemia dengan Deep Learning")

st.markdown("""
    Aplikasi ini dirancang sebagai implementasi penelitian untuk memprediksi jenis sel darah eritrosit (**Red Blood Cells**) menggunakan arsitektur model **CNN berbasis Gabor**. 
    Tujuannya adalah membantu identifikasi morfologi sel yang sering dikaitkan dengan kelainan Thalassemia.""")

with st.expander("ℹ️ Catatan Penggunaan", expanded=True):
    st.markdown("""
    Harap perhatikan poin berikut:
    1.  **Input Gambar:** Pastikan gambar yang diunggah adalah **citra sel tunggal (single cell)**, bukan gambar apusan darah utuh yang berisi banyak sel.
    2.  **Arsitektur Model:** Prediksi dilakukan menggunakan model yang telah dilatih khusus pada dataset penelitian menggunakan CNN dan filter Gabor.
    3.  **Keterbatasan:** Akurasi bergantung pada kualitas citra input (pencahayaan, fokus, dan resolusi).
    3.  **Jenis Sel:** Jenis citra sel yang dapat dikenali diperlihatkan pada tabel berikut).
    """)

    st.write("### 📋 Tabel Daftar Sel")
    st.write("Berikut adalah jenis sel yang dapat dikenali oleh model beserta indikasi medisnya dalam konteks Thalassemia:")

    data_sel = {
        'Tipe Sel RBC': [
            'Teardrop', 'Normal', 'Elliptocyte', 'Target Cell', 
            'Spherocyte', 'Stomatocyte', 'Acanthocyte', 'Hypochromic'
        ],
        'Terindikasi Thalassemia': [
            'Ya', 'Tidak', 'Ya', 'Ya', 
            'Tidak', 'Tidak', 'Ya', 'Ya'
        ]
    }
    
    df_info = pd.DataFrame(data_sel)
    st.table(df_info)

    st.warning("""
    **PENTING:** Hasil prediksi dari aplikasi ini bersifat **indikatif** untuk keperluan penelitian. 
    Aplikasi ini **tidak menggantikan diagnosis medis profesional**. Konsultasikan hasil dengan ahli patologi klinik untuk validasi lebih lanjut.
    """)
st.info("Terdapat dua mode prediksi yang dapat dipilih:\n1. Satu gambar tunggal\n2. Batch gambar dalam file ZIP.")

tab1, tab2 = st.tabs(["🖼️ Single Image", "📦 Batch (Zip File)"])

# === TAB 1: SINGLE IMAGE ===
with tab1:
    st.header("Upload Satu Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'], key="single")

    if uploaded_file is not None and loaded_model:
        st.success("Status input Gambar:\nSukses")

        if st.button("Prediksi"):

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_real = cv2.imdecode(file_bytes, 1)

            img_pre, label, conf = process_and_predict(
                image_real, loaded_model, CLASS_NAME
            )

            st.metric(label="Kelas", value=label)
            st.metric(label="Confidence", value=f"{conf:.2f}%")

            c1, c2 = st.columns([2, 2])

            with c1:
                st.image(image_real, caption="Asli", channels="BGR", width='stretch')

            if img_pre is not None:
                with c2:
                    st.image(img_pre, caption="Preprocessing", width=150, clamp=True)
# === TAB 2: ZIP FILE
with tab2:
    st.header("Upload File ZIP")
    zip_file = st.file_uploader("Pilih file ZIP...", type="zip", key="zip")

    if zip_file is not None and loaded_model:
        if st.button("Prediksi"):
            st.session_state['batch_results'] = []
            st.session_state['page_number'] = 0 
            
            with zipfile.ZipFile(zip_file, "r") as z:
                image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '__MACOSX' not in f]
                
                if not image_files:
                    st.warning("ZIP kosong atau tidak ada gambar.")
                else:
                    progress_bar = st.progress(0)
                    temp_results = []
                    
                    for i, filename in enumerate(image_files):
                        content = z.read(filename)
                        file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
                        image_real = cv2.imdecode(file_bytes, 1)

                        if image_real is not None:
                            img_pre, label, conf = process_and_predict(image_real, loaded_model, CLASS_NAME)
                            
                            temp_results.append({
                                'filename': filename,
                                'image_real': image_real, 
                                'image_pre': img_pre,
                                'label': label,
                                'confidence': conf
                            })
                        progress_bar.progress((i + 1) / len(image_files))
                    
                    st.session_state['batch_results'] = temp_results
                    st.success(f"Selesai memproses {len(temp_results)} gambar!")
    results = st.session_state['batch_results']
    if len(results) > 0:
        st.divider()
        st.subheader("📊 Hasil Prediksi")
        df_res = pd.DataFrame(results)
        counts = df_res['label'].value_counts().reset_index()
        counts.columns = ['Kelas', 'Jumlah']
        col_sum1, col_sum2 = st.columns([1, 1])
        with col_sum1:
            st.dataframe(counts, hide_index=True, width='stretch')
        
        with col_sum2:
            st.bar_chart(counts, x='Kelas', y='Jumlah')

        st.divider()
        st.subheader("📄 Detail Prediksi Gambar")
        
        # Konfigurasi Pagination
        ITEMS_PER_PAGE = 5
        total_items = len(results)
        total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
        
        if st.session_state['page_number'] >= total_pages:
            st.session_state['page_number'] = total_pages - 1
        if st.session_state['page_number'] < 0:
            st.session_state['page_number'] = 0
            
        current_page = st.session_state['page_number']
        
        start_idx = current_page * ITEMS_PER_PAGE
        end_idx = start_idx + ITEMS_PER_PAGE
        page_items = results[start_idx:end_idx]
        
        # Tampilkan Item di Halaman Ini
        for item in page_items:
            with st.container():
                st.markdown(f"**File:** `{item['filename']}`")
                c1, c2, c3 = st.columns([2, 2, 1])
                
                with c1:
                    st.image(item['image_real'], channels="BGR", width=100, caption="Asli")
                with c2:
                    st.image(item['image_pre'], width=100, caption="Prepro", clamp=True)
                with c3:
                    st.markdown(f"### {item['label']}")
                    st.caption(f"Confidence: {item['confidence']:.2f}%")
                st.divider()

        # TOMBOL NAVIGASI HALAMAN
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("Sebelumnya", disabled=(current_page == 0)):
                st.session_state['page_number'] -= 1
                st.rerun() 
        
        with col_info:
            st.markdown(f"<div style='text-align: center'>Halaman <b>{current_page + 1}</b> dari {total_pages}</div>", unsafe_allow_html=True)
        
        with col_next:
            if st.button("Berikutnya", disabled=(current_page == total_pages - 1)):
                st.session_state['page_number'] += 1
                st.rerun() 