import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from algorithm import (
    load_and_resize_image,
    segment_images,
)

# Set custom CSS for the app
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;  /* Latar belakang hitam */
    }
    h1 {
        color: #4CAF50;  /* Warna hijau untuk judul */
        font-family: 'Arial', sans-serif;  /* Font judul */
        text-align: center;  /* Rata tengah */
    }
    h4 {
        color: #FFFFFF;  /* Warna putih untuk subjudul */
        font-family: 'Arial', sans-serif;  /* Font subjudul */
        text-align: center;  /* Rata tengah */
    }
    .upload-container {
        border: 2px dashed #4CAF50;  /* Garis batas hijau untuk area upload */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #1F1F1F;  /* Latar belakang abu-abu gelap untuk area upload */
        margin-bottom: 20px;  /* Jarak antar elemen */
    }
    .card {
        border: 1px solid #4CAF50;  /* Garis batas hijau untuk kartu */
        border-radius: 10px;  /* Sudut melengkung */
        padding: 10px;
        background-color: #1F1F1F;  /* Latar belakang abu-abu gelap untuk kartu */
        box-shadow: 0 2px 5px rgba(255, 255, 255, 0.1);  /* Bayangan halus */
        margin-bottom: 20px;  /* Jarak antar kartu */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul aplikasi
st.title("Dashboard Kluster Citra Udara")

# Fungsi upload gambar
def upload_images(max_files):
    uploaded_files = st.file_uploader(
        f"Pilih satu gambar untuk diproses:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False  # Hanya satu gambar
    )
    return uploaded_files

def display_instructions(header):
    st.markdown(
        f"<h4 style='color: #1F5B8B;'>{header}</h4>", unsafe_allow_html=True)  # Warna baru untuk instruksi

# Menampilkan instruksi untuk upload gambar
display_instructions("Upload Gambar untuk Segmentasi")
uploaded_image = upload_images(1)

# Jika gambar diupload
if uploaded_image is not None:
    original_image = Image.open(uploaded_image)  # Membuka gambar
    st.success("Gambar berhasil di-upload!")

    # Menampilkan gambar yang diupload
    st.subheader("Gambar yang Diupload")
    st.image(original_image, use_column_width=True)

    # Pengaturan KMeans Klustering
    num_clusters = st.slider("Pilih jumlah cluster (2-5):", min_value=2, max_value=5, value=3)

    # Spinner untuk preprocessing dan segmentasi
    with st.spinner("Memproses gambar..."):
        segmented_image = segment_images([original_image], n_clusters=num_clusters)[0]  # Memproses gambar

    st.success("Proses segmentasi selesai!")

    # Tampilkan hasil segmentasi
    st.subheader("Hasil Segmentasi Gambar")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Gambar Asli", use_column_width=True)
    
    with col2:
        st.image(segmented_image, caption="Gambar Tersegmentasi", use_column_width=True)

# Informasi jika tidak ada gambar yang diupload
if uploaded_image is None:
    st.info("Silakan upload gambar untuk melihat hasil analisis.")

# Menampilkan anggota tim
st.markdown("---")  
st.markdown("""<div style="text-align: center;"><h4>Tim Pengembang</h4></div>""",
            unsafe_allow_html=True)

# Fungsi untuk menampilkan gambar bulat
def make_rounded_image(image_path):
    img = Image.open(image_path).convert("RGB")
    size = (150, 150)

    # Mengubah ukuran gambar dengan menjaga aspek rasio
    img.thumbnail(size, Image.LANCZOS)

    # Membuat mask bulat
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)

    # Membuat gambar bulat
    rounded_img = ImageOps.fit(img, mask.size, centering=(0.5, 0.3))
    rounded_img.putalpha(mask)
    return rounded_img

# Tampilkan anggota tim
cols = st.columns(3)
with cols[0]:
    st.image(make_rounded_image('public/rumi.jpg'), width=150,
             caption="Muhammad Rumi Rifai\n140810220026")
with cols[1]:
    st.image(make_rounded_image('public/adnan2.jpg'), width=150,
             caption="Adnan Hafizh Sinatria\n140810220048")
with cols[2]:
    st.image(make_rounded_image('public/rio.jpg'), width=150,
             caption="Rio Irawan\n140810220084")