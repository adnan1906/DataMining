import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk membaca dan resize gambar
def load_and_resize_image(image_path, new_size=(100, 100)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi dari BGR ke RGB
    resized_img = cv2.resize(img, new_size)
    return resized_img

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Fungsi untuk algoritma K-Means buatan sendiri
def kmeans_clustering(image, n_clusters=3, max_iter=100):
    # Ubah gambar menjadi dua dimensi (lebar x tinggi, dan channel)
    pixel_values = image.reshape((-1, 3))
    
    # Inisialisasi centroid secara acak
    np.random.seed(42)
    centroids = pixel_values[np.random.choice(pixel_values.shape[0], n_clusters, replace=False)]
    
    for _ in range(max_iter):
        # Tentukan jarak setiap piksel ke setiap centroid
        distances = np.array([np.linalg.norm(pixel_values - centroid, axis=1) for centroid in centroids])
        
        # Tentukan cluster terdekat untuk setiap piksel
        cluster_labels = np.argmin(distances, axis=0)
        
        # Hitung ulang centroid berdasarkan rata-rata piksel di setiap cluster
        new_centroids = np.array([pixel_values[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Cek apakah ada nilai centroid yang di luar rentang 0-255, kemudian koreksi
        new_centroids = np.clip(new_centroids, 0, 255)
        
        # Jika tidak ada perubahan pada centroid, maka berhenti
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Ubah setiap piksel menjadi nilai centroid hasil clustering
    clustered_img = centroids[cluster_labels]
    clustered_img = clustered_img.reshape(image.shape)
    
    return clustered_img.astype(np.uint8), cluster_labels

# Fungsi untuk menampilkan hasil clustering dengan label cluster
def show_clustered_image_with_labels(original_image, clustered_image, cluster_labels, n_clusters):
    plt.figure(figsize=(10, 5))
    
    # Tampilkan gambar asli
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    # Tampilkan hasil clustering
    plt.subplot(1, 2, 2)
    plt.title(f'Clustered Image ({n_clusters} clusters)')
    plt.imshow(clustered_image)
    plt.axis('off')
    
    # Tambahkan label di gambar cluster
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        mask = (cluster_labels == label)
        # Temukan posisi rata-rata piksel yang ada di cluster untuk menaruh labelnya
        y, x = np.mean(np.column_stack(np.where(mask.reshape(original_image.shape[:2]))), axis=0)
        plt.text(x, y, f'Cluster {label+1}', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.show()

# Main program
if __name__ == "__main__":
    # Path ke gambar yang ingin di-clustering
    image_path = os.path.join('img', 'Perkampusan.png')
    
    # Load dan resize gambar tanpa normalisasi
    resized_image = load_and_resize_image(image_path, (100, 100))
    
    # Lakukan clustering dari 1 sampai 5 cluster
    for n_clusters in range(1, 6):
        clustered_image, cluster_labels = kmeans_clustering(resized_image, n_clusters)
        
        # Simpan hasil clustering ke file
        output_image_path = f'clustered_image_{n_clusters}_clusters.png'
        cv2.imwrite(output_image_path, cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))
        
        # Tampilkan hasil dengan label cluster
        show_clustered_image_with_labels(resized_image, clustered_image, cluster_labels, n_clusters)
