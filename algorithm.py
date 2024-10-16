import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def load_and_resize_image(image_path, new_size=(100, 100)):
    """Load and resize the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    resized_img = cv2.resize(img, new_size)
    return resized_img

def preprocess_image(image: np.ndarray, image_size=(75, 75)) -> np.ndarray:
    """Memproses gambar dengan denoising dan sharpening."""
    if image.shape[2] == 4:  # Jika gambar memiliki 4 channel (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(image, image_size)

    # Menghilangkan noise gambar
    img_denoised = cv2.GaussianBlur(img, (5, 5), 0)

    # Sharpening gambar
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharpened = cv2.filter2D(img_denoised, -1, kernel)

    return img_sharpened

def kmeans_clustering(pixel_values, n_clusters=3, max_iter=100, tol=1e-4):
    """Melakukan K-Means clustering pada pixel gabungan."""
    np.random.seed()  # Inisialisasi acak untuk centroid
    centroids = pixel_values[np.random.choice(pixel_values.shape[0], n_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(pixel_values - centroid, axis=1) for centroid in centroids])
        cluster_labels = np.argmin(distances, axis=0)
        new_centroids = np.array([pixel_values[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
        new_centroids = np.clip(new_centroids, 0, 255)

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, cluster_labels

def segment_images(images, n_clusters):
    """Segmentasi gambar berdasarkan centroid yang dihitung dari semua gambar."""
    all_pixel_values = []
    image_shapes = []  # Menyimpan ukuran asli gambar

    # Proses setiap gambar
    for image in images:
        processed_image = preprocess_image(np.array(image))
        pixel_values = processed_image.reshape(-1, 3)
        all_pixel_values.append(pixel_values)
        image_shapes.append(processed_image.shape[:2])  # Simpan dimensi gambar

    all_pixel_values = np.vstack(all_pixel_values)  # Gabungkan semua pixel

    # Lakukan clustering untuk jumlah kluster yang ditentukan
    centroids, cluster_labels = kmeans_clustering(all_pixel_values, n_clusters)

    # Hitung Silhouette Score
    if len(set(cluster_labels)) > 1:  # Pastikan ada lebih dari satu cluster
        silhouette_avg = silhouette_score(all_pixel_values, cluster_labels)
        print(f'Silhouette Score untuk {n_clusters} kluster: {silhouette_avg:.4f}')
    else:
        print("Silhouette Score tidak dapat dihitung, karena hanya ada satu cluster.")

    # Membuat gambar tersegmentasi untuk kluster yang diproses
    segmented_images = []
    start_index = 0

    for shape in image_shapes:
        h, w = shape
        end_index = start_index + (h * w)  # Hitung index akhir untuk reshaping
        segmented_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Mengambil cluster labels untuk gambar ini
        cluster_labels_reshaped = cluster_labels[start_index:end_index].reshape(h, w)

        for cluster in range(n_clusters):
            segmented_image[cluster_labels_reshaped == cluster] = centroids[cluster].astype(int)

        segmented_images.append(segmented_image)
        start_index = end_index  # Update start_index untuk gambar berikutnya

    return segmented_images

def visualize_clusters(segmented_images, n_clusters):
    """Menampilkan gambar-gambar yang tersegmentasi."""
    fig, axes = plt.subplots(1, len(segmented_images), figsize=(15, 5))
    
    for ax, segmented_image in zip(axes, segmented_images):
        ax.imshow(segmented_image)
        ax.axis('off')
    
    plt.suptitle(f'Segmented Images with {n_clusters} Clusters', fontsize=16)
    plt.show()

# Fungsi utama untuk menguji
if __name__ == "__main__":
    folder_path = 'img'  # Path ke folder yang berisi gambar
    output_folder = 'clustering'  # Folder untuk menyimpan gambar tercluster

    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Ambil semua gambar
    images = [load_and_resize_image(os.path.join(folder_path, f)) 
              for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Segmentasi gambar dengan rentang kluster
    n_clusters = 3
    segmented_images = segment_images(images, n_clusters)

    # Visualisasi hasil clustering
    visualize_clusters(segmented_images, n_clusters)  # Tampilkan hasil untuk jumlah kluster yang ditentukan
