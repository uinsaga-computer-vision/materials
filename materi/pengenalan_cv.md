# Materi 1: Pengenalan Computer Vision & Tools

## 1.1 Konsep Dasar Computer Vision (3 Jam)

### Apa itu Computer Vision?
Computer Vision (CV) adalah bidang ilmu interdisipliner yang memungkinkan komputer untuk memperoleh, memproses, menganalisis, dan memahami informasi visual dari dunia nyata.

**Definisi Inti:**
- Kemampuan komputer untuk "melihat" dan "memahami" konten visual
- Ekstraksi informasi dari data visual (gambar/video)
- Interpretasi dan pemahaman makna dari data visual

### Sejarah Singkat Computer Vision
- **1960-an**: Project MAC di MIT - pertama kali mencoba membuat komputer "melihat"
- **1970-an**: Pengembangan algoritma edge detection dan feature extraction dasar
- **1980-an**: Pendekatan geometri dan model 3D
- **1990-an**: Statistika dan pembelajaran mesin mulai diterapkan
- **2000-an**: Ledakan data dan komputasi yang memungkinkan deep learning
- **2010-sekarang**: Dominasi deep learning dan convolutional neural networks (CNN)

### Komponen Utama Computer Vision

#### 1. Image Acquisition
- Proses mendapatkan gambar dari berbagai sumber
- Kamera digital, scanner, sensor medis, satelit, dll.
- Format file: JPEG, PNG, TIFF, RAW, DICOM

#### 2. Image Processing
- Enhancement: meningkatkan kualitas gambar
- Restoration: memperbaiki gambar yang rusak
- Compression: mengurangi ukuran file

#### 3. Image Analysis
- Segmentasi: memisahkan objek dari background
- Feature extraction: mengidentifikasi karakteristik penting
- Pattern recognition: mengenali pola dalam gambar

#### 4. Image Understanding
- Interpretasi makna dari gambar
- Konteks dan hubungan antar objek
- Decision making berdasarkan analisis

## 1.2 Aplikasi Computer Vision (45 Menit)

### A. Healthcare & Medicine
- **Diagnosis Medis**: Deteksi kanker, analisis MRI/CT scan
- **Telemedicine**: Konsultasi jarak jauh dengan analisis gambar
- **Bedah Robotik**: Bantuan visual untuk operasi presisi
- **Monitoring Pasien**: Analisis gerakan dan kondisi pasien

### B. Automotive & Transportation
- **Kendaraan Otonom**: Tesla, Waymo, mobil self-driving
- **ADAS**: Sistem bantuan pengemudi (lane detection, collision avoidance)
- **Parkir Otomatis**: Deteksi ruang parkir dan bantuan parkir
- **Traffic Management**: Analisis lalu lintas dan pengenalan plat nomor

### C. Retail & E-commerce
- **Visual Search**: Pencarian produk dengan gambar
- **Augmented Reality**: Preview produk di lingkungan nyata
- **Inventory Management**: Pelacakan stok otomatis
- **Cashier-less Stores**: Amazon Go - belanja tanpa kasir

### D. Security & Surveillance
- **Facial Recognition**: Pengenalan wajah untuk autentikasi
- **Anomaly Detection**: Deteksi perilaku mencurigakan
- **Object Tracking**: Pelacakan pergerakan objek
- **Perimeter Security**: Deteksi intrusi otomatis

### E. Agriculture & Environment
- **Precision Farming**: Analisis kesehatan tanaman
- **Crop Monitoring**: Deteksi penyakit dan hama
- **Wildlife Conservation**: Pelacakan dan monitoring hewan
- **Climate Monitoring**: Analisis perubahan lingkungan

### F. Industrial & Manufacturing
- **Quality Control**: Deteksi cacat produk otomatis
- **Robotic Guidance**: Panduan robot dalam assembly line
- **Predictive Maintenance**: Deteksi awal kerusakan mesin
- **Sorting Systems**: Klasifikasi dan pemilahan produk

### G. Entertainment & Media
- **Augmented Reality Games**: Pokemon Go, filter Instagram
- **Special Effects**: CGI dalam film dan video
- **Content Moderation**: Deteksi konten tidak pantas otomatis
- **Sports Analytics**: Analisis performa atlet

## 1.3 Python untuk Computer Vision (1 Jam)

### Mengapa Python untuk CV?
- **Syntax yang Mudah**: Mudah dipelajari dan dibaca
- **Ekosistem Kaya**: Banyak library dan framework
- **Komunitas Besar**: Dukungan dan resources melimpah
- **Integrasi Baik**: Bekerja baik dengan tools lain

### Library Python Penting untuk CV

#### 1. OpenCV (Open Source Computer Vision Library)
- Library paling populer untuk computer vision
- C++ dengan binding Python
- 2500+ algoritma optimized

```python
# Contoh penggunaan dasar OpenCV
import cv2
import numpy as np

# Membaca gambar
image = cv2.imread('image.jpg')

# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Menampilkan gambar
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 2. NumPy
- Fundamental untuk komputasi numerik
- Operasi array multidimensi yang efisien
- Dasar dari banyak library CV lainnya

```python
import numpy as np

# Membuat array gambar
image_array = np.array([[ [255, 0, 0], [0, 255, 0], [0, 0, 255] ],
                       [ [255, 255, 0], [255, 0, 255], [0, 255, 255] ]])

print("Shape:", image_array.shape)
print("Data type:", image_array.dtype)
```

#### 3. Matplotlib
- Visualisasi data dan gambar
- Plotting dan analisis hasil

```python
import matplotlib.pyplot as plt
import cv2

# Membaca dan menampilkan gambar
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title('Contoh Gambar dengan Matplotlib')
plt.axis('off')
plt.show()
```

#### 4. Scikit-image
- Collection of algorithms untuk image processing
- Built on top of NumPy and SciPy

```python
from skimage import io, filters
import matplotlib.pyplot as plt

# Membaca gambar
image = io.imread('image.jpg')

# Apply Gaussian filter
filtered_image = filters.gaussian(image, sigma=1)

plt.imshow(filtered_image)
plt.show()
```

#### 5. TensorFlow & Keras
- Deep learning frameworks
- Untuk CNN dan model complex lainnya

```python
from tensorflow.keras import layers, models

# Contoh simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

#### 6. PyTorch
- Alternative deep learning framework
- Popular untuk research

```python
import torch
import torchvision
import torch.nn as nn

# Contoh simple model dengan PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 15, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 15 * 15)
        x = self.fc1(x)
        return x
```

### Setup Environment Python untuk CV

#### 1. Install Python
```bash
# Download dari python.org atau gunakan package manager
# Untuk Ubuntu/Debian:
sudo apt update
sudo apt install python3 python3-pip

# Untuk Windows:
# Download installer dari python.org
```

#### 2. Virtual Environment
```bash
# Membuat virtual environment
python3 -m venv cv_env

# Aktifasi environment
# Linux/Mac:
source cv_env/bin/activate

# Windows:
cv_env\Scripts\activate
```

#### 3. Install Libraries
```bash
# Install library dasar
pip install opencv-python
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scikit-image

# Untuk deep learning
pip install tensorflow
# atau
pip install torch torchvision

# Jupyter Notebook untuk eksperimen
pip install jupyter
```

#### 4. Verifikasi Installasi
```python
# test_installation.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)

# Test simple operation
arr = np.array([1, 2, 3])
print("NumPy array:", arr)

print("All libraries imported successfully!")
```

## 1.4 OpenCV Deep Dive (45 Menit)

### Sejarah OpenCV
- Dikembangkan oleh Intel pada 1999
- Open source sejak 2000
- Sekarang didukung oleh OpenCV.org
- Ditulis dalam C++ dengan binding Python, Java, MATLAB

### Arsitektur OpenCV

#### 1. Core Module
- Struktur data dasar (Mat, Point, Size, Rect)
- Operasi array dan matrix
- Manajemen memory

#### 2. Image Processing Module
- Filtering dan transformasi
- Color conversion
- Histogram processing

#### 3. Video Module
- Video capture dan playback
- Background subtraction
- Object tracking

#### 4. Calib3d Module
- Camera calibration
- 3D reconstruction
- Epipolar geometry

#### 5. Features2d Module
- Feature detection dan extraction
- Feature matching
- Object recognition

#### 6. ML Module
- Machine learning algorithms
- Statistical classifiers
- Clustering

#### 7. DNN Module
- Deep Neural Networks
- Pre-trained models
- Inference optimization

### Fundamental OpenCV

#### Struktur Data Dasar

```python
import cv2
import numpy as np

# Mat object - struktur data utama
image = cv2.imread('image.jpg')
print("Type:", type(image))
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Size:", image.size)

# Point
point = (100, 200)
print("Point:", point)

# Rectangle
rect = (50, 50, 200, 100)  # x, y, width, height
print("Rectangle:", rect)
```

#### Basic Operations

```python
# Membaca dan menulis gambar
image = cv2.imread('input.jpg')
cv2.imwrite('output.jpg', image)

# Mengakses pixel
pixel_value = image[100, 50]  # y, x
print("Pixel value at (100,50):", pixel_value)

# Mengubah pixel
image[100, 50] = [255, 0, 0]  # Set to blue

# ROI (Region of Interest)
roi = image[100:200, 50:150]  # y1:y2, x1:x2

# Splitting dan merging channels
b, g, r = cv2.split(image)
merged = cv2.merge([b, g, r])

# Color conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

### Contoh Aplikasi Praktis dengan OpenCV

#### 1. Face Detection
```python
import cv2

# Load pre-trained classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read image
image = cv2.imread('group_photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display output
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 2. Edge Detection
```python
import cv2
import numpy as np

# Read image
image = cv2.imread('image.jpg', 0)  # Read as grayscale

# Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 3. Image Thresholding
```python
import cv2
import numpy as np

# Read image
image = cv2.imread('document.jpg', 0)  # Grayscale

# Different thresholding methods
_, thresh_binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
_, thresh_adaptive = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Binary Threshold', thresh_binary)
cv2.imshow('Otsu Threshold', thresh_adaptive)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Best Practices dalam OpenCV

#### 1. Memory Management
```python
# Good practice: Release resources
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame
cap.release()
cv2.destroyAllWindows()
```

#### 2. Error Handling
```python
import cv2
import sys

try:
    image = cv2.imread('non_existent.jpg')
    if image is None:
        raise FileNotFoundError("Image not found or unable to read")
    
    # Process image
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

#### 3. Performance Optimization
```python
# Use optimized operations
import time

start_time = time.time()

# Slow: Looping through pixels
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image[i, j] = [255, 255, 255] - image[i, j]

print("Loop time:", time.time() - start_time)

# Fast: Vectorized operation
start_time = time.time()
image = 255 - image
print("Vectorized time:", time.time() - start_time)
```

## Latihan Praktis (30 Menit)

### Exercise 1: Setup Environment
```bash
# 1. Buat virtual environment
python -m venv cv_course

# 2. Aktifkan environment
# Windows: cv_course\Scripts\activate
# Linux/Mac: source cv_course/bin/activate

# 3. Install required packages
pip install opencv-python numpy matplotlib jupyter

# 4. Verifikasi installasi
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Exercise 2: Basic Image Operations
```python
# basic_operations.py
import cv2
import numpy as np

# Create a simple image programmatically
image = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw different shapes
cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue rectangle
cv2.circle(image, (200, 100), 50, (0, 255, 0), -1)  # Green circle
cv2.line(image, (50, 200), (250, 200), (0, 0, 255), 5)  # Red line

# Add text
cv2.putText(image, 'OpenCV Demo', (75, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display image
cv2.imshow('Demo Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('demo_image.jpg', image)
```

### Exercise 3: Webcam Capture
```python
# webcam_capture.py
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display both frames
    cv2.imshow('Original', frame)
    cv2.imshow('Grayscale', gray)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

## Resources Tambahan

### Buku Recommendation
1. "Learning OpenCV" by Gary Bradski & Adrian Kaehler
2. "Computer Vision: Algorithms and Applications" by Richard Szeliski
3. "Deep Learning for Computer Vision" by Rajalingappaa Shanmugamani

### Online Courses
1. OpenCV Official Courses (opencv.org)
2. Coursera: Computer Vision Specialization
3. Udemy: Complete Computer Vision Course with Python

### Communities
1. OpenCV Forum (forum.opencv.org)
2. Stack Overflow (opencv tag)
3. GitHub OpenCV Repository

### Datasets untuk Latihan
1. MNIST - Handwritten digits
2. CIFAR-10/100 - Object recognition
3. ImageNet - Large-scale image database
4. COCO - Common Objects in Context
5. Kaggle Datasets - Various computer vision datasets

## Q&A Session (15 Menit)

### Pertanyaan Umum
1. **Apa perbedaan Computer Vision dengan Image Processing?**
   - Image Processing fokus pada manipulasi gambar untuk enhancement
   - Computer Vision fokus pada ekstraksi informasi dan understanding

2. **Mengapa OpenCV lebih populer daripada library lain?**
   - Open source dan free
   - Performance optimized (C++ backend)
   - Comprehensive functionality
   - Large community support

3. **Kapan sebaiknya menggunakan deep learning vs traditional CV?**
   - Traditional CV: Ketika dataset kecil, aturan jelas, butuh interpretability
   - Deep Learning: Ketika dataset besar, pattern complex, butuh accuracy tinggi

4. **Apa hardware requirement untuk belajar Computer Vision?**
   - Minimum: CPU dengan 4GB RAM
   - Recommended: GPU dengan CUDA support untuk deep learning
   - Cloud options: Google Colab, AWS, Azure

### Tips untuk Pemula
1. Start dengan fundamental - pahami basic image operations
2. Practice dengan small projects
3. Pelajari matematika dasar (linear algebra, calculus)
4. Ikuti tutorial dan replicate results
5. Bergabung dengan communities untuk bertanya dan berbagi

### Roadmap Learning Path
1. **Month 1-2**: Python basics + OpenCV fundamentals
2. **Month 3-4**: Image processing techniques
3. **Month 5-6**: Feature detection and machine learning
4. **Month 7-8**: Deep learning for computer vision
5. **Month 9-12**: Advanced topics and specialization

---
**Selamat belajar Computer Vision!** 🚀