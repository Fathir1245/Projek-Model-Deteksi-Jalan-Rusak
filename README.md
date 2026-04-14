# 🚧 Road Surface Classification using Deep Learning (MobileNetV2)

Proyek ini merupakan implementasi model **Deep Learning berbasis Convolutional Neural Network (CNN)** untuk mengklasifikasikan kondisi permukaan jalan menjadi 3 kategori:

* 🟢 Jalan Baik
* 🟡 Jalan Kurang Baik
* 🔴 Jalan Rusak

Model dikembangkan menggunakan **Transfer Learning dengan MobileNetV2** dan dilatih menggunakan dataset citra jalan.

---

## 🎯 Tujuan Proyek

* Mengidentifikasi kondisi jalan secara otomatis dari citra
* Membantu monitoring infrastruktur jalan berbasis AI
* Menjadi implementasi nyata Deep Learning dalam bidang Smart Environment

---

## 🧠 Metode yang Digunakan

* **Transfer Learning** menggunakan MobileNetV2
* **Image Classification (3 kelas)**
* **Data Augmentation** untuk meningkatkan variasi data
* **Early Stopping** untuk menghentikan training otomatis
* **Model Checkpoint** untuk menyimpan model terbaik
* **Reduce Learning Rate** untuk optimasi training

---

## 📂 Struktur Project

```
Projek-Deni/
│
├── road_surface_dataset/
│   ├── Jalan_Kategori_Baik/
│   ├── Jalan_Kurang_Baik/
│   └── Jalan_Rusak/
│
├── model/
│   ├── model_terbaik.h5
│   └── model_final.h5
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
└── README.md
```

---

## 📊 Dataset

Dataset terdiri dari:

* 250 gambar jalan baik
* 250 gambar jalan kurang baik
* 300 gambar jalan rusak

Total: **800 gambar**

Link : https://www.kaggle.com/datasets/angelikafarahmanoppo/road-surface-dataset-datasetjalan

---

## ⚙️ Instalasi

1. Clone repository:

```
git clone https://github.com/username/repository-name.git
cd repository-name
```

2. Buat virtual environment:

```
python -m venv venv
```

3. Aktifkan environment:

Windows:

```
venv\Scripts\activate
```

4. Install dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Training Model

Jalankan perintah berikut:

```
python src/train.py
```

Fitur training:

* Early Stopping
* Model Checkpoint
* Data Augmentation

---

## 📈 Evaluasi Model

Untuk melihat performa model:

```
python src/evaluate.py
```

Evaluasi meliputi:

* Confusion Matrix
* Precision, Recall, F1-score

---

## 📊 Hasil Model

* Accuracy: **98%**
* Model mampu mengklasifikasikan sebagian besar data dengan sangat baik

### 🔍 Insight:

* Kelas *Jalan Baik* dan *Kurang Baik* terdeteksi sempurna
* Sedikit kesalahan terjadi pada kelas *Jalan Rusak* karena kemiripan visual

---

## 🖼️ Visualisasi

Model menghasilkan:

* Grafik Accuracy
* Grafik Loss
* Confusion Matrix

---

## 🧩 Teknologi yang Digunakan

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

---

## 🔥 Pengembangan Selanjutnya

* Menambahkan dataset lebih banyak
* Implementasi Object Detection (YOLO)
* Deploy ke Web App
* Integrasi dengan sistem Smart City

---

## 👨‍💻 Author

Dibuat oleh:
**Fadhil Akmal**

---

## 📌 Catatan

Model ini masih dapat dikembangkan lebih lanjut dengan dataset yang lebih besar dan variasi kondisi lingkungan yang lebih kompleks.

---

## ⭐ Kontribusi

Silakan fork repository ini dan kembangkan lebih lanjut 🚀
