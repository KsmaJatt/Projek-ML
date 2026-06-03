# Klasifikasi Penyakit Tanaman Menggunakan Perbandingan CNN, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost

## Proyek

**Judul**: Klasifikasi Penyakit Tanaman Menggunakan Perbandingan CNN, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost pada Dataset PlantVillage untuk Mendukung Ketahanan Pangan
**Topik**: Computer Vision, Deep Learning, dan Klasifikasi Penyakit Tanaman
**Tujuan**: Proyek ini bertujuan untuk membangun sistem klasifikasi penyakit tanaman berbasis citra daun menggunakan pendekatan deep learning dan machine learning. Penelitian dilakukan dengan membandingkan performa CNN sederhana, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost untuk mengetahui model terbaik dalam mengklasifikasikan penyakit tanaman secara otomatis.

**Anggota Tim (Kelompok 3 - 2024C)**:

1. Chaesar Giveson (24031554058)
2. Sasmita Kusuma Jati (24031554052)
3. Nagatan Alief Putra Silahen (24031554086)

---

## Dataset

Dataset yang digunakan berasal dari:

[PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

PlantVillage merupakan dataset citra daun tanaman yang berisi gambar daun sehat dan daun yang terinfeksi berbagai penyakit tanaman. Dataset digunakan untuk tugas multi-class image classification menggunakan pendekatan deep learning.

### Informasi Dataset

* Total gambar digunakan: 20.638 citra
* Jumlah kelas: 15 kelas
* Resolusi gambar: 224×224 piksel
* Format warna: RGB
* Dataset bersifat multi-class classification

Contoh kelas:

* Pepper__bell___Bacterial_spot
* Pepper__bell___healthy
* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy
* Tomato_healthy
* Tomato__Target_Spot
* Tomato_Spider_mites_Two_spotted_spider_mite

---

## Tujuan

1. Membandingkan performa CNN sederhana, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost.
2. Mengimplementasikan transfer learning menggunakan EfficientNetB0 pretrained ImageNet.
3. Menggabungkan deep feature extraction dan XGBoost classifier pada pendekatan hybrid.
4. Mengevaluasi performa model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.
5. Menentukan model terbaik untuk klasifikasi penyakit tanaman berbasis citra daun.

---

## Metode

### 1. Preprocessing Citra

Tahapan preprocessing yang dilakukan:

* Resize gambar menjadi 224×224 piksel
* Normalisasi nilai piksel
* Label encoding
* Train-validation split (80:20)
* Data augmentation

### 2. CNN Sederhana

CNN sederhana digunakan sebagai baseline model untuk melihat performa deep learning tanpa pretrained model.

Arsitektur CNN:

* Conv2D 32 + ReLU
* MaxPooling2D
* Conv2D 64 + ReLU
* MaxPooling2D
* Conv2D 128 + ReLU
* MaxPooling2D
* Flatten
* Dropout 0.5
* Dense 256 + ReLU
* Softmax Output 15 kelas

### 3. EfficientNetB0

EfficientNetB0 digunakan sebagai model transfer learning menggunakan pretrained weights dari ImageNet.

Arsitektur:

* EfficientNetB0 (include_top=False)
* GlobalAveragePooling2D
* Dropout 0.3
* Dense 256 + ReLU
* Dense Softmax

Training dilakukan dalam dua fase:

* Freeze seluruh layer EfficientNetB0
* Fine-tuning 20 layer terakhir

### 4. Hybrid EfficientNetB0 + XGBoost

Pendekatan hybrid menggunakan:

* EfficientNetB0 sebagai feature extractor
* Feature vector hasil ekstraksi sebagai input XGBoost
* XGBoost sebagai classifier akhir

Parameter XGBoost:

* n_estimators = 200
* max_depth = 6
* learning_rate = 0.1
* eval_metric = mlogloss

### 5. Evaluasi Model

Evaluasi model dilakukan menggunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Struktur Folder

```text
📁 Projek-ML/
├── notebook/
│   ├── Projek_ML_imbalanced.ipynb
│   └── Projek_ML_balanced.ipynb
│
├── model/
│   ├── cnn_best.keras
│   ├── effnet_phase2_best.keras
│   ├── xgb_model.pkl
│   ├── cnn_balanced_best.keras
│   ├── effnet_phase2_balanced_best.keras
│   └── xgb_balanced_model.pkl
│
├── hasil/
│   ├── confusion_matrix/
│   ├── classification_report/
│   └── training_curve/
│
├── visualisasi/
│   ├── distribusi_kelas.png
│   ├── cnn_curve.png
│   ├── efficientnet_curve.png
│   ├── xgboost_curve.png
│   └── perbandingan_imbalanced_vs_balanced.png
│
├── README.md
└── requirements.txt
```

---

### 6. Penanganan Class Imbalance (Eksperimen Balanced)
- CNN & EfficientNetB0: `compute_class_weight('balanced')`
- XGBoost: `compute_sample_weight('balanced')`

---

## Hasil Eksperimen

### Kondisi Imbalanced (Tanpa Class Weighting)

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| **EfficientNetB0** | **98.50%** | **0.98** | **0.99** |
| Hybrid XGBoost | 97.67% | 0.98 | 0.97 |
| CNN Sederhana | 93.84% | 0.94 | 0.93 |

### Kondisi Balanced (Dengan Class Weighting)

| Model | Accuracy | Weighted F1 | Macro F1 |
|-------|----------|-------------|----------|
| **Hybrid XGBoost** | **95.15%** | **0.95** | 0.94 |
| EfficientNetB0 | 94.76% | 0.95 | **0.95** |
| CNN Sederhana | 87.77% | 0.88 | 0.85 |

### Perbandingan Imbalanced vs Balanced

| Model | Imbalanced | Balanced | Selisih |
|-------|-----------|---------|---------|
| EfficientNetB0 | 98.50% | 94.76% | -3.74% |
| Hybrid XGBoost | 97.67% | 95.15% | -2.52% |
| CNN Sederhana | 93.84% | 87.77% | -6.07% |

---

## Temuan Utama

- **EfficientNetB0 imbalanced** adalah model terbaik secara keseluruhan dengan accuracy **98.50%** dan Macro F1 **0.99**
- **Hybrid XGBoost balanced** sedikit mengungguli EfficientNetB0 pada kondisi balanced (95.15% vs 94.76%), menunjukkan XGBoost lebih responsif terhadap sample weighting
- **Class weighting justru menurunkan performa** ketiga model, membuktikan bahwa transfer learning sudah cukup robust terhadap class imbalance tanpa teknik balancing tambahan
- **Tomato_Early_blight** dan **Tomato__Target_Spot** secara konsisten menjadi kelas tersulit di semua model akibat kemiripan visual yang tinggi
- CNN sederhana mengalami overfitting ringan pada epoch akhir namun berhasil dimitigasi oleh EarlyStopping dan ReduceLROnPlateau

---

## Framework dan Library

* TensorFlow / Keras
* EfficientNetB0
* XGBoost
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

Seluruh eksperimen dijalankan menggunakan GPU acceleration untuk mempercepat proses training deep learning.

---

## Alur Pengerjaan

```text
Dataset PlantVillage
↓
Preprocessing Citra
↓
Resize dan Normalisasi
↓
Augmentasi Data
↓
Training CNN Sederhana
↓
Training EfficientNetB0
↓
Ekstraksi Fitur EfficientNetB0
↓
Klasifikasi dengan XGBoost
↓
Evaluasi Model
↓
Perbandingan Hasil
↓
Kesimpulan
```

---

## Syarat

* Python 3.10 atau lebih tinggi
* TensorFlow
* XGBoost
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

Install dependensi:

```bash
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

## Penggunaan

1. Clone repository:

```bash
git clone https://github.com/KsmaJatt/Projek-ML.git
```

2. Jalankan notebook:

```bash
Projek_ML_imbalanced.ipynb  → eksperimen tanpa class weighting
Projek_ML_balanced.ipynb    → eksperimen dengan class weighting
```

3. Buka file notebook:

```bash
Projek_ML_balanced.ipynb
```

4. Jalankan seluruh cell untuk:

* preprocessing data
* training model
* evaluasi model
* visualisasi hasil

---

## Pengembangan Selanjutnya

* Pengujian pada citra daun di lingkungan nyata (lahan pertanian)
* Eksplorasi Focal Loss sebagai alternatif class weighting
* Implementasi Grad-CAM
* Deployment model berbasis web/mobile
* Eksplorasi arsitektur lain: ResNet50, MobileNetV3, Vision Transformer

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik sebagai bagian dari tugas Pembelajaran Mesin S1 Sains Data Universitas Negeri Surabaya.
