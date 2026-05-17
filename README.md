# Klasifikasi Penyakit Tanaman Menggunakan Perbandingan CNN, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost

Proyek ini bertujuan untuk mengklasifikasikan penyakit tanaman berdasarkan citra daun menggunakan dataset PlantVillage. Dataset ini berisi gambar daun tanaman sehat dan daun tanaman yang terkena penyakit. Pada proyek ini digunakan tiga pendekatan model, yaitu CNN sederhana, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost.

Model CNN sederhana digunakan sebagai baseline, EfficientNetB0 digunakan sebagai model transfer learning, sedangkan Hybrid EfficientNetB0-XGBoost digunakan sebagai model utama. EfficientNetB0 berperan sebagai feature extractor untuk mengambil fitur visual dari gambar daun, kemudian XGBoost digunakan sebagai classifier untuk menentukan kelas penyakit tanaman.

# Kelompok 3

| Nama | NIM |
|---|---|
| Chaesar Giveson | 24031554058 |
| Sasmita Kusuma Jati | 24031554052 |
| Nagatan Alief Putra Silahen | 24031554086 |

## Dataset

[PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

PlantVillage Dataset merupakan dataset citra daun tanaman yang berisi berbagai jenis tanaman sehat dan tanaman berpenyakit. Dataset ini digunakan untuk tugas image classification, yaitu mengklasifikasikan gambar daun ke dalam kelas tertentu berdasarkan jenis tanaman dan kondisi penyakitnya.

Dataset ini memiliki beberapa kelas, seperti daun sehat dan daun yang terkena penyakit pada tanaman tertentu. Karena data yang digunakan berbentuk gambar, fitur utama yang digunakan adalah informasi visual dari citra daun, seperti warna, tekstur, bercak, pola kerusakan, dan bentuk gejala penyakit.

Proposal kami yang berjudul **"Klasifikasi Penyakit Tanaman Menggunakan Perbandingan CNN, EfficientNetB0, dan Hybrid EfficientNetB0-XGBoost pada Dataset PlantVillage untuk Mendukung Ketahanan Pangan"** mengajukan pengembangan sistem klasifikasi penyakit tanaman menggunakan perbandingan tiga model. Tujuannya adalah untuk mengetahui model mana yang memberikan performa terbaik dalam mengklasifikasikan penyakit tanaman berdasarkan citra daun.

## Metode

1. Preprocessing citra:
   - Resize gambar
   - Normalisasi nilai piksel
   - Label encoding
   - Train-test split
   - Augmentasi data

2. Model CNN sederhana:
   - Digunakan sebagai baseline untuk melihat performa dasar klasifikasi citra daun.

3. Model EfficientNetB0:
   - Menggunakan transfer learning dengan EfficientNetB0 pretrained ImageNet.
   - Digunakan sebagai model pembanding yang lebih kuat dan efisien.

4. Model Hybrid EfficientNetB0-XGBoost:
   - EfficientNetB0 digunakan sebagai feature extractor.
   - Feature vector hasil ekstraksi digunakan sebagai input untuk XGBoost.
   - XGBoost digunakan sebagai classifier utama.

5. Evaluasi model:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

6. Perbandingan model:
   - CNN sederhana
   - EfficientNetB0
   - Hybrid EfficientNetB0-XGBoost

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
