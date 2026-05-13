# Klasifikasi Penyakit Daun Padi dengan Hybrid CNN-XGBoost
Proyek ini mengklasifikasikan penyakit daun padi (Bacterial Blight, Blast,Brown Spot, Healthy) dari citra menggunakan pendekatan hybrid: MobileNetV2 sebagai feature extractor dan XGBoost sebagai classifier.

# Kelompok 3

| Nama | NIM |
|---|---|
| Chaesar Giveson | 24031554058 |
| Sasmita Kusuma Jati | 24031554052 |
| Nagatan Alief Putra Silahen | 24031554086 |

## Dataset 
[Rice Leaf Disease Image Dataset — Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)

Rice Leaf Disease Image Dataset berisi citra daun padi untuk klasifikasi beberapa jenis penyakit tanaman seperti Bacterial Blight, Blast, Brown Spot, dan Healthy. Dataset digunakan untuk membantu proses deteksi penyakit tanaman secara otomatis menggunakan pendekatan machine learning dan deep learning. Dataset memiliki ukuran lebih dari 100 MB dengan format gambar .jpg dan cocok digunakan untuk image classification.

Proposal kami yang berjudul **"Klasifikasi Penyakit Daun Padi Menggunakan Pendekatan Hybrid CNN-XGBoost untuk Mendukung Ketahanan Pangan"** mengajukan pengembangan sistem klasifikasi penyakit daun padi menggunakan MobileNetV2 sebagai feature extraction dan XGBoost sebagai classifier untuk membedakan jenis penyakit daun padi berdasarkan citra digital.

## Metode
1. Preprocessing citra: resize 224×224, normalisasi, augmentasi
2. Ekstraksi fitur dengan MobileNetV2 (pretrained ImageNet, tanpa top layer)
3. Klasifikasi dengan XGBoost + hyperparameter tuning
4. Evaluasi: Accuracy, Precision, Recall, Weighted F1-Score, Confusion Matrix
5. Perbandingan dengan baseline CNN murni

## Hasil 
Coming soon
