# Klasifikasi Penyakit Daun Padi dengan Hybrid CNN-XGBoost
Proyek ini mengklasifikasikan penyakit daun padi (Bacterial Blight, Blast,Brown Spot, Healthy) dari citra menggunakan pendekatan hybrid: MobileNetV2 sebagai feature extractor dan XGBoost sebagai classifier.

## Dataset 
[Rice Leaf Disease Image Dataset — Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image)

## Metode
1. Preprocessing citra: resize 224×224, normalisasi, augmentasi
2. Ekstraksi fitur dengan MobileNetV2 (pretrained ImageNet, tanpa top layer)
3. Klasifikasi dengan XGBoost + hyperparameter tuning
4. Evaluasi: Accuracy, Precision, Recall, Weighted F1-Score, Confusion Matrix
5. Perbandingan dengan baseline CNN murni

## Hasil 
Coming soon
