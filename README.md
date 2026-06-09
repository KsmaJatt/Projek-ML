# 🌱 Plant Disease Classification using CNN, EfficientNetB0, and Hybrid EfficientNetB0-XGBoost

## 📌 Overview

Plant diseases are one of the major factors affecting agricultural productivity. Early and accurate disease detection can help farmers reduce crop losses and improve food security.

This project develops an automated plant disease classification system using leaf images from the PlantVillage Dataset. Three different approaches are compared:

* Convolutional Neural Network (CNN)
* EfficientNetB0 Transfer Learning
* Hybrid EfficientNetB0 + XGBoost

The objective is to identify the most effective model for multiclass plant disease classification and evaluate the impact of class imbalance handling techniques.

---

## 🎯 Objectives

This project aims to:

* Build a plant disease classification model using leaf images.
* Compare CNN, EfficientNetB0, and Hybrid EfficientNetB0-XGBoost.
* Implement transfer learning using ImageNet pretrained weights.
* Investigate the effect of class weighting on imbalanced datasets.
* Evaluate model performance using multiple classification metrics.

---

## 👥 Team Members

**Machine Learning Project - Group 3 (2024C)**

| Name                        | Student ID  |
| --------------------------- | ----------- |
| Chaesar Giveson             | 24031554058 |
| Sasmita Kusuma Jati         | 24031554052 |
| Nagatan Alief Putra Silahen | 24031554086 |

---

# 📊 Dataset

Dataset Source:

https://www.kaggle.com/datasets/emmarex/plantdisease

### Dataset Summary

| Information       | Value                      |
| ----------------- | -------------------------- |
| Total Images      | 20,638                     |
| Number of Classes | 15                         |
| Image Size        | 224 × 224                  |
| Color Format      | RGB                        |
| Task Type         | Multi-Class Classification |

### Example Classes

* Pepper__bell___Bacterial_spot
* Pepper__bell___healthy
* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy
* Tomato___healthy
* Tomato___Target_Spot
* Tomato___Spider_mites_Two_spotted_spider_mite

---

# 🧠 Methodology

## 1. Data Preprocessing

The following preprocessing steps were applied:

* Image resizing (224×224)
* Pixel normalization
* Label encoding
* Train-validation split (80:20)
* Data augmentation

### Data Augmentation

* Rotation
* Zoom
* Width Shift
* Height Shift
* Horizontal Flip

---

## 2. CNN Baseline Model

A simple CNN architecture was developed as the baseline model.

### Architecture

```text
Input Image (224×224×3)
        ↓
Conv2D (32) + ReLU
        ↓
MaxPooling2D
        ↓
Conv2D (64) + ReLU
        ↓
MaxPooling2D
        ↓
Conv2D (128) + ReLU
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dropout (0.5)
        ↓
Dense (256) + ReLU
        ↓
Softmax (15 Classes)
```

---

## 3. EfficientNetB0 Transfer Learning

EfficientNetB0 pretrained on ImageNet was used to leverage learned visual representations.

### Architecture

```text
EfficientNetB0 (include_top=False)
                ↓
GlobalAveragePooling2D
                ↓
Dropout (0.3)
                ↓
Dense (256) + ReLU
                ↓
Softmax Output
```

### Training Strategy

#### Phase 1

* Freeze all EfficientNetB0 layers
* Train classification head

#### Phase 2

* Unfreeze last 20 layers
* Fine-tuning

---

## 4. Hybrid EfficientNetB0-XGBoost

This approach combines deep feature extraction with gradient boosting classification.

### Pipeline

```text
Image
  ↓
EfficientNetB0
  ↓
Feature Vector
  ↓
XGBoost
  ↓
Prediction
```

### XGBoost Parameters

```python
n_estimators = 200
max_depth = 6
learning_rate = 0.1
eval_metric = "mlogloss"
```

---

## 5. Class Imbalance Handling

An additional experiment was conducted to evaluate class imbalance mitigation techniques.

### CNN & EfficientNetB0

```python
compute_class_weight("balanced")
```

### XGBoost

```python
compute_sample_weight("balanced")
```

---

# 🔄 Project Workflow

```text
PlantVillage Dataset
        ↓
Data Exploration
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
CNN Training
        ↓
EfficientNetB0 Training
        ↓
Feature Extraction
        ↓
XGBoost Classification
        ↓
Model Evaluation
        ↓
Performance Comparison
        ↓
Conclusion
```

---

# 📈 Experimental Results

## Imbalanced Dataset

| Model          | Accuracy | Weighted F1 | Macro F1 |
| -------------- | -------- | ----------- | -------- |
| EfficientNetB0 | 98.50%   | 0.98        | 0.99     |
| Hybrid XGBoost | 97.67%   | 0.98        | 0.97     |
| CNN            | 93.84%   | 0.94        | 0.93     |

---

## Balanced Dataset

| Model          | Accuracy | Weighted F1 | Macro F1 |
| -------------- | -------- | ----------- | -------- |
| Hybrid XGBoost | 95.15%   | 0.95        | 0.94     |
| EfficientNetB0 | 94.76%   | 0.95        | 0.95     |
| CNN            | 87.77%   | 0.88        | 0.85     |

---

## Accuracy Comparison

| Model          | Imbalanced | Balanced | Difference |
| -------------- | ---------- | -------- | ---------- |
| EfficientNetB0 | 98.50%     | 94.76%   | -3.74%     |
| Hybrid XGBoost | 97.67%     | 95.15%   | -2.52%     |
| CNN            | 93.84%     | 87.77%   | -6.07%     |

---

# 🔍 Key Findings

### Best Overall Model

**EfficientNetB0 (Imbalanced)**

* Accuracy: 98.50%
* Macro F1: 0.99

This model achieved the highest overall performance and demonstrated strong robustness against class imbalance.

### Hybrid Model Performance

The Hybrid EfficientNetB0-XGBoost model achieved the highest accuracy under the balanced setting and showed greater responsiveness to sample weighting.

### Impact of Class Weighting

Contrary to expectations, class weighting reduced performance across all models.

This suggests that transfer learning using EfficientNetB0 already provides highly discriminative feature representations, reducing the need for additional imbalance handling techniques.

### Difficult Classes

The most challenging classes across all experiments were:

* Tomato_Early_blight
* Tomato_Target_Spot

These diseases exhibit highly similar visual symptoms, making them difficult to distinguish.

---

# 📂 Project Structure

```text
📁 Projek-ML
│
├── notebook
│   ├── Projek_ML_imbalanced.ipynb
│   ├── Projek_ML_balanced1.ipynb
│   └── Projek_ML_balanced2.ipynb
│
├── model
│   ├── cnn_best.keras
│   ├── effnet_phase2_best.keras
│   ├── xgb_model.pkl
│   ├── cnn_balanced_best.keras
│   ├── effnet_phase2_balanced_best.keras
│   └── xgb_balanced_model.pkl
│
├── hasil
│   ├── confusion_matrix
│   ├── classification_report
│   └── training_curve
│
├── visualisasi
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

# 🛠️ Technologies Used

* TensorFlow
* Keras
* EfficientNetB0
* XGBoost
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

# 🚀 Installation

Clone repository:

```bash
git clone https://github.com/KsmaJatt/Projek-ML.git
cd Projek-ML
```

Install dependencies:

```bash
pip install tensorflow xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

# ▶️ Running Experiments

### Imbalanced Experiment

```bash
Projek_ML_imbalanced.ipynb
```

### Balanced Experiment

```bash
Projek_ML_balanced1.ipynb
Projek_ML_balanced2.ipynb
```

Run all notebook cells to:

* preprocess images
* train models
* evaluate performance
* generate visualizations

---

# 🔮 Future Work

Several improvements can be explored in future research:

* Real-world field image evaluation
* Grad-CAM explainability
* Focal Loss implementation
* Web/mobile deployment
* ResNet50 comparison
* MobileNetV3 comparison
* Vision Transformer (ViT) comparison

---

# 📚 Academic Purpose

This project was developed as part of the **Machine Learning Course** in the **Bachelor of Data Science Program, Universitas Negeri Surabaya (UNESA)**.
