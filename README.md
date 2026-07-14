# 🌴 Date Fruit Classification using Artificial Neural Networks (PyTorch)

## 📌 Project Overview

This project implements a **Multiclass Classification Model** using an **Artificial Neural Network (ANN)** built with **PyTorch** to classify different varieties of Date Fruits.

The workflow includes:

- 📂 Data Loading
- 🧹 Data Preprocessing
- 🔖 Label Encoding
- 📏 Feature Scaling
- ✂️ Train-Test Split
- 🧠 ANN Model Development
- ⚡ Model Training
- 📈 Model Evaluation

---

## 📊 Dataset

Dataset used:

**Date Fruit Dataset**

Target Column:

```
Class
```

---

## 🚀 Technologies Used

- 🐍 Python
- 🔥 PyTorch
- 📊 Pandas
- 🔢 NumPy
- 🤖 Scikit-Learn

---

## 📁 Project Structure

```
date-fruit-classification-ann-pytorch/
│
├── ANN_Classification.py
├── DateFruit_Dataset.csv
├── README.md
└── requirements.txt
```

---

## 🧠 Neural Network Architecture

```
Input Layer
      │
      ▼
Dense (64)
      │
    ReLU
      │
      ▼
Dense (64)
      │
    ReLU
      │
      ▼
Output Layer (7 Classes)
```

---

## ⚙️ Installation

Clone the repository

Move into project directory

```bash
cd date-fruit-classification-ann-pytorch
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python ANN_Classification.py
```

---

## 📈 Workflow

```
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Label Encoding
   │
   ▼
Standard Scaling
   │
   ▼
Train/Test Split
   │
   ▼
PyTorch ANN
   │
   ▼
Training
   │
   ▼
Prediction
   │
   ▼
Accuracy Evaluation
```

---

## 📚 Libraries

```python
pandas
numpy
torch
scikit-learn
```

---

## 📊 Evaluation Metric

- ✅ Accuracy Score

---

## 🔥 Future Improvements

- Hyperparameter Tuning
- Dropout Regularization
- Early Stopping
- Confusion Matrix
- Precision, Recall & F1-Score
- Model Saving & Loading
- TensorBoard Visualization
- Learning Rate Scheduler

---
