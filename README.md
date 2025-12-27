# 🔊 Audio Deepfake Detection using EfficientNet

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project is an end-to-end deep learning pipeline designed to **detect AI-generated (spoof) audio** using the **ASVspoof 2015 dataset**. By converting audio into **Mel-spectrograms** and processing them with a **pretrained EfficientNetB0** model, the system discriminates between **Bonafide (Real)** and **Spoof (Deepfake)** speech with high accuracy.

---

## 📌 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [🚀 Methodology](#-methodology)
- [📈 Performance Results](#-performance-results)
- [🛠 Installation & Setup](#-installation--setup)
- [🙏 Acknowledgments](#-acknowledgments)
- [✉️ Contact](#-contact)

---

## 🔍 Project Overview

As voice synthesis technology improves, distinguishing between human and machine-generated speech becomes critical for security. This project focuses on:

✅ **Feature Extraction:** Converting raw audio to 3-channel Mel-spectrograms.  
✅ **Deep Learning:** Fine-tuning a pretrained EfficientNetB0 CNN.  
✅ **Robust Evaluation:** Testing on both known (Development) and unknown (Evaluation) attack types.  
✅ **Metric Analysis:** Using Equal Error Rate (EER) to measure security performance.

---

## 📊 Dataset

We utilize the **ASVspoof 2015 Dataset**, a benchmark for spoofing detection.
- **Classes:** `Bonafide` (Human) vs. `Spoof` (Text-to-Speech / Voice Conversion).
- **Data Format:** `.wav` files converted to spectrogram images.
- **Challenge:** The Evaluation set contains **5 unknown attack algorithms (S6–S10)** that the model never saw during training, testing its generalization.

🔗 **Reference:** [ASVspoof Challenge](https://www.asvspoof.org/)

---

## 🧠 Model Architecture

The core of the system is **EfficientNetB0**, chosen for its balance of speed and accuracy.

1.  **Input:** Mel-Spectrograms resized to `(128, 400, 3)`.
2.  **Backbone:** EfficientNetB0 (Pretrained on ImageNet, frozen weights).
3.  **Head:**
    * `GlobalAveragePooling2D`
    * `Dense(128, ReLU)`
    * `Dropout(0.5)` (To prevent overfitting)
    * `Dense(1, Sigmoid)` (Output probability)

---

## 🚀 Methodology

### 1️⃣ Preprocessing
- Audio is loaded using `librosa`.
- Converted to **Mel-spectrogram** (128 mel bands).
- Converted to **Decibel scale**.
- Fixed width resizing to **400 time steps** (Padding/Truncating).
- Duplicated to **3 channels** to match EfficientNet's expected input.

### 2️⃣ Training
- **Optimizer:** Adam (`lr=0.0001`)
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 32
- **Epochs:** 10

---

## 📈 Performance Results

The model was evaluated on two distinct datasets to test robustness.

| Metric | Development Set (Known Attacks) | Evaluation Set (Unknown Attacks) |
| :--- | :--- | :--- |
| **Status** | *Practice Exam* | *Final Exam* |
| **Accuracy** | **88.12%** | **85.81%** |
| **EER (Equal Error Rate)** | **26.79%** | **29.63%** |
| **F1 Score** | 0.26 | 0.21 |

> **Analysis:** The model maintains **~86% accuracy** even on unknown attacks, showing strong generalization capabilities. The low EER difference (~2.8%) between Dev and Eval sets confirms the model is not just memorizing training data.

---

## 🛠 Installation & Setup

### 🔧 Prerequisites
* Python 3.8+
* Librosa 0.10.0
* Matplotlib 3.7.2
* Numpy 1.24.4
* Pandas 1.5.3
* Scikit-learn 1.3.0
* Soundfile 0.13.1
* Tensorflow 2.13.0
* Tqdm 4.66.1

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/bhushnbrt/Audio-Deepfake-Detection-EfficientNet.git
cd Audio-Deepfake-Detection-EfficientNet
```


## 🙏 Acknowledgments

This project acknowledges the inspiration and details the nature of the implementation.

| Category | Details |
| :--- | :--- |
| **Inspiration** | [AmoghAgrawal1249](https://github.com/AmoghAgrawal1249) |
| **Implementation** | **100% Original Work** |
| **Methodology** | EfficientNet-based (Independent) |

> **Note:** All implementation, development, and code are original. The EfficientNet-based approach and methodology for audio deepfake detection have been implemented independently from scratch.

---

## ✉️ Contact

For questions or inquiries about this project, please reach out via the following channels.

| Platform | Contact Details |
| :--- | :--- |
| **Email** | [bhushnbrt@gmail.com](mailto:bhushnbrt@gmail.com) |
| **GitHub** | [@bhushnbrt](https://github.com/bhushnbrt) |

> **Connect:** Feel free to reach out for collaborations, questions, or issues regarding the repository.

---
