# 🎭 Comparative Study: Deepfake Detection using Sequential Architectures

[![Research Category: Riset](https://img.shields.io/badge/Category-Riset-blue.svg)](#)
[![Dataset: Celeb--DF_v2](https://img.shields.io/badge/Dataset-Celeb--DF_v2-green.svg)](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2/)

This repository contains comprehensive research on Deepfake detection. We compare three different Deep Learning architectures to evaluate their effectiveness in capturing facial temporal anomalies using the Celeb-DF v2 dataset.

## 👥 Authors (Team Chronos)
* **Faisal Nur Qolbi** - its.salll@upi.edu
* **Datuk Daneswara Raditya Samsura** - daneswara@upi.edu
* **Shizuka Maulia Putri** - mauliaputri23@upi.edu

---

## 📂 Research Notebooks

Each notebook contains the full pipeline: Feature Loading → Model Training → Optimal Threshold Finding → Advanced Evaluation.

### 1. Baseline Model (DNN)
The simplest approach using Fully Connected Layers to process averaged frame features.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WJKhVMJg9HtR80gNaQF3FStKiWFxE3HG?usp=sharing)

### 2. Recurrent Model (BiLSTM)
A bidirectional sequence-aware model designed to capture long-term temporal dependencies in facial movements.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/169VFECZgXPJ45t0oViY6CWwnPBayJwW7?usp=sharing)

### 3. Attention Model (Transformer Encoder)
Utilizes **Self-Attention** mechanisms to analyze all video frames simultaneously, focusing on the most suspicious temporal artifacts.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QwuUlVUDLSbK-GJEYuQJBPICVXI8VdJN?usp=sharing)

---

## 🔄 Research Methodology

Our research follows a standardized pipeline across all models to ensure a fair comparison:



1. **Face Cropping:** Utilizing **MTCNN** to isolate facial regions and eliminate background noise.
2. **Feature Embedding:** Extracting 1280-dimensional vectors per frame using **EfficientNet-B0**.
3. **Sequential Processing:** Feeding the 16-frame sequences into DNN, BiLSTM, or Transformer.
4. **Inference:** Live testing using YouTube URLs and "In-the-wild" video analysis.

---

## 📊 Evaluation Results (Final Experiment)

Based on our final research run, here is the performance comparison:

| Model Architecture | Best AUC | Optimal Threshold | Accuracy | F1-Score (Fake) |
| :--- | :---: | :---: | :---: | :---: |
| **DNN (Baseline)** | 0.9407 | 0.50 | 94.0% | 0.96 |
| **BiLSTM** | **0.9618** | **0.8811** | **96.0%** | **0.97** |
| **Transformer** | 0.9565 | 0.13 | 94.6% | **0.97** |



### 💡 Key Findings:
* **BiLSTM Excellence:** The **BiLSTM** model achieved the highest AUC (0.9618) and overall Accuracy (96%). It requires a high optimal threshold (**0.8811**) to maintain a balance between identifying real videos and detecting fakes.
* **Transformer Sensitivity:** The **Transformer Encoder** is highly effective at class separation (AUC 0.9565) but operates optimally at a low threshold (**0.13**).
* **Preprocessing Impact:** We found that bypassing standard ImageNet normalization is crucial for "in-the-wild" testing to match the training data distribution.

---

## 📜 Acknowledgments
This research is built upon the **Celeb-DF v2** dataset and utilizes pretrained weights from **EfficientNet-B0** and **MTCNN**. Special thanks to the open-source community for the tools and frameworks that made this study possible.
