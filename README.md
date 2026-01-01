# 🎭 Comparative Study: Deepfake Detection using Sequential Architectures

[![Research Category: Riset](https://img.shields.io/badge/Category-Riset-blue.svg)](#)
[![Dataset: Celeb--DF_v2](https://img.shields.io/badge/Dataset-Celeb--DF_v2-green.svg)](#)

This repository contains a comprehensive research on Deepfake detection. We compare three different Deep Learning architectures to evaluate their effectiveness in capturing facial temporal anomalies.

## 👥 Authors (Team Chronos)
* **Faisal Nur Qolbi** - its.salll@upi.edu
* **Datuk Daneswara Raditya Samsura** - daneswara@upi.edu
* **Shizuka Maulia Putri** - mauliaputri23@upi.edu

---

## 📂 Research Notebooks

We conducted our experiments across three distinct architectures. Each notebook contains the full pipeline from feature loading to advanced evaluation.

### 1. Baseline Model (DNN)
The simplest approach using Fully Connected Layers. Used as a performance benchmark for more complex models.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_COLAB_DNN_ANDA)

### 2. Recurrent Model (BiLSTM)
A sequence-aware model that processes frames in both forward and backward directions to capture long-term dependencies.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_COLAB_BILSTM_ANDA)

### 3. Attention Model (Transformer Encoder)
Our primary research focus. It uses **Self-Attention** mechanisms to analyze all video frames simultaneously, providing the highest detection accuracy.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_COLAB_TRANSFORMER_ANDA)

---

## 🔄 Research Methodology

Our research follows a standardized pipeline across all models to ensure a fair comparison:



1.  **Face Cropping:** Using **MTCNN** to isolate facial regions and remove background noise.
2.  **Feature Embedding:** Extracting 1280-dimensional vectors per frame using **EfficientNet-B0**.
3.  **Sequential Processing:** Feeding the frame sequences into DNN, BiLSTM, or Transformer.
4.  **Inference:** Live testing using YouTube URLs and "In-the-wild" videos.

---

## 📊 Evaluation Results

| Model Architecture | Best AUC | Optimal Threshold | Accuracy |
| :--- | :---: | :---: | :---: |
| **DNN (Baseline)** | 0.8624 | 0.50 | 85.2% |
| **BiLSTM** | 0.9502 | 0.65 | **96.0%** |
| **Transformer** | **0.9565** | **0.13** | 94.6% |



### Key Finding:
The **Transformer Encoder** achieved the highest AUC (0.9565), proving that self-attention is highly effective at spotting subtle deepfake artifacts. However, we found that **BiLSTM** remains more stable when predicting "Real" videos with standard thresholds.

---

## 📜 Acknowledgments
This research is built upon the **Celeb-DF v2** dataset and utilizes pretrained weights from **EfficientNet-B0** and **MTCNN**.
