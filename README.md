# 🏥 NeuroDiagnose: AI-Powered Health Diagnostic Dashboard

**NeuroDiagnose** is a multi-modal medical diagnostic application built with Python and Streamlit. It leverages various Machine Learning and Deep Learning architectures to provide preliminary health assessments across three distinct medical domains: Endocrinology, Pulmonology, and Neurology.

---

## 🚀 Live Demo
**[https://neurodiagnose.streamlit.app/]**

---

## 🧠 Key Modules & Architectures

### 1. Endocrinology (Diabetes Prediction)
* **Goal:** Predict the likelihood of diabetes based on diagnostic metrics (Glucose, BMI, Age, etc.).
* **Model:** Support Vector Machine (SVM) with a Linear Kernel.
* **Performance:** ~78% Accuracy.
* **Data Source:** PIMA Indians Diabetes Dataset.

### 2. Pulmonology (Pneumonia Detection)
* **Goal:** Detect presence of viral/bacterial pneumonia in Chest X-Rays.
* **Model:** Custom Convolutional Neural Network (CNN).
* **Features:**
    * Binary Classification (Normal vs. Pneumonia).
    * Grayscale image processing.
* **Performance:** ~91% Accuracy.

### 3. Neurology (Brain Tumor Classification) ⚠️ *Experimental*
* **Goal:** Classify MRI scans into 4 categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
* **Model:** **ResNet50** (Transfer Learning) with custom top layers.
* **Advanced Engineering:**
    * **Test-Time Augmentation (TTA):** Implements a voting system (Original + Flipped + Zoomed) to improve prediction reliability during inference.
    * **Lambda Preprocessing:** Incorporates ResNet-specific preprocessing directly into the model architecture.
* **Status:** Beta/R&D (Current Accuracy: ~75%).

---

## 🛠️ Tech Stack
* **Interface:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-Learn
* **Image Processing:** PIL (Python Imaging Library), NumPy
* **Version Control:** Git & GitHub

## ⚠️ Disclaimer
**This project is for educational and research purposes only.**
The models deployed in this application are prototypes and should **NOT** be used for real medical diagnosis. Always consult a certified medical professional for health concerns.
