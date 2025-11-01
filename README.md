# PPG-Based Diabetes Detection using Machine Learning

This project focuses on detecting **diabetes from Photoplethysmography (PPG) signals** using multiple machine learning algorithms. It applies **signal preprocessing, feature extraction, and model training** to classify whether a subject is diabetic or non-diabetic — aiming for a **non-invasive, low-cost diagnostic approach**.

Beyond offline model development, this project also includes **software integration and real-time model deployment**. The trained machine learning model (XGBoost) was integrated with a **data acquisition system** built using the **MAX30102 PPG sensor** and an **ESP32 microcontroller**.  

The ESP32 captures live PPG signals (infrared and red light waveforms) from the MAX30102 sensor and transmits them to a **Python-based interface** running on a PC. The software then preprocesses the real-time data, extracts relevant features, and uses the **deployed XGBoost model** to predict whether the signal indicates a diabetic or non-diabetic condition.  

This enables a complete **end-to-end smart diagnostic system** — from **hardware signal acquisition** to **machine learning inference** — demonstrating a practical, IoT-based approach for non-invasive diabetes screening.


---

## Dataset

Non-invasive monitoring and surveillance methods of blood glucose measurement provide **ease of use** and **reduced risk** compared to traditional invasive methods. One of the most promising techniques is based on **Photoplethysmography (PPG)** signals, which have been widely studied for glucose level estimation.

The dataset used in this study is the **Mazandaran PPG Diabetes Dataset**, developed by the **Digital Systems Research Group** of the **University of Science and Technology of Mazandaran, Behshahr, Iran**.  

This publicly available dataset includes:
- **67 raw PPG signal recordings**
- **Sampling frequency:** 2175 Hz  
- **Labeled attributes:** age, gender, and invasively measured blood glucose level  

These recordings were collected from both **diabetic and non-diabetic subjects**, making the dataset suitable for training and evaluating machine learning algorithms for non-invasive diabetes detection.

**Dataset link:** [Mazandaran PPG Dataset](https://data.mendeley.com/datasets/37pm7jk7jn/2)

---

## Features Used

## 🧩 Features Used

From each PPG waveform and patient metadata, a total of **10 features** were extracted — **7 PPG-derived** and **3 demographic**.  
These features capture both the **morphological patterns** of the PPG signal and **subject-specific physiological information**, improving model reliability and interpretability.

### 🩸 PPG-Derived Features (7)
1. **Skewness** – Measures asymmetry of the signal distribution  
2. **Kurtosis** – Indicates the heaviness of signal tails  
3. **Mean** – Average amplitude of the PPG waveform  
4. **Standard Deviation (STD)** – Quantifies signal variability  
5. **Mean Absolute Deviation (MAD)** – Robust measure of variability, less sensitive to outliers  
6. **Variance** – Squared standard deviation (overall dispersion)  
7. **Root Mean Square (RMS)** – Reflects the signal’s power content  

### 👤 Demographic Features (3)
8. **Age** – Patient’s age (in years)  
9. **Body Mass Index (BMI)** – Calculated as weight (kg) / height² (m²)  
10. **Gender** – Encoded as binary (0 = Male, 1 = Female)

These **10 combined features** were used as inputs to train multiple machine learning classifiers, including **Logistic Regression**, **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **XGBoost**.


## Machine Learning Models & Results

| Model | Training Accuracy | Testing Accuracy | Observation |
|--------|--------------------|------------------|--------------|
| Logistic Regression | 100% | 71% | Overfitted |
| Random Forest | 100% | 73% | Overfitted |
| SVM (RBF Kernel) | 100% | 70% | Overfitted |
| **XGBoost** | **96%** | **86%** | Best-performing model |
| KNN | 99% | 74% | Slightly overfitted |

**XGBoost achieved the best generalization performance** and balanced bias-variance tradeoff, making it the optimal model for this dataset.  
Other models achieved high training accuracy but failed to generalize well on unseen data, indicating overfitting.

---

## Project Workflow

1. **Data Loading and Cleaning** – PPG data loaded from the Mazandaran dataset.  
2. **Feature Extraction** – Computed key time and frequency domain features.  
3. **Data Normalization** – Scaled features using Min-Max or Standard Scaler.  
4. **Model Training** – Trained and compared SVM, RF, XGBoost, KNN, and Logistic Regression.  
5. **Performance Evaluation** – Calculated accuracy, confusion matrix, and ROC curves.  
6. **Result Visualization** – Compared overfitting behavior and visualized XGBoost’s performance.  
7. **Software Integration & Hardware Data Acquisition** –  
   - Integrated real-time PPG signal acquisition using the **MAX30102 pulse oximeter sensor** connected to an **ESP32 microcontroller**.  
   - The ESP32 collects live PPG data (IR and Red LED signals) and transmits it via serial to the **PC interface** for further analysis.  
   - The real-time data is processed to extract features similar to those used in the Mazandaran dataset, allowing **on-device diabetes risk estimation** using the trained XGBoost model.  

---

## Hardware Integration Overview

- **Sensor:** MAX30102 (IR + Red LED for PPG signal acquisition)  
- **Microcontroller:** ESP32 
- **Communication Interface:** Serial (USB) 
- **Software Environment:**  
  - Arduino IDE (for ESP32 firmware)  
  - Python / Tkinter GUI (for data logging, visualization, and ML-based classification)  

This integration demonstrates the **real-world application** of the trained model — transforming it into a **smart IoT-based non-invasive diabetes detection system**.

---
