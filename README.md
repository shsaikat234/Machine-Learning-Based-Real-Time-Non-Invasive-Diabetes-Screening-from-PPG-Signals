# PPG-Based Diabetes Detection using Machine Learning

This project focuses on detecting **diabetes from Photoplethysmography (PPG) signals** using multiple machine learning algorithms. It applies signal preprocessing, feature extraction, and model training to classify whether a subject is diabetic or non-diabetic — aiming for a **non-invasive, low-cost diagnostic approach**.

---

## Dataset

The dataset used in this study is the **Mazandaran PPG Diabetes Dataset**, collected from clinical sources in Mazandaran, Iran.  
It contains PPG signal recordings from both diabetic and healthy individuals.

**Dataset link:** [Mazandaran PPG Dataset](https://data.mendeley.com/datasets/37pm7jk7jn/2)

---

## Features Used

From each PPG waveform, several **time-domain** and **frequency-domain** features were extracted, such as:

- Mean, Standard Deviation, Variance  
- Skewness, Kurtosis  
- Heart Rate (HR)  
- Peak-to-Peak Interval  
- Signal Entropy  
- Energy and Power Spectral Density (PSD)

These features were used to train multiple machine learning classifiers.

---

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

1. **Data Loading and Cleaning** – PPG data loaded from Mazandaran dataset.  
2. **Feature Extraction** – Computed key time/frequency domain features.  
3. **Data Normalization** – Scaled features using Min-Max or Standard Scaler.  
4. **Model Training** – Tested multiple ML classifiers (SVM, RF, XGB, etc.).  
5. **Performance Evaluation** – Accuracy, confusion matrix, ROC curves.  
6. **Result Visualization** – Compared overfitting and XGBoost performance.  

---
