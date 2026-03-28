# Machine Learning–Based Prediction of Right Ventricular Dysfunction in Acute Pulmonary Embolism

A complete, reproducible machine learning pipeline for predicting **Right Ventricular Dysfunction (RVD)** in **Acute Pulmonary Embolism** using multiple machine learning models.  
Includes full preprocessing, resampling, training, evaluation, and visualization.

---

## 📌 Project Overview

This repository implements an end-to-end workflow for:

- Data preprocessing and feature scaling  
- Class imbalance handling using **SMOTE**  
- Training and evaluating **six ML models**  
- Model comparison via ROC, PR, and cross-validation  
- Feature importance and correlation analysis  
- Cox regression for clinical risk factors  

The goal is to identify reliable predictors of RVD using open-source tools.

---

## 🧬 Dataset

Expected file: `Pulmonary_Embolism_data_set.csv`

Each row = patient record  
Each column = clinical or paraclinical feature  
Final column = binary target (RVD = 1, no-RVD = 0)

During execution, the script also generates:
train_set.csv
test_set.csv

---

## 🧠 Machine Learning Models Implemented

- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- Decision Tree  
- XGBoost  
- Neural Network (MLPClassifier)


---

## ⚙️ Pipeline Summary
1. Preprocessing
Median imputation
Train/test split (80/20, stratified)
StandardScaler applied to all features
SMOTE applied only to training data

2. Model Training
Each model is:
Trained on SMOTE‑resampled data
Evaluated with:

Training metrics (original scale)

Test metrics

5‑fold CV metrics

Metrics computed for each model:

| Metric | Description |
| ------- | ------------ |
| AUC | Area Under ROC Curve |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| Precision | Positive predictive value |
| Confusion Matrix | TP, TN, FP, FN counts |

---

## 📈 Visualizations Generated

Key outputs automatically saved in the working directory:
confusion_<model>.png

feature_importance_<model>.png

roc_curves.png

pr_curve.png

heatmap.png

calibration_curve.png

model_performance.csv


---

## 📊 Cox Regression Analysis

A Cox proportional hazards model is included to identify independent predictors of RVD.  
Important covariates are ranked by significance (p-values), providing interpretability alongside ML results.

---

## 📦 Installation and Requirements

Clone the repository and install the dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```
▶️ Usage
Run the analysis:
```bash
python analysis.py
```

All plots and CSV outputs will appear in the project directory.

Reproducibility Details
Random seed fixed (random_state=42)
Stratified train/test split (80/20)
SMOTE applied only to training data
Cross-validation performed on SMOTE data
Test set evaluation uses original class proportions
## 📁 File Structure
```
├── analysis.py
├── Pulmonary_Embolism_data_set.csv
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
└── (generated outputs)
```
## License
This project is licensed under the MIT License – see LICENSE for details.

## 📜 Citation
If you use this repository for academic work, please cite:

[Pooya Eini]. Machine Learning–Based Prediction of Right Ventricular Dysfunction in Acute Pulmonary Embolism. 2026. 
GitHub repository:github.com/pooyaeini/Right-Ventricular-Dysfunction-in-Acute-Pulmonary-Embolism

## 🤝 Contributions
Pull requests are welcome!

Open an issue to suggest improvements or new models.
