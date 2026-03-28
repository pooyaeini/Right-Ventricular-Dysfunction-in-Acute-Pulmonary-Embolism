Machine Learning Pipeline for Predicting Right Ventricular Dysfunction in Acute Pulmonary Embolism
A complete end‑to‑end machine learning workflow for preprocessing, resampling, model training, evaluation, visualization, and feature analysis in a clinical dataset of pulmonary embolism patients.

📌 Project Overview
This repository contains a fully implemented Python pipeline for predicting right ventricular dysfunction (RVD) in acute pulmonary embolism using multiple machine learning classifiers. The workflow includes:

Data cleaning, preprocessing, and feature scaling
Handling class imbalance with SMOTE
Training six ML models
Full evaluation using ROC, PR, cross‑validation, confusion matrices
Feature importance analysis (tree‑based + coefficients)
Calibration curves
Cox proportional hazards modeling
Automatic export of performance tables and visualizations
The project serves as a reproducible research tool for healthcare professionals, ML engineers, and clinical researchers.

🧬 Dataset
The script uses a dataset named:

text
Pulmonary_Embolism_data_set.csv
Expected structure:

Rows: patient cases
Columns: clinical or paraclinical features
Final column: binary target variable (RVD = 1, No RVD = 0)
Missing numeric values are automatically imputed with column medians.

Two new files are generated:

text
train_set.csv  
test_set.csv

🧠 Machine Learning Models Implemented
The code evaluates a comprehensive suite of commonly used ML classifiers:

Logistic Regression
Random Forest
Support Vector Machine (SVM, probability enabled)
Decision Tree
XGBoost
Neural Network (MLPClassifier, 64‑32 hidden layers)
All models are:

Trained on SMOTE‑balanced training data
Evaluated on the original imbalanced test set
Assessed with 5‑fold Stratified K‑Fold cross‑validation
⚙️ Pipeline Summary
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
Metrics include:

AUC
Sensitivity
Specificity
Precision
Confusion matrix (TP, TN, FP, FN)
3. Model Evaluation Outputs
The script saves:

Confusion matrices (confusion_<model>.png)
Feature importances (tree‑based models)
ROC curves for all models (roc_curves.png)
Precision‑Recall curves (pr_curve.png)
Calibration curve for the best model
Correlation heatmap (heatmap.png)
A full metrics table (model_performance.csv)
Additionally:

Best model selected based on highest test AUC
Top 10 features printed (via coefficients or importances)

📊 Cox Regression Analysis
The script includes a Cox proportional hazards model using:

Dummy duration = 1
Outcome = target column
It prints the top 10 most significant predictors ranked by p‑value, offering additional clinical insight beyond ML.

📈 Visualizations Generated
The following plots are automatically saved:

ROC curves for all models
Precision–Recall curves for all models
Confusion matrix for each classifier
Feature importance bar plots (tree models)
Full correlation heatmap
Calibration curve for the best model
These figures appear in the working directory as PNG files.

📁 File Structure
text
├── Pulmonary_Embolism_data_set.csv
├── train_set.csv
├── test_set.csv
├── confusion_<model>.png
├── feature_importance_<model>.png
├── heatmap.png
├── roc_curves.png
├── pr_curve.png
├── calibration_curve.png
├── model_performance.csv
├── your_script.py  (or whatever filename you choose)
└── README.md
📦 Requirements
Install dependencies:

text
pip install pandas numpy scikit-learn xgboost seaborn lifelines imbalanced-learn matplotlib
▶️ Usage
Run the script with:

text
python your_script.py
Outputs will be generated automatically in the current folder.

🧪 Reproducibility Notes
All random processes use random_state=42
Stratification ensures identical class distribution in train/test
SMOTE is applied only to training data
Cross‑validation is performed on SMOTE data
Evaluation on test set uses original class imbalance (realistic clinical setting)

📜 Citation
If you use this code in academic work, please cite this repository and acknowledge the pipeline’s contribution to your analysis.

🤝 Contributions
Feel free to open issues, report bugs, or submit pull requests to improve model performance or extend the pipeline (e.g., SHAP values, hyperparameter tuning, additional models).
