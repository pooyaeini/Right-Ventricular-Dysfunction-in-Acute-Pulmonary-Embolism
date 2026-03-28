import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from lifelines import CoxPHFitter
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os

# Load data
file = 'Pulmonary_Embolism_data_set.csv'
df = pd.read_csv(file)

# Handle missing values (if any)
df = df.fillna(df.median(numeric_only=True))

# Identify target and features
target_col = df.columns[-1]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save train and test sets
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)
train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
}

# Metrics function
def get_metrics(y_true, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_pred_proba)
    return auc, sensitivity, specificity, precision, tp, tn, fp, fn

results = {}
pr_curves = {}
conf_matrices = {}
feature_importances = {}
all_test_probas = {}

for name, model in models.items():
    # Train on SMOTE data
    model.fit(X_train_res, y_train_res)
    # Cross-validation on SMOTE data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_metrics = []
    for train_idx, val_idx in skf.split(X_train_res, y_train_res):
        X_tr, X_val = X_train_res[train_idx], X_train_res[val_idx]
        y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]
        if name == 'Logistic Regression':
            model_cv = LogisticRegression(max_iter=1000)
        elif name == 'SVM':
            model_cv = SVC(probability=True)
        elif name == 'XGBoost':
            model_cv = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif name == 'Neural Network':
            model_cv = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        else:
            model_cv = type(model)()
        model_cv.fit(X_tr, y_tr)
        y_val_pred = model_cv.predict(X_val)
        y_val_pred_proba = model_cv.predict_proba(X_val)[:, 1]
        cv_metrics.append(get_metrics(y_val, y_val_pred, y_val_pred_proba))
    cv_metrics = np.array(cv_metrics)
    val_auc, val_sens, val_spec, val_prec, val_tp, val_tn, val_fp, val_fn = cv_metrics.mean(axis=0)
    # Test set metrics
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_auc, test_sens, test_spec, test_prec, test_tp, test_tn, test_fp, test_fn = get_metrics(y_test, y_pred, y_pred_proba)
    # Train set metrics (on original, not SMOTE, train set)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    train_auc, train_sens, train_spec, train_prec, train_tp, train_tn, train_fp, train_fn = get_metrics(y_train, y_pred_train, y_pred_proba_train)
    results[name] = {
        'Test AUC': test_auc,
        'Train AUC': train_auc,
        'Test Sensitivity': test_sens,
        'Train Sensitivity': train_sens,
        'Test Specificity': test_spec,
        'Train Specificity': train_spec,
        'Test Precision': test_prec,
        'Train Precision': train_prec,
        'Test TP': test_tp,
        'Train TP': train_tp,
        'Test TN': test_tn,
        'Train TN': train_tn,
        'Test FP': test_fp,
        'Train FP': train_fp,
        'Test FN': test_fn,
        'Train FN': train_fn,
        'Val AUC': val_auc,
        'Val Sensitivity': val_sens,
        'Val Specificity': val_spec,
        'Val Precision': val_prec,
        'Val TP': val_tp,
        'Val TN': val_tn,
        'Val FP': val_fp,
        'Val FN': val_fn
    }
    # Store for PR and calibration
    all_test_probas[name] = y_pred_proba
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_{name.replace(" ", "_").lower()}.png')
    plt.close()
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_curves[name] = (precision, recall, average_precision_score(y_test, y_pred_proba))
    # Feature Importance (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances[name] = importances
        plt.figure(figsize=(8, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.title(f'Feature Importance - {name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{name.replace(" ", "_").lower()}.png')
        plt.close()

# Results table
perf_df = pd.DataFrame(results).T

# Cox regression
cph = CoxPHFitter()
df_cox = df.copy()
df_cox['duration'] = 1  # No time-to-event data
cph.fit(df_cox, duration_col='duration', event_col=target_col)
cox_summary = cph.summary.sort_values('p', ascending=True)
print("\nTop 10 Most Important Features (Cox Regression):")
print(cox_summary.head(10)[['coef', 'exp(coef)', 'p']])

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# ROC curves
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_pred_proba = all_test_probas[name]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["Test AUC"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.savefig('roc_curves.png')
plt.show()

# Precision-Recall Curves (all models)
plt.figure(figsize=(10, 6))
for name, (precision, recall, avg_prec) in pr_curves.items():
    plt.plot(recall, precision, label=f'{name} (AP = {avg_prec:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend()
plt.tight_layout()
plt.savefig('pr_curve.png')
plt.show()

# Calibration Curve for best model (highest Test AUC)
best_model_name = perf_df['Test AUC'].idxmax()
best_model = models[best_model_name]
y_pred_proba_best = all_test_probas[best_model_name]
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_best, n_bins=10)
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title(f'Calibration Curve - {best_model_name}')
plt.legend()
plt.tight_layout()
plt.savefig('calibration_curve.png')
plt.show()

# Results table
print("\nModel Performance Metrics (Train, Test, and Validation):")
print(perf_df.round(3))
perf_df.to_csv('model_performance.csv')

# Print top 10 most important features for the best model
best_model_name = perf_df['Test AUC'].idxmax()
print(f"\nTop 10 Most Important Features for Best Model ({best_model_name}):")
best_model = models[best_model_name]
if hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
    feature_names = X.columns
    top_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[top_idx[i]]}: {importances[top_idx[i]]}")
elif hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    top_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        print(f"{feature_names[top_idx[i]]}: {importances[top_idx[i]]}")
else:
    print("Feature importance not available for this model.") 
