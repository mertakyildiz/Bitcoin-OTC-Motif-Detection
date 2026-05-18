import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    RocCurveDisplay
)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
X_df = pd.read_csv('extracted_motifs_X.csv')
gt_df = pd.read_csv('otc_gt.csv', names=['node', 'label'])

# Standardize labels: 1 = Fraud, 0 = Normal
gt_df['is_fraud'] = gt_df['label'].apply(lambda x: 1 if x == -1 else 0)
gt_df = gt_df.drop(columns=['label'])

model_data = pd.merge(X_df, gt_df, on='node', how='inner')
X = model_data[['out_burst', 'in_burst', 'ping_pong', 'repetition', 'convey', 'weakly_connected']]
y = model_data['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==========================================
# 2. HYPERPARAMETER TUNING (GridSearchCV)
# ==========================================
print("\nInitiating 5-Fold Grid Search Cross-Validation...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"\nOptimal Parameters Found: {grid_search.best_params_}")

# ==========================================
# 3. THRESHOLD OPTIMIZATION
# ==========================================
# Extract raw probabilities for the test set
y_probs = best_rf.predict_proba(X_test)[:, 1]

# Generate precision and recall arrays across all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Calculate F1-Scores safely
with np.errstate(divide='ignore', invalid='ignore'):
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)

# Identify the mathematical optimum
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
peak_f1 = f1_scores[optimal_idx]

print(f"\n=== THRESHOLD OPTIMIZATION ===")
print(f"Algorithmically Optimal Threshold: {optimal_threshold:.4f}")
print(f"Projected Peak F1-Score: {peak_f1:.4f}\n")

# ==========================================
# 4. FINAL EVALUATION & VISUALIZATION
# ==========================================
# Apply the optimal threshold
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# 1. Classification Report
print(f"--- FINAL CLASSIFICATION REPORT (Threshold {optimal_threshold:.2f}) ---")
print(classification_report(y_test, y_pred_optimal, target_names=['Normal (0)', 'Fraud (1)']))

# 2. ROC AUC Score & Plot
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC AUC Score: {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_probs, ax=ax, name="Random Forest", color="darkorange")
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
plt.legend()
plt.tight_layout()
plt.show()

# 3. Feature Importance Plot
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importances, hue='Feature', palette='magma', legend=False)
plt.title("Tuned Feature Importance (6-Letter Motif Alphabet)")
plt.xlabel("Predictive Weight")
plt.tight_layout()
plt.show()