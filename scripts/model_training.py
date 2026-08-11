import os
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
    RocCurveDisplay,
    confusion_matrix,
    auc
)
import shap

# Get the directory where model_training is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Join the paths
x_path = os.path.join(PROJECT_ROOT, 'outputs', 'extracted_motifs_X.csv')
gt_path = os.path.join(PROJECT_ROOT, 'data', 'otc_gt.csv')

# Load and prepare data
X_df = pd.read_csv(x_path)
gt_df = pd.read_csv(gt_path, names=['node', 'label'])

# Standardize labels: 1 = Fraud, 0 = Normal
gt_df['is_fraud'] = gt_df['label'].apply(lambda x: 1 if x == -1 else 0)
gt_df = gt_df.drop(columns=['label'])

model_data = pd.merge(X_df, gt_df, on='node', how='inner')
X = model_data[['out_burst', 'in_burst', 'ping_pong', 'repetition', 'convey', 'weakly_connected']]
y = model_data['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Hyperparameter tuning (GridSearchCV)
print("Initiating 5-Fold Grid Search Cross-Validation")

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
print(f"Optimal Parameters Found: {grid_search.best_params_}")


# Threshold Optimization
# Extract raw probabilities for the test set
y_probs = best_rf.predict_proba(X_test)[:, 1]

# Generate precision and recall arrays across all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# Calculate F1-Scores
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


# Final evaluation and visualization
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


# 4. Confusion Matrix Heatmap (Using Optimal Threshold)
cm = confusion_matrix(y_test, y_pred_optimal)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Fraud (1)'],
            yticklabels=['Normal (0)', 'Fraud (1)'])
plt.title(f'Random Forest Confusion Matrix\n(Optimized Threshold = {optimal_threshold:.2f})', fontsize=12)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300) # Saves the image for your presentation!
plt.show()


# 5. Precision-Recall (PR) Curve
pr_auc = auc(recalls, precisions)

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')

# Mark the optimal threshold point on the PR curve
plt.plot(recalls[optimal_idx], precisions[optimal_idx], marker='o', markersize=8,
         color='red', label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.xlabel('Recall (True Positive Rate)')
plt.ylabel('Precision (Positive Predictive Value)')
plt.title('Precision-Recall Curve for Temporal Motifs')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=300) # Saves the image for your presentation!
plt.show()


# 6. Probability Separation Distribution
plt.figure(figsize=(8, 5))
# Plot the distribution of predicted probabilities for the Normal class
sns.kdeplot(y_probs[y_test == 0], fill=True, color='blue', label='Normal (0)', alpha=0.5)
# Plot the distribution of predicted probabilities for the Fraud class
sns.kdeplot(y_probs[y_test == 1], fill=True, color='red', label='Fraud (1)', alpha=0.5)

# Draw the optimized decision boundary
plt.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
            label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.title('Predicted Probability Distribution by Class', fontsize=14)
plt.xlabel('Predicted Probability of Fraud')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('probability_distribution.png', dpi=300)
plt.show()


# 7. SHAP Value Summary Plot (Directional Impact)
print("\nCalculating SHAP values")

explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values_fraud = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values_fraud = shap_values[:, :, 1]
else:
    shap_values_fraud = shap_values

plt.figure(figsize=(8, 6))
plt.title('SHAP Summary: Motif Impact on Fraud Classification', fontsize=14)

# Generate the summary plot
shap.summary_plot(shap_values_fraud, X_test, show=False)

plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()