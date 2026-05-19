import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score

# 1. Load data
network_df = pd.read_csv('data/soc-sign-bitcoinotc.csv', names=['source', 'target', 'rating', 'time'])
gt_df = pd.read_csv('data/otc_gt.csv', names=['node', 'label'])
gt_df['is_fraud'] = gt_df['label'].apply(lambda x: 1 if x == -1 else 0)

# Load the motifs-only features for the 316 users
X_motifs_df = pd.read_csv('outputs/extracted_motifs_X.csv')

# 2. Calculate simple features for the 316 users
simple_features = []
for user in X_motifs_df['node']:
    direct_edges = network_df[(network_df['source'] == user) | (network_df['target'] == user)]
    s_u = len(direct_edges)
    s_u_out = len(direct_edges[direct_edges['source'] == user])
    out_ratio = (s_u_out / s_u) if s_u > 0 else 0.0
    neighbors = set(direct_edges['source']).union(set(direct_edges['target']))
    k_u = len(neighbors - {user})
    simple_features.append({'node': user, 's_u': s_u, 'out_ratio': out_ratio, 'k_u': k_u})

X_simple_df = pd.DataFrame(simple_features)

# 3. Merge
model_data = pd.merge(X_motifs_df, gt_df[['node', 'is_fraud']], on='node')
model_data = pd.merge(model_data, X_simple_df, on='node')

y = model_data['is_fraud']
motifs_cols = ['out_burst', 'in_burst', 'ping_pong', 'repetition', 'convey', 'weakly_connected']
combined_cols = motifs_cols + ['s_u', 'out_ratio', 'k_u']


# 4. Evaluate function
def evaluate(features_list, name):
    X = model_data[features_list]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_probs = rf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.nan_to_num(2 * (precisions * recalls) / (precisions + recalls))
    opt_idx = np.argmax(f1_scores)

    return {'Model': name, 'ROC AUC': roc_auc, 'Peak F1': f1_scores[opt_idx]}


# --- Run Evaluation and Capture Results ---
results_df = pd.DataFrame([
    evaluate(motifs_cols, "Motifs Only"),
    evaluate(combined_cols, "Motifs + Simple Graph")
])

print("--- Feature fusion study ---")
print(results_df)

# --- Generate the Comparison Bar Chart ---
plt.figure(figsize=(9, 6))

# Define the x-axis positions and width of the bars
bar_width = 0.35
index = np.arange(len(results_df['Model']))

# Plot the bars
bar1 = plt.bar(index, results_df['ROC AUC'], bar_width, label='ROC AUC', color='#005088')
bar2 = plt.bar(index + bar_width, results_df['Peak F1'], bar_width, label='Peak F1', color='#6366f1')

# Aesthetics and formatting
plt.xlabel('Feature Set', fontweight='bold', fontsize=12)
plt.ylabel('Score', fontweight='bold', fontsize=12)
plt.title('Performance Uplift: Feature Fusion', fontweight='bold', fontsize=14)
plt.xticks(index + bar_width / 2, results_df['Model'], fontsize=11)
plt.ylim(0, 1.1) # Gives the bars breathing room at the top
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add exact numbers on top of the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),  # 5 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

add_value_labels(bar1)
add_value_labels(bar2)

plt.tight_layout()

plt.savefig('outputs/feature_fusion_comparison.png', dpi=300, bbox_inches='tight')
print("Chart successfully saved")
