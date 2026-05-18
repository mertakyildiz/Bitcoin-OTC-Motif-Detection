import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score

# 1. Load data
network_df = pd.read_csv('soc-sign-bitcoinotc.csv', names=['source', 'target', 'rating', 'time'])
gt_df = pd.read_csv('otc_gt.csv', names=['node', 'label'])
gt_df['is_fraud'] = gt_df['label'].apply(lambda x: 1 if x == -1 else 0)

# Load the motifs-only features for the 316 users
X_motifs_df = pd.read_csv('extracted_motifs_X.csv')

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


print("--- Feature fusion study ---")
print(pd.DataFrame([evaluate(motifs_cols, "Motifs Only"), evaluate(combined_cols, "Motifs + Simple Graph")]))