import pandas as pd
import inter_event_time


# Load data and ground truth
print("Loading network data")
df = pd.read_csv('soc-sign-bitcoinotc.csv', names=['source', 'target', 'rating', 'time'])

print("Loading official ground truth (otc_gt.csv)")
gt_df = pd.read_csv('otc_gt.csv', names=['node', 'label'])
verified_users = set(gt_df['node'])

# Empirically calculated and rounded up to the nearest hour per Liu et al.
DELTA_C = inter_event_time.delta_c_hours_round_in_sec


# Egocentric event-pair extraction
def extract_normalized_motifs(target_user, network_df, delta_c):
    """
    Extracts 2-event motifs for a 1-hop egocentric network and applies the
    x_i normalization formula from 'Temporal Motifs for Financial Networks'.
    """
    # 1. Isolate the true 1-hop Egocentric Network (G_u)
    # Find all direct neighbors of the target user
    direct_edges = network_df[(network_df['source'] == target_user) | (network_df['target'] == target_user)]
    neighbors = set(direct_edges['source']).union(set(direct_edges['target']))

    # Extract all edges involving any of the neighbors
    ego_df = network_df[(network_df['source'].isin(neighbors)) & (network_df['target'].isin(neighbors))]
    ego_df = ego_df.sort_values('time').reset_index(drop=True)

    # 2. Initialize Counters for the 6 Event Pairs
    counts = {
        'repetition': {'u': 0, 'not_u': 0},
        'ping_pong': {'u': 0, 'not_u': 0},
        'in_burst': {'u': 0, 'not_u': 0},
        'out_burst': {'u': 0, 'not_u': 0},
        'convey': {'u': 0, 'not_u': 0},
        'weakly_connected': {'u': 0, 'not_u': 0}
    }

    # 3. Scan for Event Pairs within Delta_C
    total_events = len(ego_df)
    for i in range(total_events):
        u1, v1, t1 = ego_df.iloc[i]['source'], ego_df.iloc[i]['target'], ego_df.iloc[i]['time']

        # Look ahead for connected events within the 3-hour window
        for j in range(i + 1, total_events):
            u2, v2, t2 = ego_df.iloc[j]['source'], ego_df.iloc[j]['target'], ego_df.iloc[j]['time']

            # If the time gap exceeds Delta_C, stop looking ahead for Event 1
            if t2 - t1 > delta_c:
                break

            # Check if they share at least one node (connectivity constraint)
            if len({u1, v1}.intersection({u2, v2})) > 0:
                motif_type = None

                # The 6-Letter Event Pair Alphabet definitions
                if u1 == u2 and v1 == v2:
                    motif_type = 'repetition'
                elif u1 == v2 and v1 == u2:
                    motif_type = 'ping_pong'
                elif u1 != u2 and v1 == v2:
                    motif_type = 'in_burst'
                elif u1 == u2 and v1 != v2:
                    motif_type = 'out_burst'
                elif v1 == u2 and u1 != v2:
                    motif_type = 'convey'
                elif u1 == v2 and v1 != u2:
                    motif_type = 'weakly_connected'

                if motif_type:
                    # Check if the target user was involved in this specific motif
                    if target_user in [u1, v1, u2, v2]:
                        counts[motif_type]['u'] += 1
                    else:
                        counts[motif_type]['not_u'] += 1

    # 4. Normalize Features (The x_i formula)
    features = {'node': target_user}
    for motif, val in counts.items():
        m_u = val['u']  # |M_u^i|
        m_not_u = val['not_u']  # |M_¬u^i|

        # x_i = |M_u^i| / (|M_u^i| + |M_¬u^i|)
        total_occurrences = m_u + m_not_u
        features[motif] = (m_u / total_occurrences) if total_occurrences > 0 else 0.0

    return features


# Execute pipeline
print(f"Extracting normalized features for {len(verified_users)} verified users")
results = []

for idx, user in enumerate(verified_users):
    user_features = extract_normalized_motifs(user, df, DELTA_C)
    results.append(user_features)


# Convert to dataframe and enforce clean column order
all_features_df = pd.DataFrame(results)
all_features_df = all_features_df[['node', 'out_burst', 'in_burst', 'ping_pong', 'repetition', 'convey',
                                   'weakly_connected']]


# Save final matrix

all_features_df.to_csv('extracted_motifs_X.csv', index=False)
print(f"-> Saved 'extracted_motifs_X.csv' ({len(all_features_df)} rows)")
print("Feature extraction complete. Ready for model training.")