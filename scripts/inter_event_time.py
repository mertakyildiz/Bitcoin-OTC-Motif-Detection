import math
import pandas as pd

# Load the dataset
df = pd.read_csv('data/soc-sign-bitcoinotc.csv', names=['source', 'target', 'rating', 'time'])

# 1. Sort strictly by time to simulate the global chronological feed
df = df.sort_values('time').reset_index(drop=True)

print("Calculating Mean Inter-event Time (delta)")
# Calculate the time difference between row[i] and row[i-1] globally
df['time_diff'] = df['time'].diff()
# Mean inter-event time in seconds
delta = df['time_diff'].mean()
print(f"Global Mean Inter-event Time (delta): {delta:.2f} seconds")

print("Calculating Connectivity Rate (gamma)")
# To find gamma, we look at consecutive pairs of events globally.
# A pair is 'connected' if they share at least one node (source or target).
# We shift the source and target columns to compare row[i] with row[i-1]
df['prev_source'] = df['source'].shift(1)
df['prev_target'] = df['target'].shift(1)

# Check if current source/target matches previous source/target
connected_mask = (
    (df['source'] == df['prev_source']) |
    (df['source'] == df['prev_target']) |
    (df['target'] == df['prev_source']) |
    (df['target'] == df['prev_target'])
)

# Gamma is the fraction of connected consecutive events
gamma = connected_mask.mean()
print(f"Global connectivity rate (gamma): {gamma:.4f} (or {gamma*100:.2f}%)")

print("--- Final Calculation ---")
# Calculate Delta_C
delta_c_seconds = delta / gamma
delta_c_hours = delta_c_seconds / (60 * 60)
delta_c_days = delta_c_seconds / (60 * 60 * 24)
delta_c_hours_round = math.ceil(delta_c_hours)
delta_c_hours_round_in_sec = delta_c_hours_round * (60 * 60)

print(f"Ideal Inter-event Time Window (Delta_C): {delta_c_seconds:.2f} seconds")
print(f"Equivalent in hours: {delta_c_days:.2f} hours")
print(f"Rounded Delta_C for building degree vectors: {delta_c_hours_round_in_sec:.2f} seconds")
