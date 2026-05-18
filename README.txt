Temporal Motif-Based Fraud Detection

This repository contains an end-to-end machine learning pipeline for dynamic graph anomaly detection. It identifies fraudulent users in the Bitcoin-OTC rating network based purely on temporal network motifs and acting as a "blind classifier" that completely ignores potentially manipulated rating scores. 

The architecture is mathematically backed by the state-of-the-art literature on temporal network motifs and dynamic graphlets.

## Repository Structure

### Data Files
*   'soc-sign-bitcoinotc.csv': The raw temporal network dataset (from Stanford SNAP). Formatted as: `[source, target, rating, time]`.
*   'otc_gt.csv': The verified ground truth labels. Contains verified benign (`1`) and fraudulent (`-1`) nodes.

### Pipeline Scripts
*   'inter_event_time.py': Calculates the empirical rhythm of the network.
*   'temp_motif_deg_vec.py': The core feature extraction engine. 
*   'model_training.py': The machine learning classifier.
*   'ablation_study.py': The contextual upgrade script for performance testing.

---

## How to Run the Pipeline

Execute the scripts in the following order to successfully train the fraud detection model:

### Step 1: Find the Network's Pulse
Run 'inter_event_time.py'. 
Instead of arbitrarily guessing a time window, this script empirically calculates the Inter-event Time Window (Delta_C) using the formula Delta_C = delta / gamma (mean inter-event time divided by the consecutive connectivity rate). It rounds up the output to a strict 3-hour window. 

### Step 2: Extract Topological Fingerprints
Run 'temp_motif_deg_vec.py'.
This script dynamically imports the Delta_C limit from Step 1. It isolates each user into a 1-hop egocentric network and scans for 6 fundamental 2-event building blocks (Repetition, Ping-pong, In-burst, Out-burst, Convey, Weakly-connected). It outputs an 'extracted_motifs_X.csv' matrix where counts are mathematically normalized (x_i) to prevent high-volume users from being falsely flagged as fraudulent.

### Step 3: Train the Blind Classifier
Run 'model_training.py'.
This script merges the extracted feature matrix with the 'otc_gt.csv' ground truth. It trains a Random Forest Classifier using 5-Fold GridSearchCV. It performs automated threshold optimization to maximize the F1-Score against "camouflaged" fraudsters and outputs the Feature Importance plot to reveal the dominant malicious motifs (e.g. retaliatory "Ping-Pongs").

### Step 4: Evaluate with the Ablation Study
Run 'ablation_study.py'.
This script computes three "Simple Graph Features" (s_u total volume, s_u^{out}/s_u out-ratio, and k_u node degree). It evaluates the classifier's performance using "Motifs Only" versus "Motifs + Simple Graph" to demonstrate the predictive boost provided by contextual network statistics.

---

## Academic References
This pipeline synthesizes the methodologies from the following core papers:
1. Longa et al. (2021). "An efficient procedure for mining egocentric temporal motifs." (Introduces the concept of an egocentric temporal neighborhood (ETN))
1. Liu et al. (2023). "Temporal Motifs for Financial Networks." (Egocentric isolation, x_i normalization, simple graph feature integration).
2. Liu et al. (2020). "Temporal Network Motifs - Models, Limitations, Evaluation." (Bypassing subgraph isomorphism via the 6-letter Event Pair alphabet).
3. Kumar, S., et al. (2018). "REV2: Fraudulent User Prediction in Rating Platforms." (Verified ground truth labeling and addressing fraudster camouflage).