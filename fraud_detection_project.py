# =============================================================================
# CREDIT CARD FRAUD DETECTION PROJECT
# Using: Isolation Forest | Local Outlier Factor | Autoencoder (Deep Learning)
# Suitable for: Python Beginners
# =============================================================================
# ABOUT THE DATASET:
# We use the famous "creditcard.csv" dataset from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#
# It has 284,807 transactions with 492 frauds (~0.17% fraud rate).
# Features V1-V28 are PCA-transformed (anonymous for privacy).
# 'Time', 'Amount', and 'Class' (0=legit, 1=fraud) are original columns.
#
# HOW TO RUN THIS FILE:
# 1. Download creditcard.csv from Kaggle (link above)
# 2. Place it in the same folder as this script
# 3. Install dependencies: pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
# 4. Run: python fraud_detection_project.py
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# TensorFlow / Keras for the Autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# Make results reproducible
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 65)
print("   CREDIT CARD FRAUD DETECTION — ANOMALY DETECTION PROJECT")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD AND EXPLORE THE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading and Exploring Data...")

# Load the dataset
df = pd.read_csv("creditcard.csv")

print(f"\n  Dataset Shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  First 5 rows:")
print(df.head())

# Check for missing values
print(f"\n  Missing Values: {df.isnull().sum().sum()} (none expected)")

# Check class distribution
fraud_count   = df["Class"].sum()
legit_count   = len(df) - fraud_count
total         = len(df)

print(f"\n  Class Distribution:")
print(f"    Legitimate Transactions : {legit_count} ({100*legit_count/total:.2f}%)")
print(f"    Fraudulent Transactions : {fraud_count} ({100*fraud_count/total:.2f}%)")
print(f"\n  ⚠ This is a highly IMBALANCED dataset — fraud is very rare!")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: VISUALIZE THE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 2] Creating Visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Credit Card Fraud Detection — Data Overview", fontsize=15, fontweight="bold")

# Plot 1: Class Distribution
axes[0].bar(["Legitimate", "Fraud"], [legit_count, fraud_count],
            color=["#2ecc71", "#e74c3c"], edgecolor="black")
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Number of Transactions")
for i, v in enumerate([legit_count, fraud_count]):
    axes[0].text(i, v + 500, str(v), ha="center", fontweight="bold")

# Plot 2: Transaction Amount Distribution
axes[1].hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.7,
             color="#2ecc71", label="Legitimate", density=True)
axes[1].hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.7,
             color="#e74c3c", label="Fraud", density=True)
axes[1].set_title("Transaction Amount Distribution")
axes[1].set_xlabel("Amount")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].set_xlim(0, 1000)

# Plot 3: Transaction Time Distribution
axes[2].hist(df[df["Class"] == 0]["Time"], bins=50, alpha=0.7,
             color="#2ecc71", label="Legitimate", density=True)
axes[2].hist(df[df["Class"] == 1]["Time"], bins=50, alpha=0.7,
             color="#e74c3c", label="Fraud", density=True)
axes[2].set_title("Transaction Time Distribution")
axes[2].set_xlabel("Time (seconds)")
axes[2].legend()

plt.tight_layout()
plt.savefig("01_data_overview.png", dpi=120, bbox_inches="tight")
plt.show()
print("  Saved: 01_data_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PREPROCESS THE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP 3] Preprocessing Data...")

# Scale 'Amount' and 'Time' to match the V1-V28 PCA features
# StandardScaler makes every feature have mean=0 and std=1
# This is important because models are sensitive to scale

scaler = StandardScaler()

df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])

# Drop original Amount and Time columns (replaced by scaled versions)
df.drop(columns=["Amount", "Time"], inplace=True)

# Separate features (X) and labels (y)
X = df.drop(columns=["Class"]).values   # shape: (284807, 29)
y = df["Class"].values                  # 0 = legitimate, 1 = fraud

print(f"  Feature matrix shape : {X.shape}")
print(f"  Label vector shape   : {y.shape}")

# For Isolation Forest and LOF, we work on a SAMPLE for speed
# (These algorithms are slow on 280k rows for beginners)
# We take all frauds + a random sample of legitimate transactions

fraud_idx = np.where(y == 1)[0]
legit_idx = np.where(y == 0)[0]

# Use all 492 frauds + 5000 random legitimate ones (for fast demos)
sample_legit_idx = np.random.choice(legit_idx, size=5000, replace=False)
sample_idx       = np.concatenate([fraud_idx, sample_legit_idx])
np.random.shuffle(sample_idx)

X_sample = X[sample_idx]
y_sample = y[sample_idx]

print(f"\n  Sampled Dataset for IF & LOF:")
print(f"    Total samples : {len(X_sample)}")
print(f"    Fraud         : {y_sample.sum()}")
print(f"    Legitimate    : {(y_sample == 0).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION: Evaluate and Visualize Results
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name, y_true, y_pred):
    """
    Prints precision, recall, F1-score and plots a confusion matrix.
    
    Parameters:
    - model_name : string name for display
    - y_true     : actual labels (0 or 1)
    - y_pred     : predicted labels (0 or 1)
    """
    print(f"\n  {'─'*50}")
    print(f"  Results for: {model_name}")
    print(f"  {'─'*50}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    print(f"  Precision  : {precision:.4f}  (of predicted frauds, how many are real?)")
    print(f"  Recall     : {recall:.4f}    (of all real frauds, how many did we catch?)")
    print(f"  F1-Score   : {f1:.4f}    (harmonic mean of precision & recall)")
    print(f"\n  Full Report:")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))

    # Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"])
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    fname = f"confusion_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")

    return {"model": model_name, "precision": precision,
            "recall": recall, "f1": f1}


# Store results for final comparison
all_results = []


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: ISOLATION FOREST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("   MODEL 1: ISOLATION FOREST")
print("=" * 65)
print("""
  HOW IT WORKS:
  ─────────────
  Isolation Forest is like a game of "isolate the odd one out."
  
  Imagine you're sorting a bag of 100 red marbles and 5 blue ones.
  If you randomly split the bag again and again, the rare blue marbles
  get isolated FASTER because they're different and don't cluster.
  
  The algorithm builds random decision trees. Frauds (anomalies) are
  isolated in fewer splits → they have shorter "path lengths" in the tree.
  
  Key parameter: contamination = expected fraction of anomalies in data.
""")

# Contamination = fraction of frauds in our sample
contamination_rate = y_sample.sum() / len(y_sample)
print(f"  Contamination rate (fraud %) : {contamination_rate:.4f}")

# Train Isolation Forest
# n_estimators = number of trees to build (more = more accurate but slower)
iso_forest = IsolationForest(
    n_estimators=100,         # Build 100 isolation trees
    contamination=contamination_rate,  # Expected % of anomalies
    random_state=42,
    n_jobs=-1                 # Use all CPU cores
)

print("\n  Training Isolation Forest...")
iso_forest.fit(X_sample)

# Predict: Isolation Forest returns -1 for anomaly, +1 for normal
# We convert: -1 → 1 (fraud), +1 → 0 (legitimate)
y_pred_iso = iso_forest.predict(X_sample)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

# Evaluate
result_iso = evaluate_model("Isolation Forest", y_sample, y_pred_iso)
all_results.append(result_iso)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: LOCAL OUTLIER FACTOR (LOF)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("   MODEL 2: LOCAL OUTLIER FACTOR (LOF)")
print("=" * 65)
print("""
  HOW IT WORKS:
  ─────────────
  LOF is a neighborhood-based method. Think of it like this:
  
  "If a house is in a dense neighborhood but is surrounded by a 
   big empty field — it's an outlier."
  
  For each transaction, LOF looks at its k nearest neighbors and 
  compares the LOCAL DENSITY of that point vs. its neighbors.
  
  - If a point is in a dense cluster → normal (low LOF score)
  - If a point is isolated from its neighbors → anomaly (high LOF score)
  
  Key parameter: n_neighbors = how many neighbors to compare with.
  
  ⚠ LOF only supports predict() after fit() with novelty=True,
    or we use fit_predict() on the same data (transductive).
""")

# n_neighbors: number of neighbors to look at
# contamination: expected fraction of outliers
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination_rate,
    n_jobs=-1
)

print("  Training & Predicting with LOF (fit_predict)...")
# LOF uses fit_predict() — it trains and predicts on the same data
y_pred_lof = lof.fit_predict(X_sample)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)  # -1 → fraud, +1 → legit

# Evaluate
result_lof = evaluate_model("Local Outlier Factor", y_sample, y_pred_lof)
all_results.append(result_lof)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3: AUTOENCODER (Deep Learning)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("   MODEL 3: AUTOENCODER (Neural Network)")
print("=" * 65)
print("""
  HOW IT WORKS:
  ─────────────
  An Autoencoder is a neural network that learns to COMPRESS and 
  then RECONSTRUCT its input.
  
  Architecture:
    Input (29 features) → Encoder → Bottleneck → Decoder → Output (29 features)
  
  Training Strategy:
    We train ONLY on NORMAL transactions (legitimate ones).
    The network learns what "normal" looks like.
  
  Detection Strategy:
    When we feed a FRAUD transaction, the autoencoder struggles to 
    reconstruct it well (because it's never seen that pattern).
    The RECONSTRUCTION ERROR is high → we flag it as fraud!
    
    Low reconstruction error  → Legitimate
    High reconstruction error → Fraud (anomaly)
""")

# ── 3a. Prepare Data for Autoencoder ──────────────────────────────────────
# Use the FULL dataset for the autoencoder (it handles scale better)
# Scale all features
X_scaled = StandardScaler().fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# For Autoencoder training: use ONLY legitimate transactions from train set
X_train_normal = X_train[y_train == 0]

print(f"  Autoencoder Training Set (legitimate only): {X_train_normal.shape}")
print(f"  Autoencoder Test Set (all): {X_test.shape}")

# ── 3b. Build the Autoencoder ─────────────────────────────────────────────
input_dim = X_train.shape[1]   # 29 features

# Input layer
input_layer = Input(shape=(input_dim,), name="Input")

# ENCODER: compress the input into a smaller representation
encoded = Dense(16, activation="relu", name="Encoder_1")(input_layer)
encoded = Dense(8,  activation="relu", name="Bottleneck")(encoded)   # Compressed!

# DECODER: reconstruct back to original size
decoded = Dense(16, activation="relu", name="Decoder_1")(encoded)
decoded = Dense(input_dim, activation="linear", name="Output")(decoded)   # Reconstruction

# Full Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded, name="Autoencoder")
autoencoder.compile(optimizer="adam", loss="mse")   # MSE = reconstruction loss

print("\n  Autoencoder Architecture:")
autoencoder.summary()

# ── 3c. Train the Autoencoder ─────────────────────────────────────────────
print("\n  Training Autoencoder on NORMAL transactions only...")

# EarlyStopping: stop training if validation loss doesn't improve
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,          # Wait 5 epochs before stopping
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_normal, X_train_normal,   # Input = Target (self-supervised!)
    epochs=30,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ── 3d. Plot Training History ─────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Training Loss",   color="#3498db")
plt.plot(history.history["val_loss"], label="Validation Loss", color="#e74c3c")
plt.title("Autoencoder Training — Reconstruction Loss", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("02_autoencoder_training.png", dpi=120, bbox_inches="tight")
plt.show()
print("  Saved: 02_autoencoder_training.png")

# ── 3e. Detect Fraud via Reconstruction Error ──────────────────────────────
print("\n  Computing reconstruction errors on test set...")

# Get reconstructed outputs
X_test_reconstructed = autoencoder.predict(X_test, verbose=0)

# Reconstruction error = Mean Squared Error per sample
# Higher error → more likely to be fraud
reconstruction_errors = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)

# ── 3f. Choose Threshold ──────────────────────────────────────────────────
# We pick the threshold as the 95th percentile of errors on NORMAL test data
# (meaning: flag top 5% errors as fraud)
normal_errors = reconstruction_errors[y_test == 0]
threshold = np.percentile(normal_errors, 95)

print(f"  Threshold for fraud detection: {threshold:.6f}")
print(f"  (Transactions with error > {threshold:.6f} are flagged as fraud)")

# Classify based on threshold
y_pred_ae = (reconstruction_errors > threshold).astype(int)

# ── 3g. Visualize Reconstruction Errors ───────────────────────────────────
plt.figure(figsize=(10, 5))

plt.hist(reconstruction_errors[y_test == 0], bins=100, alpha=0.7,
         color="#2ecc71", label="Legitimate", density=True)
plt.hist(reconstruction_errors[y_test == 1], bins=100, alpha=0.7,
         color="#e74c3c", label="Fraud", density=True)
plt.axvline(threshold, color="black", linestyle="--", linewidth=2,
            label=f"Threshold = {threshold:.4f}")
plt.title("Autoencoder — Reconstruction Error Distribution", fontweight="bold")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Density")
plt.legend()
plt.xlim(0, np.percentile(reconstruction_errors, 99))   # Zoom in
plt.tight_layout()
plt.savefig("03_reconstruction_error.png", dpi=120, bbox_inches="tight")
plt.show()
print("  Saved: 03_reconstruction_error.png")

# Evaluate
result_ae = evaluate_model("Autoencoder", y_test, y_pred_ae)
all_results.append(result_ae)


# ═════════════════════════════════════════════════════════════════════════════
# FINAL: COMPARE ALL THREE MODELS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("   FINAL COMPARISON OF ALL MODELS")
print("=" * 65)

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index("model")

print("\n", results_df.to_string())

# Plot comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.25

bars1 = ax.bar(x - width, results_df["precision"], width, label="Precision",
               color="#3498db", edgecolor="black")
bars2 = ax.bar(x,          results_df["recall"],   width, label="Recall",
               color="#e74c3c",  edgecolor="black")
bars3 = ax.bar(x + width,  results_df["f1"],       width, label="F1-Score",
               color="#2ecc71",  edgecolor="black")

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{height:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Precision, Recall, F1-Score", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(results_df.index)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("04_model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("  Saved: 04_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY & BEGINNER NOTES
# ─────────────────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════╗
║              SUMMARY & KEY LEARNING POINTS                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📌 METRICS TO UNDERSTAND:                                   ║
║  ──────────────────────────────────────────────────────────  ║
║  Precision = Of all flagged frauds, how many were real?      ║
║  Recall    = Of all real frauds, how many did we catch?      ║
║  F1-Score  = Balance between Precision and Recall            ║
║                                                              ║
║  In fraud detection, HIGH RECALL is more important!          ║
║  Missing a fraud is worse than a false alarm.                ║
║                                                              ║
║  📌 MODEL COMPARISON:                                        ║
║  ──────────────────────────────────────────────────────────  ║
║  Isolation Forest  → Fast, good for large datasets           ║
║  LOF               → Excellent for local pattern detection   ║
║  Autoencoder       → Best overall, learns deep patterns      ║
║                                                              ║
║  📌 KEY CONCEPTS USED:                                       ║
║  ──────────────────────────────────────────────────────────  ║
║  • Anomaly Detection (unsupervised/semi-supervised)          ║
║  • Class Imbalance handling                                  ║
║  • StandardScaler for feature normalization                  ║
║  • Reconstruction Error as an anomaly score                  ║
║  • Threshold tuning for classification                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("\n✅ All done! Check your folder for the saved plots.")
