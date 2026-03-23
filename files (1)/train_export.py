# =============================================================
# train_export.py
# Step 1: Load & Balance Dataset
# Step 2: Train Random Forest vs Gradient Boosting
# Step 3: Export best model as .pkl using joblib
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("AgroFertilizerLoss.csv")
print(f"[1] Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} columns")

# ── 2. DATASET BALANCING ──────────────────────────────────────
TARGET = "Total_Fertilizer_Loss_kg_ha"
DROP   = ["Fertilizer_Loss_Percentage","Nitrogen_Loss","Phosphorus_Loss","Potassium_Loss"]

# Bin target into 3 equal groups using percentiles
p33 = df[TARGET].quantile(0.33)
p66 = df[TARGET].quantile(0.66)
bins   = [0, p33, p66, 9999]
labels = ["Low Loss", "Medium Loss", "High Loss"]
df["loss_group"] = pd.cut(df[TARGET], bins=bins, labels=labels)
before = df["loss_group"].value_counts().sort_index()
print("\n[2] Distribution BEFORE balancing:")
print(before)

# Under-Sampling — match all groups to the smallest
min_count  = int(before.min())
balanced_df = pd.concat([
    df[df["loss_group"] == g].sample(min_count, random_state=42)
    for g in labels
]).drop(columns=["loss_group"]).reset_index(drop=True)

after_bins = pd.cut(balanced_df[TARGET], bins=bins, labels=labels).value_counts().sort_index()
print("\n[2] Distribution AFTER balancing (Under-Sampling):")
print(after_bins)

# Save comparison chart
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
before.plot(kind="bar", ax=axes[0], color=["#2C5F2D","#97BC62","#E07B39"], rot=0)
axes[0].set_title("Before Balancing")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()+50), ha="center", fontsize=9)

after_bins.plot(kind="bar", ax=axes[1], color=["#2C5F2D","#97BC62","#E07B39"], rot=0)
axes[1].set_title("After Balancing (Under-Sampling)")
axes[1].set_ylabel("Count")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height()):,}", (p.get_x()+p.get_width()/2, p.get_height()+20), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("static/balance_chart.png", dpi=120)
plt.close()
print("[2] Balance chart saved")

# ── 3. PREPARE FEATURES ───────────────────────────────────────
X = balanced_df.drop(columns=[TARGET] + DROP, errors="ignore")
y = balanced_df[TARGET]

# Compute dataset-wide defaults for the hidden columns
# (used later in Flask when user only fills 6 fields)
cat_defaults = {}
num_defaults = {}
all_cat_cols = X.select_dtypes(include="object").columns.tolist()
all_num_cols = X.select_dtypes(exclude="object").columns.tolist()

for c in all_cat_cols:
    cat_defaults[c] = str(X[c].mode()[0])
for c in all_num_cols:
    cat_defaults[c] = float(X[c].mean())  # won't hit; just for safety

# Encode all categoricals
encoders = {}
for col in all_cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

feature_cols = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
print(f"\n[3] Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4. TRAIN MODELS ───────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("[4] Random Forest trained")

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("[4] Gradient Boosting trained")

# ── 5. EVALUATE ───────────────────────────────────────────────
def get_metrics(y_true, y_pred, name):
    return {
        "Model": name,
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "MSE":  round(mean_squared_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }

results = pd.DataFrame([
    get_metrics(y_test, rf_pred, "Random Forest"),
    get_metrics(y_test, gb_pred, "Gradient Boosting"),
]).set_index("Model")

print("\n── RESULTS ─────────────────────────────")
print(results)
best = results["R2"].idxmax()
print(f"\n Best Model: {best}")

# ── 6. EXPORT ─────────────────────────────────────────────────
best_model = gb if best == "Gradient Boosting" else rf

# Defaults for the 16 hidden columns (avg from full dataset)
hidden_defaults = {
    "Soil_pH":                7.002,
    "Soil_Organic_Carbon":    1.5046,
    "Total_Nitrogen":         176.2543,
    "Available_Phosphorus":   44.9216,
    "Available_Potassium":    224.7007,
    "Bulk_Density":           1.3504,
    "Temperature":            27.4337,
    "Humidity":               67.5975,
    "Wind_Speed":             3.2812,
    "Evapotranspiration":     5.0059,
    "Soil_Moisture_Level":    "Medium",
    "Growth_Stage":           "Mid",
    "Fertilizer_Treatment":   "CF",
    "Application_Method":     "Broadcast",
    "Coating_Type":           "Normal",
    "Water_Amount":           40.0,
}

joblib.dump({
    "model":           best_model,
    "encoders":        encoders,
    "feature_cols":    feature_cols,
    "cat_cols":        all_cat_cols,
    "results":         results.reset_index().to_dict("records"),
    "best":            best,
    "hidden_defaults": hidden_defaults,
}, "model.pkl")

print("\n[6] model.pkl saved!")
print("    Now run: python app.py")
