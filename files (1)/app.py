# =============================================================
# app.py — Flask Web Application
# AgroFertilizerLoss Prediction System
# Only 6 user inputs — rest auto-filled with dataset averages
# =============================================================

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model bundle
bundle         = joblib.load("model.pkl")
model          = bundle["model"]
encoders       = bundle["encoders"]
feature_cols   = bundle["feature_cols"]
cat_cols       = bundle["cat_cols"]
results        = bundle["results"]
best_name      = bundle["best"]
hidden_defaults= bundle["hidden_defaults"]

# Dropdown choices for the 6 visible inputs
OPTIONS = {
    "Crop_Type":       ["Rice", "Wheat", "Corn", "Citrus", "Vegetable"],
    "Soil_Type":       ["Sandy", "Clay", "Loamy"],
    "Irrigation_Type": ["Flood", "Drip", "Sprinkler"],
    "Fertilizer_Type": ["NPK", "Urea", "PK", "NK"],
}

# ── HOME PAGE ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           options=OPTIONS,
                           results=results,
                           best=best_name)

# ── PREDICT ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        # Build full row using user inputs + hidden defaults
        raw_row = {}

        # Fill all columns with defaults first
        for col, val in hidden_defaults.items():
            raw_row[col] = val

        # Override with the 6 user-provided inputs
        raw_row["Crop_Type"]        = form.get("Crop_Type")
        raw_row["Soil_Type"]        = form.get("Soil_Type")
        raw_row["Irrigation_Type"]  = form.get("Irrigation_Type")
        raw_row["Fertilizer_Type"]  = form.get("Fertilizer_Type")
        raw_row["Rainfall"]         = float(form.get("Rainfall"))
        raw_row["Application_Rate"] = float(form.get("Application_Rate"))

        # Encode categoricals using saved LabelEncoders
        encoded_row = {}
        for col in feature_cols:
            val = raw_row[col]
            if col in cat_cols:
                le  = encoders[col]
                val = le.transform([str(val)])[0]
            else:
                val = float(val)
            encoded_row[col] = val

        input_df   = pd.DataFrame([encoded_row])[feature_cols]
        prediction = round(float(model.predict(input_df)[0]), 2)

        # Loss level
        if prediction < 50:
            level       = "Low Loss"
            level_color = "success"
            emoji       = "✅"
            advice      = "Great! Fertilizer loss is low. Your farming conditions are efficient and well-managed."
        elif prediction < 100:
            level       = "Medium Loss"
            level_color = "warning"
            emoji       = "⚠️"
            advice      = "Moderate fertilizer loss detected. Consider switching to Drip irrigation or using Controlled-Release coating to reduce waste."
        else:
            level       = "High Loss"
            level_color = "danger"
            emoji       = "🚨"
            advice      = "High fertilizer loss! Try reducing your Application Rate, switching to Drip irrigation, or using Biochar coating to retain nutrients."

        return render_template("result.html",
            prediction  = prediction,
            level       = level,
            level_color = level_color,
            emoji       = emoji,
            advice      = advice,
            crop        = form.get("Crop_Type"),
            soil        = form.get("Soil_Type"),
            irrigation  = form.get("Irrigation_Type"),
            fertilizer  = form.get("Fertilizer_Type"),
            rainfall    = form.get("Rainfall"),
            app_rate    = form.get("Application_Rate"),
        )

    except Exception as e:
        return render_template("result.html", prediction=None, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
