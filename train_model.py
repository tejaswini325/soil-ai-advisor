"""
train_model.py — SoilSense AI Model Training Script
=====================================================
Trains a Random Forest Classifier on the UCI Crop Recommendation Dataset.
Saves model artifacts to models/ directory for use in the Streamlit app.

Dataset: UCI Crop Recommendation Dataset (Ingle, 2020) - Kaggle
         kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
         2,200 records | 7 features | 22 crop classes

References:
  - Ingle, A. (2020). Crop Recommendation Dataset. Kaggle.
  - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
  - ICAR Agrometeorology Guidelines (2022)

Usage:
  python train_model.py

Outputs (saved to models/):
  - crop_model.pkl        : Trained Random Forest model
  - label_encoder.pkl     : LabelEncoder for crop class names
  - scaler.pkl            : StandardScaler for feature normalization
  - model_metrics.json    : Accuracy, F1, confusion matrix data
  - feature_importance.png: Feature importance bar chart
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "crop_recommendation.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"


def load_data():
    print("📂 Loading UCI Crop Recommendation Dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✅ Loaded {len(df)} records | {df[TARGET_COL].nunique()} crop classes")
    print(f"   Crops: {sorted(df[TARGET_COL].unique())}\n")
    return df


def preprocess(df):
    print("⚙️  Preprocessing data...")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"   Features : {FEATURE_COLS}")
    print(f"   Classes  : {list(le.classes_)}\n")
    return X_scaled, y_enc, le, scaler


def train(X, y):
    print("🌲 Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"\n📊 MODEL PERFORMANCE")
    print(f"   Test Accuracy    : {acc*100:.2f}%")
    print(f"   Weighted F1      : {f1*100:.2f}%")
    print(f"   5-Fold CV Mean   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return model, X_test, y_test, y_pred, acc, f1, cv_scores


def save_artifacts(model, le, scaler, acc, f1, cv_scores, y_test, y_pred):
    print("\n💾 Saving model artifacts...")

    with open(MODELS_DIR / "crop_model.pkl",    "wb") as f: pickle.dump(model,  f)
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f: pickle.dump(le,     f)
    with open(MODELS_DIR / "scaler.pkl",        "wb") as f: pickle.dump(scaler, f)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy"       : round(float(acc), 4),
        "f1_weighted"    : round(float(f1),  4),
        "cv_mean"        : round(float(cv_scores.mean()), 4),
        "cv_std"         : round(float(cv_scores.std()),  4),
        "cv_scores"      : [round(s, 4) for s in cv_scores.tolist()],
        "classes"        : list(le.classes_),
        "n_estimators"   : model.n_estimators,
        "feature_names"  : FEATURE_COLS,
        "train_samples"  : int(len(y_test) * 4),   # approx 80%
        "test_samples"   : int(len(y_test)),
        "confusion_matrix": cm.tolist()
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"   ✅ crop_model.pkl    ({(MODELS_DIR/'crop_model.pkl').stat().st_size/1024:.1f} KB)")
    print(f"   ✅ label_encoder.pkl")
    print(f"   ✅ scaler.pkl")
    print(f"   ✅ model_metrics.json")

    return metrics


def plot_feature_importance(model, metrics):
    print("\n📈 Generating plots...")
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names       = [FEATURE_COLS[i] for i in indices]
    vals        = importances[indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SoilSense AI — Random Forest Model Analysis", fontsize=14, fontweight="bold")

    # Feature importance
    colors = ["#2D6A4F" if v == max(vals) else "#52B788" for v in vals]
    axes[0].barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    axes[0].set_xlabel("Importance Score", fontsize=11)
    axes[0].set_title("Feature Importance", fontsize=12, fontweight="bold")
    axes[0].axvline(np.mean(vals), color="#F39C12", linestyle="--", alpha=0.7, label=f"Mean={np.mean(vals):.3f}")
    axes[0].legend(fontsize=9)
    for i, (v, n) in enumerate(zip(vals[::-1], names[::-1])):
        axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color="#1B4F72")

    # CV scores bar
    cv_vals  = metrics["cv_scores"]
    cv_folds = [f"Fold {i+1}" for i in range(len(cv_vals))]
    bar_colors = ["#2D6A4F" if v == max(cv_vals) else "#74C69D" for v in cv_vals]
    axes[1].bar(cv_folds, [v*100 for v in cv_vals], color=bar_colors, edgecolor="white")
    axes[1].axhline(metrics["cv_mean"]*100, color="#F39C12", linestyle="--",
                    label=f"Mean={metrics['cv_mean']*100:.2f}%")
    axes[1].set_ylim(80, 105)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].set_title("5-Fold Cross-Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    for i, v in enumerate(cv_vals):
        axes[1].text(i, v*100 + 0.3, f"{v*100:.1f}%", ha="center", fontsize=9, color="#1B4F72")

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ feature_importance.png")


def plot_confusion_matrix(y_test, y_pred, le):
    classes   = le.classes_
    cm        = confusion_matrix(y_test, y_pred)
    n         = len(classes)
    fig_size  = max(10, n * 0.7)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Greens", xticks_rotation=45)
    ax.set_title("Confusion Matrix — Crop Prediction\n(SoilSense AI)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ confusion_matrix.png")


def print_summary(metrics):
    print("\n" + "="*55)
    print("  🌱 SOILSENSE AI — TRAINING COMPLETE")
    print("="*55)
    print(f"  Algorithm     : Random Forest ({metrics['n_estimators']} trees)")
    print(f"  Dataset       : UCI Crop Recommendation (Ingle 2020)")
    print(f"  Train samples : {metrics['train_samples']}")
    print(f"  Test samples  : {metrics['test_samples']}")
    print(f"  Test Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Weighted F1   : {metrics['f1_weighted']*100:.2f}%")
    print(f"  CV Accuracy   : {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
    print(f"  Crop classes  : {len(metrics['classes'])}")
    print("="*55)
    print(f"\n  Model saved to: models/")
    print("  Run app : streamlit run app.py\n")


if __name__ == "__main__":
    df                                      = load_data()
    X, y, le, scaler                        = preprocess(df)
    model, X_test, y_test, y_pred, acc, f1, cv_scores = train(X, y)
    metrics                                 = save_artifacts(model, le, scaler, acc, f1, cv_scores, y_test, y_pred)
    plot_feature_importance(model, metrics)
    plot_confusion_matrix(y_test, y_pred, le)
    print_summary(metrics)
