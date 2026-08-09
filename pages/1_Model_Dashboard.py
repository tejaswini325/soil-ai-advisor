"""
Model Dashboard — SoilSense AI
Displays training metrics, feature importance, and confusion matrix
for the trained Random Forest model.
"""

import streamlit as st
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_engine import get_model_metrics

st.set_page_config(page_title="Model Dashboard · SoilSense AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
* { font-family: 'Sora', sans-serif !important; }
.metric-card {
    background: white; border-radius: 12px; padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
    border-top: 4px solid #2D6A4F;
}
.big-num { font-size: 2.4rem; font-weight: 700; color: #1B4F72; }
.label   { font-size: 0.85rem; color: #888; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Model Training Dashboard")
st.caption("Random Forest Classifier · UCI Crop Recommendation Dataset (Ingle, 2020)")

try:
    m = get_model_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Test Accuracy",   f"{m['accuracy']*100:.2f}%"),
        ("Weighted F1",     f"{m['f1_weighted']*100:.2f}%"),
        ("CV Accuracy",     f"{m['cv_mean']*100:.2f}%"),
        ("Crop Classes",    str(len(m['classes']))),
        ("Trees",           str(m['n_estimators'])),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="big-num">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns(2)
    plots_dir = Path(__file__).parent.parent / "models"

    with left:
        st.markdown("### 🌿 Feature Importance")
        img = plots_dir / "feature_importance.png"
        if img.exists():
            st.image(str(img), use_container_width=True)

    with right:
        st.markdown("### 🗺️ Confusion Matrix")
        img2 = plots_dir / "confusion_matrix.png"
        if img2.exists():
            st.image(str(img2), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔁 5-Fold Cross-Validation Results")
    import pandas as pd
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(m["cv_scores"]))],
        "Accuracy (%)": [round(s*100, 2) for s in m["cv_scores"]]
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.info(f"Mean CV Accuracy: **{m['cv_mean']*100:.2f}%** ± {m['cv_std']*100:.2f}%")

    st.markdown("### 🌾 Supported Crop Classes")
    classes_html = " ".join([
        f'<span style="background:#D5F5E3;color:#1E8449;padding:4px 12px;border-radius:20px;'
        f'font-size:0.82rem;font-weight:600;margin:3px;display:inline-block">{c.title()}</span>'
        for c in m["classes"]
    ])
    st.markdown(classes_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**References:**
- Ingle, A. (2020). *Crop Recommendation Dataset*. Kaggle.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- scikit-learn: Pedregosa et al. (2011). *JMLR*, 12, 2825–2830.
    """)

except Exception as e:
    st.error(f"Could not load model metrics: {e}")
    st.info("Run `python train_model.py` first to train the model.")
