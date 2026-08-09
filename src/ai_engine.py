"""
AI Engine for Soil Report Interpreter and Fertiliser Recommender
Uses:
  - Trained Random Forest (UCI Dataset) for crop prediction
  - Anthropic Claude API for narrative report generation
  - ICAR reference data for fertiliser recommendations
  - FAO / Soil Health Card thresholds for parameter analysis

References:
  1. Ingle, A. (2020). UCI Crop Recommendation Dataset. Kaggle.
  2. ICAR-NBSS&LUP Fertilizer Dose Tables (2022)
  3. FAO World Soil Database / Soil Health Card Portal, GoI
"""

import json
import os
import pickle
import anthropic
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from typing import Any

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

load_dotenv(BASE_DIR / ".env")

_anthropic_client: Any = None


def _get_anthropic_client() -> Any:
    """Create the Anthropic client only when an AI report is requested."""
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "sk-ant-your-key-here":
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Create a .env file from .env.example "
                "and add your Anthropic API key, or set it in your terminal before running Streamlit."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client

# ─── Load model artifacts (cached at module level) ───────────────────────────
_model: Any = None
_le: Any = None
_scaler: Any = None
_metrics: dict[str, Any] | None = None


def _load_model() -> tuple[Any, Any, Any, dict[str, Any]]:
    global _model, _le, _scaler, _metrics
    if _model is None:
        with open(MODELS_DIR / "crop_model.pkl",    "rb") as f: _model   = pickle.load(f)
        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f: _le      = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl",        "rb") as f: _scaler  = pickle.load(f)
        with open(MODELS_DIR / "model_metrics.json","r")  as f: _metrics = json.load(f)
    if _metrics is None:
        raise RuntimeError("Model metrics failed to load.")
    return _model, _le, _scaler, _metrics


def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load agronomic reference CSVs."""
    crop_df      = pd.read_csv(DATA_DIR / "crop_recommendation.csv")
    fertilizer_df= pd.read_csv(DATA_DIR / "icar_fertilizer_recommendations.csv")
    thresholds_df= pd.read_csv(DATA_DIR / "soil_health_thresholds.csv")
    return crop_df, fertilizer_df, thresholds_df


def get_model_metrics() -> dict[str, Any]:
    """Return stored training metrics for display in the app."""
    _, _, _, metrics = _load_model()
    return metrics


# ─── 1. Soil Parameter Analysis ──────────────────────────────────────────────
def analyze_soil_parameters(soil_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Compare each soil parameter against FAO / Soil Health Card thresholds.
    Reference: FAO World Soil Database + soilhealth.dac.gov.in
    """
    _, _, thresholds_df = load_reference_data()
    analysis = {}

    param_mapping = {
        "ph": "ph", "nitrogen": "nitrogen", "phosphorus": "phosphorus",
        "potassium": "potassium", "organic_matter": "organic_matter",
        "electrical_conductivity": "electrical_conductivity",
        "zinc": "zinc", "sulphur": "sulphur", "boron": "boron",
    }

    for key, thresh_key in param_mapping.items():
        if key not in soil_data or soil_data[key] is None:
            continue
        val = float(soil_data[key])
        row = thresholds_df[thresholds_df["parameter"] == thresh_key]
        if row.empty:
            continue
        r = row.iloc[0]
        low, opt_low, opt_high, high = float(r.low_threshold), float(r.optimal_low), float(r.optimal_high), float(r.high_threshold)

        if   val < low:      status, severity, symptoms = "Critical Low",  "critical", r.deficiency_symptoms
        elif val < opt_low:  status, severity, symptoms = "Low",           "warning",  r.deficiency_symptoms
        elif val <= opt_high:status, severity, symptoms = "Optimal",       "good",     "None — within optimal range"
        elif val <= high:    status, severity, symptoms = "High",          "warning",  r.excess_symptoms
        else:                status, severity, symptoms = "Critical High", "critical", r.excess_symptoms

        analysis[key] = {
            "value": val, "unit": r.unit, "status": status,
            "severity": severity,
            "optimal_range": f"{opt_low} – {opt_high}",
            "symptoms": symptoms
        }
    return analysis


# ─── 2. ML Crop Prediction (Random Forest) ───────────────────────────────────
def predict_crop_ml(soil_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Predict crop suitability using the trained Random Forest model.
    Returns top-N crops with probability scores.
    Reference: UCI Crop Recommendation Dataset (Ingle, 2020)
    Algorithm : Random Forest, 200 trees, sklearn 1.x
    """
    model, le, scaler, _ = _load_model()

    features = np.array([[
        soil_data.get("nitrogen",    50),
        soil_data.get("phosphorus",  30),
        soil_data.get("potassium",   40),
        soil_data.get("temperature", 25),
        soil_data.get("humidity",    70),
        soil_data.get("ph",         6.5),
        soil_data.get("rainfall",   100),
    ]])

    X_scaled = scaler.transform(features)
    proba    = model.predict_proba(X_scaled)[0]          # shape: (n_classes,)
    top_idx  = np.argsort(proba)[::-1][:6]               # top-6

    results = []
    for idx in top_idx:
        crop_name   = le.classes_[idx]
        probability = round(float(proba[idx]) * 100, 1)
        results.append({
            "crop":              crop_name.title(),
            "suitability_score": probability,
            "confidence":        "High" if probability > 20 else "Medium" if probability > 10 else "Low"
        })
    return results


def recommend_crops(soil_data: dict[str, Any], top_n: int = 6) -> list[dict[str, Any]]:
    """
    Return crop suitability recommendations for the Streamlit app.

    This wraps the trained model prediction and keeps the app-facing API stable.
    """
    return predict_crop_ml(soil_data)[:top_n]


# ─── 3. Fertiliser Recommendations (ICAR) ────────────────────────────────────
def get_fertilizer_recommendations(crop: str, soil_analysis: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Lookup and adjust fertiliser doses for a crop.
    Reference: ICAR-NBSS&LUP crop-wise dose tables (2022)
    """
    _, fertilizer_df, _ = load_reference_data()
    crop_lower = crop.lower().strip()
    row_df = fertilizer_df[fertilizer_df["crop"].str.lower() == crop_lower]

    if row_df.empty:
        for _, r in fertilizer_df.iterrows():
            if crop_lower in r["crop"].lower() or r["crop"].lower() in crop_lower:
                row_df = pd.DataFrame([r])
                break

    if row_df.empty:
        return {"crop": crop, "found": False,
                "message": f"No specific ICAR data found for {crop}. Use general guidelines."}

    row = row_df.iloc[0]
    adjustments = {"nitrogen": 0, "phosphorus": 0, "potassium": 0}
    for nutrient, (low_adj, high_adj) in {
        "nitrogen":   (20, -40), "phosphorus": (15, -30), "potassium": (15, -25)
    }.items():
        if nutrient in soil_analysis:
            st = soil_analysis[nutrient]["status"]
            if st in ("Low", "Critical Low"):    adjustments[nutrient] = low_adj
            elif st == "Optimal":                adjustments[nutrient] = -10
            elif st in ("High","Critical High"): adjustments[nutrient] = high_adj

    base_n = int(row.nitrogen_kg_ha)
    base_p = int(row.phosphorus_kg_ha)
    base_k = int(row.potassium_kg_ha)

    return {
        "crop": crop, "found": True,
        "base_recommendations": {
            "nitrogen_kg_ha":       base_n,
            "phosphorus_kg_ha":     base_p,
            "potassium_kg_ha":      base_k,
            "organic_matter_tons_ha": int(row.organic_matter_tons_ha),
        },
        "adjusted_recommendations": {
            "nitrogen_kg_ha":       max(0, base_n + adjustments["nitrogen"]),
            "phosphorus_kg_ha":     max(0, base_p + adjustments["phosphorus"]),
            "potassium_kg_ha":      max(0, base_k + adjustments["potassium"]),
            "organic_matter_tons_ha": int(row.organic_matter_tons_ha),
        },
        "lime_recommended":   "Yes" in str(row.lime_if_acidic),
        "application_notes":  row.notes,
        "source":             row.source,
    }


# ─── 4. Narrative Report (Claude AI) ─────────────────────────────────────────
def _message_text(message: Any) -> str:
    """Extract text blocks from an Anthropic message response."""
    text_parts: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _soil_health_rating(analysis: dict[str, dict[str, Any]]) -> str:
    severity_counts = {"critical": 0, "warning": 0, "good": 0}
    for item in analysis.values():
        severity_counts[item["severity"]] += 1

    if severity_counts["critical"] >= 2:
        return "Poor"
    if severity_counts["critical"] == 1 or severity_counts["warning"] >= 4:
        return "Fair"
    if severity_counts["warning"] >= 1:
        return "Good"
    return "Excellent"


def _local_soil_report(soil_data: dict, selected_crop: str,
                       analysis: dict[str, dict[str, Any]],
                       crop_recs: list[dict[str, Any]],
                       fertilizer_rec: dict[str, Any]) -> str:
    rating = _soil_health_rating(analysis)
    alerts = [name for name, item in analysis.items() if item["severity"] == "critical"]
    warnings = [name for name, item in analysis.items() if item["severity"] == "warning"]
    best_crops = ", ".join([f"{c['crop']} ({c['suitability_score']}%)" for c in crop_recs[:3]]) or "No crop matches available"

    parameter_lines = []
    for name, item in analysis.items():
        parameter_lines.append(
            f"- **{name.replace('_', ' ').title()}**: {item['value']} {item['unit']} is **{item['status']}**. "
            f"Optimal range is {item['optimal_range']} {item['unit']}. {item['symptoms']}"
        )

    if fertilizer_rec.get("found"):
        fert = fertilizer_rec["adjusted_recommendations"]
        fertilizer_text = (
            f"Adjusted nutrient plan for {selected_crop}: N {fert['nitrogen_kg_ha']} kg/ha, "
            f"P {fert['phosphorus_kg_ha']} kg/ha, K {fert['potassium_kg_ha']} kg/ha, "
            f"and organic matter {fert['organic_matter_tons_ha']} t/ha. "
            f"{fertilizer_rec.get('application_notes', '')}"
        )
    else:
        fertilizer_text = fertilizer_rec.get("message", "Use local crop-specific fertilizer guidance.")

    return f"""## 1. Executive Summary
Overall soil health is **{rating}** for {selected_crop} in {soil_data.get('district', 'the selected district')}, {soil_data.get('state', 'India')}. This local report was generated without the Anthropic API because no valid `ANTHROPIC_API_KEY` is available. The main attention areas are {', '.join(alerts + warnings) if alerts or warnings else 'none detected in the measured parameters'}. Crop matching suggests: {best_crops}.

## 2. Detailed Soil Parameter Analysis
{chr(10).join(parameter_lines)}

## 3. Crop Suitability Assessment
The selected crop is **{selected_crop}**. Compare it with the model's strongest matches: {best_crops}. If {selected_crop} is not among the top matches, confirm variety, irrigation availability, and local KVK advice before planting.

## 4. Critical Alerts & Immediate Actions
{('Critical parameters: ' + ', '.join(alerts) + '. Correct these before sowing or before the next fertilizer application.') if alerts else 'No critical soil parameters were detected.'}

## 5. Seasonal Management Plan
For the {soil_data.get('season', 'selected')} season, correct pH and nutrient imbalances first, apply basal fertilizer according to the adjusted recommendation, and split nitrogen applications across crop growth stages where relevant. Keep irrigation consistent and avoid heavy fertilizer application immediately before intense rainfall.

## 6. Long-Term Soil Health Strategy
Add organic matter each season, rotate crops with legumes where practical, and retest soil after harvest. Track pH, EC, N, P, K, organic matter, zinc, and sulphur so recommendations become more accurate over time.

**Fertiliser recommendation:** {fertilizer_text}

References: ICAR-NBSS&LUP fertilizer recommendations, FAO soil guidance, UCI Crop Recommendation Dataset, Soil Health Card Portal GoI.
"""


def _local_management_tips(soil_data: dict[str, Any], crop: str, analysis: dict[str, dict[str, Any]]) -> str:
    issues = [name.replace("_", " ") for name, item in analysis.items() if item["severity"] in ("critical", "warning")]
    issue_text = ", ".join(issues) if issues else "no major measured issues"
    return f"""1. Correct the main soil issues first: {issue_text}. Prioritise critical values before adding routine fertilizer.

2. Apply organic matter through well-decomposed FYM or compost before sowing. This improves nutrient holding capacity and supports better root growth for {crop}.

3. Split nitrogen applications instead of applying all nitrogen at once. This reduces loss and gives the crop nutrients during active growth.

4. Keep pH in the crop-friendly range. If soil is acidic, discuss lime dose with the local agriculture office or KVK before application.

5. Retest the soil after the season. Use the new report to adjust N, P, K, zinc, sulphur, and organic matter doses for the next crop."""


def generate_soil_report(soil_data: dict[str, Any], selected_crop: str,
                          analysis: dict[str, dict[str, Any]],
                          crop_recs: list[dict[str, Any]],
                          fertilizer_rec: dict[str, Any]) -> str:
    _, _, _, metrics = _load_model()

    prompt = f"""You are a senior agronomist and certified soil scientist with expertise in Indian farming systems.
Generate a comprehensive, professional soil health report based on the data below.

FARM PROFILE:
- Location : {soil_data.get('district','N/A')}, {soil_data.get('state','India')}
- Season   : {soil_data.get('season','Kharif')}
- Crop     : {selected_crop}
- Farm Size: {soil_data.get('farm_size','N/A')} acres

SOIL TEST RESULTS:
pH={soil_data.get('ph')}, N={soil_data.get('nitrogen')} kg/ha, P={soil_data.get('phosphorus')} kg/ha,
K={soil_data.get('potassium')} kg/ha, Organic Matter={soil_data.get('organic_matter')}%,
EC={soil_data.get('electrical_conductivity')} dS/m, Zn={soil_data.get('zinc')} ppm,
S={soil_data.get('sulphur')} ppm, Temp={soil_data.get('temperature')}°C,
Humidity={soil_data.get('humidity')}%, Rainfall={soil_data.get('rainfall')} mm/yr

PARAMETER STATUS (FAO + Soil Health Card thresholds):
{json.dumps(analysis, indent=2)}

ML CROP PREDICTION (Random Forest · {metrics['accuracy']*100:.1f}% accuracy · UCI Dataset):
{json.dumps(crop_recs, indent=2)}

ICAR FERTILISER RECOMMENDATION:
{json.dumps(fertilizer_rec, indent=2)}

Write the report in these sections using markdown:
## 1. Executive Summary
Rate overall soil health (Poor / Fair / Good / Excellent) with a 3–4 sentence justification.

## 2. Detailed Soil Parameter Analysis
Interpret each measured parameter, explain what it means for {selected_crop} cultivation.

## 3. Crop Suitability Assessment
Assess {selected_crop} based on soil data. Reference the ML prediction results.

## 4. Critical Alerts & Immediate Actions
Any parameters in Critical range — what to do urgently.

## 5. Seasonal Management Plan
Specific to {soil_data.get('season','Kharif')} season with timeline.

## 6. Long-Term Soil Health Strategy
Sustainable practices for the next 3 seasons.

Cite: ICAR-NBSS&LUP (2022), FAO World Soil Database, UCI Crop Dataset (Ingle 2020), Soil Health Card Portal GoI.
Keep language professional yet understandable by a literate Indian farmer."""

    try:
        msg = _get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        return _message_text(msg) or _local_soil_report(soil_data, selected_crop, analysis, crop_recs, fertilizer_rec)
    except Exception:
        return _local_soil_report(soil_data, selected_crop, analysis, crop_recs, fertilizer_rec)


def generate_management_tips(soil_data: dict[str, Any], crop: str, analysis: dict[str, dict[str, Any]]) -> str:
    issues = [k for k, v in analysis.items() if v["severity"] in ("critical", "warning")]
    prompt = f"""As an ICAR agronomist, give exactly 5 numbered, highly specific soil-management tips
for a farmer in {soil_data.get('state','India')} growing {crop}.
Soil issues: {', '.join(issues) if issues else 'none detected'}.
pH={soil_data.get('ph')}, Season={soil_data.get('season','Kharif')}.
Each tip: 2–3 sentences, mention exact products/doses/timing where possible.
Use simple language. Base on ICAR best practices."""

    try:
        msg = _get_anthropic_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        return _message_text(msg) or _local_management_tips(soil_data, crop, analysis)
    except Exception:
        return _local_management_tips(soil_data, crop, analysis)
