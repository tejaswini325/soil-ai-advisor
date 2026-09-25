import streamlit as st
import pandas as pd
import json
import re
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 AI Soil Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load CSS
# ─────────────────────────────────────────────
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

client = None


def get_groq_client():
    global client

    if client is None:
        api_key = ""

        # Streamlit Cloud secrets
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

        # Local .env fallback
        if not api_key:
            api_key = (
                os.getenv("GROQ_API_KEY", "")
                or os.getenv("GROK_API_KEY", "")
            )

        api_key = api_key.strip().strip('"').strip("'")

        if not api_key or api_key == "groq-your-key-here":
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add your Groq API key "
                "to Streamlit Secrets or the local .env file."
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    return client


def response_text(response):
    """Return text from Groq response."""
    return response.choices[0].message.content or ""


def extract_json_object(text):
    """Extract the first top-level JSON object from a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def escape_control_chars_in_json_strings(text):
    """Escape raw newlines/tabs inside JSON strings while preserving JSON whitespace."""
    result = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            result.append(char)
            escaped = False
            continue

        if char == "\\":
            result.append(char)
            escaped = True
            continue

        if char == '"':
            result.append(char)
            in_string = not in_string
            continue

        if in_string and char == "\n":
            result.append("\\n")
        elif in_string and char == "\r":
            result.append("\\r")
        elif in_string and char == "\t":
            result.append("\\t")
        elif in_string and ord(char) < 32:
            result.append(" ")
        else:
            result.append(char)

    return "".join(result)


def parse_report_json(raw):
    """Parse model JSON with cleanup for common LLM formatting mistakes."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    cleaned = extract_json_object(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = escape_control_chars_in_json_strings(cleaned)
        return json.loads(cleaned)


LANGUAGES = {
    "English": "English",
    "Hindi (हिन्दी)": "Hindi",
    "Tamil (தமிழ்)": "Tamil",
    "Telugu (తెలుగు)": "Telugu",
    "Kannada (ಕನ್ನಡ)": "Kannada",
    "Marathi (मराठी)": "Marathi",
    "Bengali (বাংলা)": "Bengali",
    "Gujarati (ગુજરાતી)": "Gujarati",
    "Punjabi (ਪੰਜਾਬੀ)": "Punjabi",
    "Odia (ଓଡ଼ିଆ)": "Odia",
    "Malayalam (മലയാളം)": "Malayalam",
}

CROPS = [
    "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean",
    "Groundnut", "Sunflower", "Mustard", "Chickpea", "Pigeon Pea",
    "Black Gram", "Green Gram", "Lentil", "Potato", "Tomato",
    "Onion", "Banana", "Mango", "Coconut", "Other"
]

SOIL_TYPES = ["Sandy", "Loamy", "Clay", "Silt", "Sandy Loam", "Clay Loam", "Silty Clay", "Other"]

FERTILISER_PRODUCTS = {
    "Nitrogen (N)": [
        {"name": "Urea (46% N)", "n_content": 0.46, "price_per_kg": 6.5},
        {"name": "Ammonium Sulphate (21% N)", "n_content": 0.21, "price_per_kg": 18},
        {"name": "Calcium Ammonium Nitrate (25% N)", "n_content": 0.25, "price_per_kg": 22},
    ],
    "Phosphorus (P)": [
        {"name": "DAP - Di-Ammonium Phosphate (46% P)", "p_content": 0.46, "price_per_kg": 27},
        {"name": "SSP - Single Super Phosphate (16% P)", "p_content": 0.16, "price_per_kg": 8},
        {"name": "TSP - Triple Super Phosphate (46% P)", "p_content": 0.46, "price_per_kg": 30},
    ],
    "Potassium (K)": [
        {"name": "MOP - Muriate of Potash (60% K)", "k_content": 0.60, "price_per_kg": 17},
        {"name": "SOP - Sulphate of Potash (50% K)", "k_content": 0.50, "price_per_kg": 32},
    ],
    "Organic": [
        {"name": "Vermicompost", "npk": "1.5-0.5-1.0", "price_per_kg": 8},
        {"name": "FYM (Farm Yard Manure)", "npk": "0.5-0.2-0.5", "price_per_kg": 2},
        {"name": "Neem Cake", "npk": "5.0-1.0-1.5", "price_per_kg": 25},
    ]
}

VOCABULARY = {
    "pH": {
        "definition": "Measure of soil acidity or alkalinity on a scale of 0–14. pH 7 is neutral; below 7 is acidic; above 7 is alkaline.",
        "ideal_range": "6.0–7.5 for most crops"
    },
    "NPK": {
        "definition": "Nitrogen (N), Phosphorus (P), and Potassium (K) — the three primary macronutrients essential for plant growth.",
        "ideal_range": "Varies by crop"
    },
    "Organic Carbon (OC)": {
        "definition": "The carbon fraction in organic matter. High OC improves soil structure, water retention, and microbial activity.",
        "ideal_range": "> 0.75% (Medium: 0.5–0.75%, Low: < 0.5%)"
    },
    "EC (Electrical Conductivity)": {
        "definition": "Measures the concentration of salts in the soil. High EC can inhibit plant growth due to osmotic stress.",
        "ideal_range": "< 1.0 dS/m (Non-saline)"
    },
    "CEC (Cation Exchange Capacity)": {
        "definition": "Ability of the soil to hold positively charged nutrient ions. Higher CEC means better nutrient retention.",
        "ideal_range": "10–20 meq/100g (Medium)"
    },
    "Macronutrients": {
        "definition": "Nutrients required in large amounts: Nitrogen (N), Phosphorus (P), Potassium (K), Calcium (Ca), Magnesium (Mg), Sulphur (S).",
        "ideal_range": "N/A"
    },
    "Micronutrients": {
        "definition": "Nutrients required in trace amounts: Zinc (Zn), Iron (Fe), Manganese (Mn), Copper (Cu), Boron (B), Molybdenum (Mo).",
        "ideal_range": "Varies by element"
    },
    "Urea": {
        "definition": "Most commonly used nitrogen fertiliser containing 46% nitrogen. Cheap and widely available.",
        "ideal_range": "N/A"
    },
    "DAP": {
        "definition": "Di-Ammonium Phosphate — granular fertiliser with 18% N and 46% P₂O₅. Good starter fertiliser.",
        "ideal_range": "N/A"
    },
    "MOP": {
        "definition": "Muriate of Potash — potassium chloride with 60% K₂O. Primary source of potassium.",
        "ideal_range": "N/A"
    },
    "SSP": {
        "definition": "Single Super Phosphate — contains 16% P₂O₅ and 11% sulphur. Economical phosphorus source.",
        "ideal_range": "N/A"
    },
    "Soil Texture": {
        "definition": "Proportion of sand, silt, and clay particles. Determines water-holding capacity and aeration.",
        "ideal_range": "Loam or Sandy Loam preferred"
    },
    "Vermicompost": {
        "definition": "Organic fertiliser produced by earthworm decomposition of organic matter. Rich in micronutrients and beneficial microbes.",
        "ideal_range": "N/A"
    },
    "Basal Dose": {
        "definition": "Fertiliser applied to soil before or at the time of sowing/planting to provide nutrients during early growth.",
        "ideal_range": "N/A"
    },
    "Top Dressing": {
        "definition": "Fertiliser applied to the soil surface around growing plants during the crop growth period.",
        "ideal_range": "N/A"
    },
    "Leaching": {
        "definition": "Movement of soluble nutrients (especially nitrates) downward through soil with water, causing nutrient loss.",
        "ideal_range": "N/A"
    },
    "Soil Amendment": {
        "definition": "Any material added to soil to improve its physical or chemical properties — e.g., lime to raise pH, gypsum to improve structure.",
        "ideal_range": "N/A"
    },
}

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
for key, val in {
    "report_generated": False,
    "soil_report": "",
    "fert_data": [],
    "chat_history": [],
    "soil_inputs": {},
    "selected_language": "English",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def safe_float(val):
    """Return float or None for NA values."""
    if val in [None, "NA", "", "N/A"]:
        return None
    try:
        return float(val)
    except Exception:
        return None


def build_soil_summary(inputs):
    """Build a concise text summary of soil inputs for the LLM."""
    lines = []
    field_labels = {
        "crop": "Crop", "soil_type": "Soil Type", "farm_size": "Farm Size (acres)",
        "state": "State/Region", "season": "Season",
        "ph": "pH", "nitrogen": "Nitrogen (kg/ha)", "phosphorus": "Phosphorus (kg/ha)",
        "potassium": "Potassium (kg/ha)", "organic_carbon": "Organic Carbon (%)",
        "ec": "EC (dS/m)", "zinc": "Zinc (ppm)", "iron": "Iron (ppm)",
        "manganese": "Manganese (ppm)", "copper": "Copper (ppm)",
        "boron": "Boron (ppm)", "sulphur": "Sulphur (ppm)",
        "calcium": "Calcium (meq/100g)", "magnesium": "Magnesium (meq/100g)",
    }
    for key, label in field_labels.items():
        v = inputs.get(key)
        if v and v != "NA":
            lines.append(f"{label}: {v}")
    return "\n".join(lines) if lines else "No soil data provided."


def classify_value(value, low, high):
    if value is None:
        return "Not provided"
    if value < low:
        return "Deficient"
    if value > high:
        return "Excess"
    return "Adequate"


def local_report(inputs):
    """Generate a deterministic report when the external AI API is unavailable."""
    crop = inputs.get("crop", "selected crop")
    ph = safe_float(inputs.get("ph"))
    nitrogen = safe_float(inputs.get("nitrogen"))
    phosphorus = safe_float(inputs.get("phosphorus"))
    potassium = safe_float(inputs.get("potassium"))
    organic_carbon = safe_float(inputs.get("organic_carbon"))
    ec = safe_float(inputs.get("ec"))

    ph_status = "Not provided"
    if ph is not None:
        if ph < 5.5:
            ph_status = "Acidic"
        elif ph > 8.0:
            ph_status = "Alkaline"
        else:
            ph_status = "Suitable"

    nutrient_rows = [
        ("Nitrogen", classify_value(nitrogen, 140, 280), 120 if nitrogen is None or nitrogen < 140 else 60 if nitrogen <= 280 else 0),
        ("Phosphorus", classify_value(phosphorus, 10, 25), 60 if phosphorus is None or phosphorus < 10 else 30 if phosphorus <= 25 else 0),
        ("Potassium", classify_value(potassium, 100, 280), 60 if potassium is None or potassium < 100 else 30 if potassium <= 280 else 0),
    ]

    concerns = []
    if ph_status in ("Acidic", "Alkaline"):
        concerns.append(f"Soil pH is {ph_status.lower()}.")
    for nutrient, status, _ in nutrient_rows:
        if status in ("Deficient", "Excess"):
            concerns.append(f"{nutrient} is {status.lower()}.")
    if organic_carbon is not None and organic_carbon < 0.5:
        concerns.append("Organic carbon is low.")
    if ec is not None and ec > 1.0:
        concerns.append("EC is high, indicating possible salinity stress.")

    return {
        "soil_health_report": {
            "overall_rating": "Fair" if concerns else "Good",
            "summary": (
                "This report was generated locally because the configured AI API is unavailable. "
                f"The soil is generally usable for {crop}, but the listed concerns should be corrected before or during cultivation."
            ),
            "ph_analysis": f"pH status: {ph_status}. Most crops perform best near pH 6.0 to 7.5.",
            "macronutrient_analysis": "N, P and K were interpreted using standard soil health threshold ranges.",
            "micronutrient_analysis": "Micronutrient values should be verified with local Soil Health Card or KVK guidance where available.",
            "organic_matter_analysis": "Add FYM, compost or green manure regularly to improve soil structure and nutrient holding capacity.",
            "key_concerns": concerns or ["No major concern detected from the provided values."],
            "management_tips": [
                "Apply fertiliser in split doses instead of one heavy application.",
                "Use well-decomposed FYM or compost before sowing.",
                "Keep irrigation consistent and avoid fertiliser application before heavy rainfall.",
                "Retest soil after harvest to update fertiliser planning.",
                "Confirm final field doses with a local agriculture officer or KVK.",
            ],
        },
        "fertiliser_recommendations": [
            {
                "nutrient": nutrient,
                "status": status,
                "recommended_dose_kg_per_ha": dose,
                "timing": "Basal plus split application where applicable",
                "method": "Broadcast and incorporate into soil; split nitrogen during growth stages",
                "notes": "Local fallback estimate based on soil status thresholds",
            }
            for nutrient, status, dose in nutrient_rows
        ],
        "crop_suitability": f"{crop} suitability should be confirmed with local climate, irrigation and variety selection.",
        "agronomic_references": [
            "ICAR fertiliser recommendation principles",
            "FAO soil health guidance",
            "Soil Health Card scheme norms, Government of India",
        ],
    }


def generate_report(inputs, language):
    lang = LANGUAGES.get(language, "English")
    soil_summary = build_soil_summary(inputs)

    prompt = f"""You are an expert agronomist and soil scientist. Analyze the following soil test data and provide a comprehensive soil health report and fertiliser recommendations.

SOIL DATA:
{soil_summary}

Please respond ENTIRELY in {lang} language (if not English, translate all content including headings).

Provide your response in the following JSON format:
{{
  "soil_health_report": {{
    "overall_rating": "<Excellent/Good/Fair/Poor>",
    "summary": "<2-3 sentence overall assessment>",
    "ph_analysis": "<analysis of pH and its impact>",
    "macronutrient_analysis": "<N, P, K status and issues>",
    "micronutrient_analysis": "<micronutrient status>",
    "organic_matter_analysis": "<organic carbon / soil health>",
    "key_concerns": ["<concern 1>", "<concern 2>"],
    "management_tips": ["<tip 1>", "<tip 2>", "<tip 3>", "<tip 4>", "<tip 5>"]
  }},
  "fertiliser_recommendations": [
    {{
      "nutrient": "<Nutrient name>",
      "status": "<Deficient/Adequate/Excess>",
      "recommended_dose_kg_per_ha": <number or null>,
      "timing": "<When to apply>",
      "method": "<How to apply>",
      "notes": "<Important notes>"
    }}
  ],
  "crop_suitability": "<Assessment of suitability for the specified crop>",
  "agronomic_references": [
    "<Reference 1 from ICAR/FAO/State Agriculture Dept>",
    "<Reference 2>",
    "<Reference 3>"
  ]
}}

Base your recommendations on:
1. ICAR (Indian Council of Agricultural Research) fertiliser dose guidelines
2. FAO World Soil Database norms
3. State Agricultural University recommendations for the region
4. Soil Health Card scheme norms (Government of India)

Only include valid JSON - no markdown fences, no extra text outside the JSON object.
Do not insert raw line breaks inside JSON string values; keep each value as a single JSON-safe string."""

    with st.spinner("🧪 Analyzing soil data..."):
        try:
            response = get_groq_client().chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=4000,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}]
            )
        except Exception as exc:
            st.warning(f"AI API unavailable, using local fallback report. Details: {exc}")
            return local_report(inputs)

    raw = response_text(response)
    try:
        return parse_report_json(raw)
    except Exception as exc:
        st.warning(f"AI response was not valid JSON, using local fallback report. Details: {exc}")
        return local_report(inputs)


def calculate_fertiliser_products(fert_recs, farm_size_acres):
    """Convert nutrient kg/ha recommendations to product quantities and costs."""
    farm_size_ha = farm_size_acres * 0.4047
    results = []

    nutrient_map = {
        "Nitrogen": "Nitrogen (N)",
        "Phosphorus": "Phosphorus (P)",
        "Potassium": "Potassium (K)",
    }

    for rec in fert_recs:
        nutrient_name = rec.get("nutrient", "")
        dose = rec.get("recommended_dose_kg_per_ha")
        if dose is None or rec.get("status") == "Adequate":
            continue

        matched_key = None
        for k in nutrient_map:
            if k.lower() in nutrient_name.lower():
                matched_key = nutrient_map[k]
                break

        if matched_key and matched_key in FERTILISER_PRODUCTS:
            products = FERTILISER_PRODUCTS[matched_key]
            for prod in products[:2]:
                content_key = [k for k in prod if "content" in k]
                if content_key:
                    content = prod[content_key[0]]
                    qty_per_ha = dose / content
                    total_qty = qty_per_ha * farm_size_ha
                    total_cost = total_qty * prod["price_per_kg"]
                    results.append({
                        "Nutrient": nutrient_name,
                        "Product": prod["name"],
                        "Qty/ha (kg)": round(qty_per_ha, 1),
                        "Total Qty (kg)": round(total_qty, 1),
                        "Rate (₹/kg)": prod["price_per_kg"],
                        "Est. Cost (₹)": round(total_cost),
                    })

    return results


def chat_with_advisor(user_msg, history, soil_context, language):
    lang = LANGUAGES.get(language, "English")
    system_prompt = f"""You are a friendly and knowledgeable agricultural advisor (Krishi Sahayak) helping Indian farmers.
You have access to the farmer's soil report and fertiliser recommendations as context.

FARMER'S SOIL & REPORT CONTEXT:
{soil_context}

Rules:
- Always respond in {lang} language
- Use simple, farmer-friendly language (avoid too much technical jargon)
- Be specific and practical in your advice
- If you reference quantities, make them practical (e.g., bags of urea, not just kg)
- Keep responses concise (3-5 sentences usually enough)
- Be warm and encouraging"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=800,
            messages=messages
        )
        return response_text(response)
    except Exception:
        return (
            "The AI chat service is currently unavailable because the configured API key has no access or credits. "
            "Please use the generated soil report and fertiliser table, or configure a valid API key."
        )


def translate_vocab(term_data, language):
    lang = LANGUAGES.get(language, "English")
    if lang == "English":
        return term_data
    prompt = f"""Translate the following agricultural glossary entry to {lang}. 
Return ONLY the translated JSON, no extra text.

{json.dumps(term_data, ensure_ascii=False)}"""
    try:
        response = get_groq_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_report_json(response_text(response))
    except Exception:
        return term_data


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", use_container_width=True)
    st.markdown("## ⚙️ Settings")

    selected_language = st.selectbox(
        "🌐 Select Language / भाषा चुनें",
        list(LANGUAGES.keys()),
        index=0,
        help="All reports, tips, and chatbot responses will be in this language"
    )
    st.session_state.selected_language = selected_language

    st.markdown("---")
    st.markdown("### 📚 Navigation")
    page = st.radio(
        "Go to",
        ["🌱 Soil Analysis", "💊 Fertiliser Calculator", "🤖 AI Chatbot", "📖 Vocabulary"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        """<div class='sidebar-footer'>
        <b>Data Sources</b><br>
        🔬 ICAR-NBSS&LUP<br>
        🌍 FAO World Soil DB<br>
        🇮🇳 Soil Health Card Portal<br>
        📊 UCI Crop Dataset
        </div>""",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# PAGE 1: Soil Analysis
# ─────────────────────────────────────────────
if "Soil Analysis" in page:
    st.markdown("<h1 class='main-title'>🌱 AI-Powered Soil Report Interpreter</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter your soil test values to get a detailed health report and fertiliser recommendations</p>", unsafe_allow_html=True)

    with st.form("soil_form"):
        st.markdown("### 🌾 Basic Farm Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            crop = st.selectbox("Crop to be Grown *", CROPS)
        with col2:
            soil_type_opts = ["NA"] + SOIL_TYPES
            soil_type = st.selectbox("Soil Type", soil_type_opts)
        with col3:
            state = st.text_input("State / Region", placeholder="e.g. Maharashtra, Tamil Nadu")

        col4, col5 = st.columns(2)
        with col4:
            farm_size = st.number_input("Farm Size (acres)", min_value=0.0, value=1.0, step=0.5)
        with col5:
            season_opts = ["Kharif (June-Nov)", "Rabi (Nov-Apr)", "Zaid (Apr-Jun)", "Year-round", "NA"]
            season = st.selectbox("Season", season_opts)

        st.markdown("---")
        st.markdown("### 🧪 Primary Soil Parameters")
        st.caption("💡 Select 'NA' if the value is not available in your soil report")

        col1, col2, col3 = st.columns(3)
        with col1:
            ph_input = st.text_input("pH", value="NA", help="Soil pH (0-14). Typical: 6.0-7.5")
        with col2:
            ec_input = st.text_input("EC (dS/m)", value="NA", help="Electrical Conductivity")
        with col3:
            oc_input = st.text_input("Organic Carbon (%)", value="NA", help="OC%: Low <0.5, Medium 0.5-0.75, High >0.75")

        st.markdown("#### Macronutrients (kg/ha)")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_input = st.text_input("Nitrogen (N)", value="NA")
        with col2:
            p_input = st.text_input("Phosphorus (P)", value="NA")
        with col3:
            k_input = st.text_input("Potassium (K)", value="NA")

        st.markdown("#### Secondary Nutrients (meq/100g)")
        col1, col2 = st.columns(2)
        with col1:
            ca_input = st.text_input("Calcium (Ca)", value="NA")
        with col2:
            mg_input = st.text_input("Magnesium (Mg)", value="NA")

        st.markdown("---")
        with st.expander("🔬 Micronutrients (ppm) — Optional"):
            st.caption("Leave as 'NA' if not tested")
            col1, col2, col3 = st.columns(3)
            with col1:
                zn_input = st.text_input("Zinc (Zn)", value="NA")
                fe_input = st.text_input("Iron (Fe)", value="NA")
            with col2:
                mn_input = st.text_input("Manganese (Mn)", value="NA")
                cu_input = st.text_input("Copper (Cu)", value="NA")
            with col3:
                b_input = st.text_input("Boron (B)", value="NA")
                s_input = st.text_input("Sulphur (S) ppm", value="NA")

        submitted = st.form_submit_button("🔍 Generate Soil Health Report", use_container_width=True, type="primary")

    if submitted:
        inputs = {
            "crop": crop, "soil_type": soil_type if soil_type != "NA" else None,
            "state": state, "farm_size": farm_size,
            "season": season if season != "NA" else None,
            "ph": ph_input, "ec": ec_input, "organic_carbon": oc_input,
            "nitrogen": n_input, "phosphorus": p_input, "potassium": k_input,
            "calcium": ca_input, "magnesium": mg_input,
            "zinc": zn_input, "iron": fe_input,
            "manganese": mn_input, "copper": cu_input,
            "boron": b_input, "sulphur": s_input,
        }
        st.session_state.soil_inputs = inputs

        try:
            result = generate_report(inputs, selected_language)
            st.session_state.soil_report_data = result
            st.session_state.report_generated = True
            farm_size_val = float(farm_size) if farm_size else 1.0
            fert_data = calculate_fertiliser_products(
                result.get("fertiliser_recommendations", []), farm_size_val
            )
            st.session_state.fert_data = fert_data
        except Exception as e:
            st.error(f"Error generating report: {e}")

    if st.session_state.report_generated and "soil_report_data" in st.session_state:
        data = st.session_state.soil_report_data
        report = data.get("soil_health_report", {})
        fert_recs = data.get("fertiliser_recommendations", [])
        references = data.get("agronomic_references", [])
        crop_suitability = data.get("crop_suitability", "")

        st.markdown("---")
        st.markdown("## 📋 Soil Health Report")

        rating = report.get("overall_rating", "Fair")
        rating_colors = {"Excellent": "#2d6a4f", "Good": "#52b788", "Fair": "#f4a261", "Poor": "#e63946"}
        color = rating_colors.get(rating, "#888")
        st.markdown(
            f"<div class='rating-badge' style='background:{color}'>Overall Rating: {rating}</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("**📊 Summary**")
            st.write(report.get("summary", ""))
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='info-card'>", unsafe_allow_html=True)
            st.markdown("**🌾 Crop Suitability**")
            st.write(crop_suitability)
            st.markdown("</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🧪 Analysis", "⚠️ Key Concerns", "💡 Management Tips", "📚 References"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**pH Analysis**")
                st.info(report.get("ph_analysis", ""))
                st.markdown("**Organic Matter**")
                st.info(report.get("organic_matter_analysis", ""))
            with col2:
                st.markdown("**Macronutrients (N-P-K)**")
                st.info(report.get("macronutrient_analysis", ""))
                st.markdown("**Micronutrients**")
                st.info(report.get("micronutrient_analysis", ""))

        with tab2:
            for c in report.get("key_concerns", []):
                st.markdown(f"<div class='concern-item'>⚠️ {c}</div>", unsafe_allow_html=True)

        with tab3:
            for i, tip in enumerate(report.get("management_tips", []), 1):
                st.markdown(f"<div class='tip-item'>✅ **Tip {i}:** {tip}</div>", unsafe_allow_html=True)

        with tab4:
            for ref in references:
                st.markdown(f"📌 {ref}")

        st.markdown("---")
        st.markdown("## 🌿 Fertiliser Recommendations")
        if fert_recs:
            df_fert = pd.DataFrame(fert_recs)
            st.dataframe(
                df_fert,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "recommended_dose_kg_per_ha": st.column_config.NumberColumn("Dose (kg/ha)"),
                }
            )

# ─────────────────────────────────────────────
# PAGE 2: Fertiliser Calculator
# ─────────────────────────────────────────────
elif "Fertiliser Calculator" in page:
    st.markdown("<h1 class='main-title'>💊 Fertiliser Product & Cost Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Convert nutrient recommendations into actual fertiliser products with quantity and cost estimates</p>", unsafe_allow_html=True)

    if not st.session_state.report_generated:
        st.warning("⚠️ Please generate a soil health report first from the **Soil Analysis** page.")
    else:
        data = st.session_state.soil_report_data
        fert_recs = data.get("fertiliser_recommendations", [])
        inputs = st.session_state.soil_inputs
        farm_size = float(inputs.get("farm_size", 1.0))
        farm_size_ha = farm_size * 0.4047

        st.info(f"🌾 Farm: **{farm_size} acres** ({round(farm_size_ha, 2)} ha) | Crop: **{inputs.get('crop', 'N/A')}**")

        col1, col2 = st.columns(2)
        with col1:
            custom_farm_size = st.number_input(
                "Adjust Farm Size (acres)", min_value=0.1, value=farm_size, step=0.5,
                help="Change farm size to recalculate quantities"
            )

        fert_products = calculate_fertiliser_products(fert_recs, custom_farm_size)

        if fert_products:
            df_prod = pd.DataFrame(fert_products)
            st.markdown("### 📦 Recommended Fertiliser Products")
            st.dataframe(df_prod, use_container_width=True, hide_index=True)

            total_cost = df_prod["Est. Cost (₹)"].sum()
            st.markdown(
                f"<div class='cost-summary'>💰 Estimated Total Fertiliser Cost: <b>₹{total_cost:,.0f}</b> for {custom_farm_size} acres</div>",
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Cost Breakdown by Product")
            chart_df = df_prod.set_index("Product")["Est. Cost (₹)"]
            st.bar_chart(chart_df)
        else:
            st.success("✅ No major nutrient deficiencies detected — standard maintenance doses may suffice.")

        st.markdown("---")
        st.markdown("### 🌿 Organic Fertiliser Options")
        org_data = []
        for o in FERTILISER_PRODUCTS["Organic"]:
            qty = custom_farm_size * 0.4047 * 2000
            cost = qty * o["price_per_kg"]
            org_data.append({
                "Product": o["name"], "NPK (%)": o["npk"],
                "Qty for Farm (kg)": round(qty),
                "Rate (₹/kg)": o["price_per_kg"],
                "Est. Cost (₹)": round(cost)
            })
        st.dataframe(pd.DataFrame(org_data), use_container_width=True, hide_index=True)
        st.caption("* Organic quantities based on standard 2 tonnes/ha application rate")

# ─────────────────────────────────────────────
# PAGE 3: AI Chatbot
# ─────────────────────────────────────────────
elif "Chatbot" in page:
    st.markdown("<h1 class='main-title'>🤖 AI Soil Advisory Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Ask any farming or soil-related question in your preferred language</p>", unsafe_allow_html=True)

    if st.session_state.report_generated and "soil_report_data" in st.session_state:
        data = st.session_state.soil_report_data
        inputs = st.session_state.soil_inputs
        soil_context = f"""
SOIL INPUTS:
{build_soil_summary(inputs)}

REPORT SUMMARY:
{data['soil_health_report'].get('summary', '')}

KEY CONCERNS:
{chr(10).join(data['soil_health_report'].get('key_concerns', []))}

FERTILISER RECOMMENDATIONS:
{json.dumps(data.get('fertiliser_recommendations', []), ensure_ascii=False, indent=2)}
"""
        st.success("✅ Your soil report is loaded as context. The chatbot knows your soil data!")
    else:
        soil_context = "No soil report available yet. Answer general agricultural questions."
        st.info("💡 Generate a soil report first for personalized answers. You can still ask general farming questions.")

    st.markdown("#### 💬 Quick Questions:")
    sugg_cols = st.columns(4)
    suggestions = [
        "Why is my nitrogen low?",
        "How much urea should I apply?",
        "Is rice suitable for my soil?",
        "What should I do before sowing?"
    ]
    for i, sugg in enumerate(suggestions):
        if sugg_cols[i].button(sugg, key=f"sugg_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": sugg})
            with st.spinner("Thinking..."):
                reply = chat_with_advisor(sugg, st.session_state.chat_history[:-1], soil_context, selected_language)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>👨‍🌾 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask your question...",
            placeholder="e.g. My soil pH is 5.5, what should I add?",
            key="chat_input",
            label_visibility="collapsed"
        )
    with col2:
        send_btn = st.button("Send 📨", use_container_width=True, type="primary")

    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            reply = chat_with_advisor(user_input, st.session_state.chat_history[:-1], soil_context, selected_language)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# PAGE 4: Vocabulary
# ─────────────────────────────────────────────
elif "Vocabulary" in page:
    st.markdown("<h1 class='main-title'>📖 Agricultural Vocabulary</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Important technical terms explained in simple language</p>", unsafe_allow_html=True)

    search_term = st.text_input("🔍 Search term...", placeholder="e.g. pH, urea, NPK")

    lang = LANGUAGES.get(selected_language, "English")
    translate_now = st.button(
        f"🌐 Translate All to {lang}" if lang != "English" else "Showing in English",
        disabled=(lang == "English"),
        type="primary" if lang != "English" else "secondary"
    )

    filtered_vocab = {
        k: v for k, v in VOCABULARY.items()
        if search_term.lower() in k.lower() or search_term.lower() in v["definition"].lower()
    } if search_term else VOCABULARY

    for term, info in filtered_vocab.items():
        display_info = info
        if translate_now and lang != "English":
            with st.spinner(f"Translating {term}..."):
                display_info = translate_vocab({"definition": info["definition"], "ideal_range": info["ideal_range"]}, selected_language)

        with st.expander(f"📘 {term}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Definition:** {display_info.get('definition', info['definition'])}")
            with col2:
                if info["ideal_range"] != "N/A":
                    st.markdown(
                        f"<div class='range-badge'>✅ Ideal: {display_info.get('ideal_range', info['ideal_range'])}</div>",
                        unsafe_allow_html=True
                    )
