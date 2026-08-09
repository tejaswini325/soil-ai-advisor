# 🌱 AI-Powered Soil Report Interpreter & Fertiliser Recommender

> PS-A3 Hackathon Project — Empowering Indian Farmers with AI

---

## Features

| Feature | Description |
|---|---|
| 🧪 Soil Analysis | Enter soil test values and get a detailed health report |
| 🌐 Regional Language Support | Reports, tips & chatbot in Hindi, Tamil, Telugu, Kannada, Marathi and 7 more languages |
| 💊 Fertiliser Calculator | Convert NPK recommendations to real products (Urea, DAP, MOP, SSP) with cost estimates |
| 🤖 AI Chatbot | Ask follow-up questions in your local language with full soil context |
| 📖 Vocabulary | Glossary of 15+ agricultural terms with regional language translation |
| ✅ NA Support | All input fields support "NA" — only provided values are analysed |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-...

# Windows
set ANTHROPIC_API_KEY=sk-ant-...
```

Or add to `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## Data Sources & References

1. **ICAR-NBSS&LUP** — Crop-wise fertiliser dose tables (nbsslup.icar.gov.in)
2. **FAO World Soil Database** — Global soil property norms (fao.org/soils-portal)
3. **Soil Health Card Portal** — District-wise soil test results (soilhealth.dac.gov.in)
4. **UCI Crop Recommendation Dataset** — 2,200 soil parameter records (kaggle.com)

---

## Tech Stack

- **LLM**: Claude (Anthropic) via API — report generation, chatbot, translations
- **Frontend**: Streamlit
- **Data**: pandas
- **Languages**: Python 3.10+

---

## Supported Languages

English, Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, Punjabi, Odia, Malayalam

---

## Project Structure

```
soil-ai-advisor/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md
└── assets/
    ├── style.css       # Custom UI styles
    └── logo.png        # App logo
```
