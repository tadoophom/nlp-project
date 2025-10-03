"""
app.py – Clinical Keyword Polarity Suite v3.2 (FULL SOURCE)
---------------------------------------------------------------------
• medspaCy ConText negation (falls back to heuristic if unavailable)
• Multi‑keyword batch processing with manual text input and PubMed integration
• Sentence ↔ Dependency toggle, keyword highlight + hover tooltips (also inside tree)
• CSV export, model picker with auto‑download / graceful fallback, optional GPU, NER disabled for speed
• Default sample: 2023 NEJM community‑acquired pneumonia paragraph; default keyword: procalcitonin
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import io, sys, tempfile
import re  # used for regex highlighting

import pandas as pd
import streamlit as st
import spacy
from spacy.language import Language
import plotly.express as px

# Import shared NLP utilities
from nlp_utils import (
    load_pipeline as nlp_load_pipeline,
    extract as nlp_extract,
    render_dependency_svg,
    set_custom_negation_triggers,
    detect_sections as nlp_detect_sections,
    scrub_phi as nlp_scrub_phi,
)

# Feedback database
from database import init_db, insert_feedback, get_feedback_summary

from ui_theme import apply_theme, render_hero, render_stat_cards, section

# ───────────────────────────────────────────────────────────────────────────────
# Lazy optional dependencies
# ───────────────────────────────────────────────────────────────────────────────

def _imp(name):
    try:
        return __import__(name)
    except ModuleNotFoundError:
        return None

medspacy    = _imp("medspacy")           # clinical negation rules

# ───────────────────────────────────────────────────────────────────────────────
# Config – selectable models & default sample
# Extend with multilingual options (Spanish and French) for broader support
# ───────────────────────────────────────────────────────────────────────────────

MODELS = [
    ("en_core_sci_md", "SciSpaCy clinical (install manually)"),
    ("en_core_web_md", "spaCy medium"),
    ("en_core_web_sm", "spaCy small"),
    ("es_core_news_md", "spaCy Spanish"),
    ("fr_core_news_md", "spaCy French"),
]

SAMPLE_TEXT = (
    "In patients in whom viral community-acquired pneumonia is suspected owing "
    "to the identification of a virus (including SARS-CoV-2) by means of a "
    "molecular test and in whom there is no evidence of concurrent bacterial "
    "infection or clinical deterioration, antibacterial treatment can be "
    "discontinued. Most patients have some clinical improvement within 48 to 72 "
    "hours after the start of antibacterial treatment. Intravenous antibiotic "
    "regimens can be transitioned to oral regimens with a similar spectrum "
    "activity as the patient’s condition improves. Duration of therapy: "
    "Typically, patients continue to receive treatment until they have been "
    "afebrile and in a clinically stable condition for at least 48 hours. "
    "Treatment should usually continue for a minimum of 5 days; however, 3 days "
    "may be an adequate treatment duration for certain patients whose condition "
    "is completely stable. Extended courses of therapy may be indicated for "
    "patients with immunocompromising conditions, infections caused by certain "
    "pathogens (e.g., Pseudomonas aeruginosa), or complications such as empyema. "
    "Serial procalcitonin thresholds as an adjunct to clinical judgment may help "
    "guide the discontinuation of antibiotic therapy. Hospital discharge is "
    "appropriate when the patient is in a clinically stable condition, is able "
    "to take oral medication, and has a safe environment for continued care; "
    "overnight observation after a switch to oral therapy is not necessary. "
    "Early discharge based on clinical stability and criteria for the switch to "
    "oral therapy is encouraged to reduce unnecessary hospital costs and risks "
    "associated with hospitalization. Communication and coordination with the "
    "patient’s primary care clinician for early outpatient follow‑up is "
    "encouraged to reduce the likelihood of readmission to the hospital. A "
    "follow‑up chest radiograph is indicated in only a minority of patients, "
    "such as those at risk for lung cancer on the basis of age, smoking history, "
    "or persistence of symptoms."
)

# Richer clinical presets: text + keyword bundles (moved below SAMPLE_TEXT)
PRESET_CASES = {
    "Pneumonia & antibiotics": {
        "text": SAMPLE_TEXT,
        "keywords": [
            "procalcitonin", "bacterial infection", "antibacterial treatment",
            "empyema", "pseudomonas aeruginosa", "oral therapy", "discharge",
        ],
    },
    "Cardiology – ACS & heart failure": {
        "text": (
            "62-year-old male with chest pain radiating to the left arm. "
            + "Troponin elevated, ST-elevation in V2-V4. He denies prior MI. "
            + "No evidence of heart failure decompensation; mild pulmonary edema resolved."
        ),
        "keywords": [
            "myocardial infarction", "st elevation", "troponin", "pulmonary edema",
            "heart failure", "decompensation", "dual antiplatelet therapy",
        ],
    },
    "Oncology – chemo toxicity": {
        "text": (
            "Patient on cisplatin reports nausea and tinnitus. Neutropenia not observed. "
            + "No evidence of metastatic progression on PET-CT. Chemotherapy held due to AKI."
        ),
        "keywords": [
            "cisplatin", "tinnitus", "neutropenia", "metastatic progression",
            "chemotherapy", "aki", "acute kidney injury",
        ],
    },
    "Infectious disease – sepsis": {
        "text": (
            "Patient meets SIRS criteria with suspected source in the urinary tract. "
            + "Lactate elevated; blood cultures pending. No meningismus. Broad-spectrum "
            + "antibiotics initiated; vasopressors not required."
        ),
        "keywords": [
            "sepsis", "sirs", "source control", "lactate", "blood cultures",
            "vasopressors", "broad-spectrum antibiotics",
        ],
    },
    "Neurology – acute ischemic stroke": {
        "text": (
            """
A 74-year-old right-handed patient presented with sudden-onset right facial droop, dense right hemiparesis, and expressive aphasia noted at home at 14:10, with last-known-well (LKW) 90 minutes prior to ED arrival. Initial vitals: BP 176/92, HR 88, RR 18, SpO2 97% RA, glucose 122 mg/dL. NIHSS on arrival was 12, driven by right arm > leg weakness and dysphasia. No seizure activity reported. The patient takes aspirin 81 mg daily and atorvastatin 40 mg; no anticoagulants. There is no history of recent surgery, trauma, or GI bleed.

Non-contrast head CT showed no acute hemorrhage or large territorial infarct early signs; ASPECTS 9. CTA head/neck demonstrated a left MCA M1 segment occlusion with good collateral flow; no cervical dissection or high-grade carotid stenosis. CT perfusion suggested a moderate mismatch with small infarct core. After exclusion of contraindications, tenecteplase was administered in the ED within the treatment time window. Blood pressure was maintained under 180/105 with IV labetalol boluses per protocol.

Following thrombolysis, the patient was transferred urgently to the neurointerventional suite for mechanical thrombectomy. Two passes with a stent retriever yielded TICI 2b reperfusion of the left MCA territory. Post-procedure CT showed no hemorrhagic transformation; the patient was admitted to the neuro-ICU for 24-hour monitoring, permissive hypertension, and DVT prophylaxis. Secondary prevention plans include escalation of statin therapy, cardiac rhythm monitoring to evaluate for occult atrial fibrillation, and echocardiography to assess for cardioembolic sources.
"""
        ),
        "keywords": [
            "nihss", "last known well", "tenecteplase", "alteplase", "thrombectomy",
            "hemorrhagic transformation", "time window",
        ],
    },
    "Endocrinology – DKA vs HHS": {
        "text": (
            """
A 26-year-old with type 1 diabetes presents with 24 hours of polyuria, polydipsia, nausea, and abdominal pain. At triage the patient is tachycardic with Kussmaul respirations and a fruity odor on the breath. Point-of-care glucose is 482 mg/dL. BMP reveals bicarbonate 11 mEq/L with an anion gap of 28; serum beta‑hydroxybutyrate markedly elevated. Serum osmolality is 325 mOsm/kg; corrected sodium 140 mEq/L; potassium 5.4 mEq/L. VBG pH 7.18.

Initial management includes 1–2 L isotonic crystalloid bolus followed by maintenance fluids with electrolyte monitoring. An insulin infusion is initiated after confirming potassium >3.3 mEq/L; potassium chloride added once levels trend downward. Phosphate replacement is deferred initially. Etiology work‑up reveals influenza A positivity; no evidence of UTI or pneumonia on exam and imaging. As the anion gap closes and the patient tolerates PO, a subcutaneous insulin regimen is overlapped for 1–2 hours before stopping the drip. Patient education addresses sick‑day rules, ketone testing, and ensuring access to rapid‑acting insulin pens.
"""
        ),
        "keywords": [
            "dka", "hhs", "anion gap", "beta-hydroxybutyrate", "osmolality",
            "insulin infusion", "potassium repletion",
        ],
    },
    "Nephrology – AKI & CRRT": {
        "text": (
            """
A 68-year-old with septic shock from pneumonia develops oliguric acute kidney injury (urine output <0.3 mL/kg/h for >12 h). Baseline creatinine 0.9 mg/dL, now 4.2 mg/dL with BUN 72 mg/dL and potassium 6.1 mEq/L; bicarbonate 15 mEq/L. KDIGO stage 3 criteria met. Renal ultrasound shows no hydronephrosis. Medication review reveals recent IV contrast and aminoglycoside exposure.

Despite fluid resuscitation and vasopressors, acidosis and hyperkalemia persist. Continuous renal replacement therapy (CRRT) is initiated with citrate anticoagulation given high bleeding risk. Strict intake/output tracking and daily weight are instituted; nephrotoxins are held and antimicrobial dosing adjusted for CRRT clearance. Goals include gradual fluid removal, electrolyte control, and avoidance of rapid shifts. Recovery of renal function will be reassessed daily with spontaneous diuresis and downtrending creatinine.
"""
        ),
        "keywords": [
            "aki", "kdigo", "oliguria", "crrt", "creatinine", "hyperkalemia", "nephrotoxic",
        ],
    },
    "Cardiology – AF anticoagulation": {
        "text": (
            """
A 72-year-old with hypertension and diabetes presents with palpitations and dyspnea. ECG shows atrial fibrillation with rapid ventricular response (HR 138). Transthoracic echo demonstrates preserved LVEF without valvular disease; LA mildly enlarged. CHA2DS2‑VASc score is 4; HAS‑BLED 2. Initial management focuses on rate control with IV diltiazem then oral metoprolol titration; blood pressure remains adequate.

Anticoagulation with apixaban is initiated after discussing risks/benefits. Given persistent symptoms after 48 hours of rate control, rhythm control is considered. Options reviewed include TEE‑guided cardioversion versus delayed cardioversion after 3 weeks of therapeutic anticoagulation; the patient prefers early TEE‑guided approach. Sleep apnea screening and lifestyle modification (alcohol reduction, weight loss) are emphasized to improve long‑term rhythm outcomes.
"""
        ),
        "keywords": [
            "atrial fibrillation", "rvr", "cardioversion", "apixaban", "warfarin",
            "cha2ds2-vasc", "has-bled", "tee",
        ],
    },
    "Pulmonology – ARDS ventilation": {
        "text": (
            """
The patient has moderate ARDS secondary to pneumonia, with PaO2/FiO2 ratio 140 on PEEP 12 and FiO2 0.6. Lung protective ventilation with tidal volume ~6 mL/kg predicted body weight and plateau pressure <30 cmH2O is maintained. Conservative fluid strategy is pursued given hemodynamic stability. A 16‑hour prone positioning session is performed with improvement in oxygenation; skin protection and eye care provided.

If refractory hypoxemia recurs, short‑course neuromuscular blockade will be considered to facilitate ventilator synchrony. Sedation is titrated to RASS targets with daily awakening trials. DVT and stress ulcer prophylaxis are in place. Daily readiness for weaning and spontaneous breathing trials will be assessed as oxygenation improves.
"""
        ),
        "keywords": [
            "ards", "pao2/fio2", "peep", "low tidal volume", "prone positioning",
            "neuromuscular blockade",
        ],
    },
    "Oncology – checkpoint inhibitor toxicity": {
        "text": (
            """
A 58-year-old with metastatic melanoma on pembrolizumab develops 6–8 watery stools/day with abdominal cramping after cycle 3. Infectious workup including C. difficile and enteric pathogen panel is negative. Colonoscopy reveals diffuse erythema and ulcerations consistent with immune‑mediated colitis. CT abdomen/pelvis shows colitis without perforation.

The toxicity is graded as CTCAE grade 3. PD‑1 therapy is held. High‑dose IV methylprednisolone is initiated with transition to oral prednisone taper over 6–8 weeks as symptoms improve. Given partial response after 72 hours, a single dose of infliximab is administered with close monitoring. The oncology team will reassess resumption versus switch of therapy after complete resolution and risk‑benefit discussion.
"""
        ),
        "keywords": [
            "pembrolizumab", "nivolumab", "immune-mediated colitis", "steroids", "infliximab",
            "checkpoint inhibitor",
        ],
    },
    "Heme/Onc – neutropenic fever": {
        "text": (
            "ANC 300 with fever 38.5°C after chemotherapy. Broad-spectrum antibiotics with pseudomonas coverage started; "
            + "blood cultures drawn; consider G-CSF support. No focal source on exam."
        ),
        "keywords": [
            "neutropenic fever", "anc", "pseudomonas coverage", "g-csf", "broad-spectrum antibiotics",
        ],
    },
    "ID – tuberculosis management": {
        "text": (
            "Quantiferon positive; CXR with apical cavitation. Sputum AFB smear pending. Start RIPE therapy; "
            + "monitor for hepatotoxicity and neuropathy; counsel on adherence and isolation."
        ),
        "keywords": [
            "latent tuberculosis", "rifampin", "isoniazid", "pyrazinamide", "ethambutol",
            "hepatotoxicity", "quantiferon", "afb smear",
        ],
    },
    "Obstetrics – preeclampsia": {
        "text": (
            "G2P1 at 35 weeks with severe-range blood pressures, headache, and RUQ pain. Proteinuria present; "
            + "magnesium sulfate for seizure prophylaxis and IV labetalol given; evaluate for HELLP."
        ),
        "keywords": [
            "preeclampsia", "severe features", "proteinuria", "magnesium sulfate", "labetalol", "hellp",
        ],
    },
}

# Persist counters for analytics
if "_doc_count" not in st.session_state:
    st.session_state._doc_count = 0
if "_hit_history" not in st.session_state:
    st.session_state._hit_history = []  # type: List[dict]

# Initialise database tables on module import
init_db()

st.set_page_config(page_title="Clinical Keyword Polarity", layout="wide")
apply_theme()

render_hero(
    "Clinical Keyword Polarity",
    "",
    tags=None,
)

render_stat_cards([
    {
        "label": "Session analyses",
        "value": st.session_state.get("_doc_count", 0),
        "description": "Documents processed in this Streamlit session",
    },
    {
        "label": "Available models",
        "value": len(MODELS),
        "description": "spaCy / SciSpaCy pipelines ready to load",
    },
    {
        "label": "Preset cases",
        "value": len(PRESET_CASES),
        "description": "Curated scenarios to explore quickly",
    },
])

# ───────────────────────────────────────────────────────────────────────────────
# Sidebar UI
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # URL params
    _qp = st.query_params if hasattr(st, "query_params") else {}

    # Import PubMed functions
    try:
        from pubmed_fetch import search_pubmed, fetch_abstracts
    except ImportError:
        search_pubmed = fetch_abstracts = None

    st.header("Analysis Configuration")

    # Model selection
    _available_models = [m for m, _ in MODELS]
    _url_m = _qp.get("m") if hasattr(_qp, "get") else None
    _model_defaults = [x for x in (_url_m.split(",") if _url_m else []) if x in _available_models] or ["en_core_web_sm"]
    model_choices = st.multiselect(
        "spaCy model(s)", _available_models,
        default=_model_defaults,
        format_func=lambda x: dict(MODELS)[x],
        max_selections=3,
    )

    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        use_gpu = st.checkbox("Use GPU", value=False)
        expand_syn = st.checkbox("Expand synonyms", value=False, help="WordNet synonyms")
        show_codes = st.checkbox("Show clinical codes", value=True)
    with col2:
        detect_sections = st.checkbox("Detect sections", value=True)
        show_temporal = st.checkbox("Extract temporal cues", value=True)
        scrub_phi = st.checkbox("Scrub PHI", value=False)

    # Keyword configuration
    with st.expander("Keyword Configuration", expanded=False):
        # Prepare color palette state
        if "_kw_colors" not in st.session_state:
            st.session_state._kw_colors = {}

        # Use the global preset for default keywords
        current_preset = st.session_state.get("global_preset", "Pneumonia & antibiotics")
        if current_preset and current_preset in PRESET_CASES:
            default_kw = ", ".join(PRESET_CASES[current_preset]["keywords"])[:200]
        else:
            default_kw = "procalcitonin"

        # URL param support for keywords (?k=fever,cough)
        url_kw = _qp.get("k") if hasattr(_qp, "get") else None
        raw_kw_default = url_kw if url_kw else default_kw
        raw_kw = st.text_input("Keywords (comma)", raw_kw_default)
        keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]

        # Seed distinct default colors per keyword
        _palette = [
            "#EF476F", "#06D6A0", "#118AB2", "#FFD166", "#8338EC",
            "#2EC4B6", "#FF9F1C", "#8AC926", "#FF595E", "#1982C4",
        ]
        for i, k in enumerate(sorted(set(keywords))[:16]):
            st.session_state._kw_colors.setdefault(k, _palette[i % len(_palette)])

        # Custom rules
        st.subheader("Custom Rules")
        custom_neg = st.text_area("Negation triggers (comma)", placeholder="no, denies, without", height=60)
        _trigs = [t.strip() for t in custom_neg.split(",") if t.strip()]
        st.session_state["custom_neg_triggers"] = _trigs
        set_custom_negation_triggers(_trigs)

    # Theme toggle
    st.header("Appearance")
    dark_mode = st.toggle("Dark mode", value=False, help="Toggle between light and dark themes")
    if dark_mode:
        st.write(
            """
            <style>
                body, .stApp { background-color: #0b1b2b; color: #e0e0e0; }
                .st-eb { color: #e0e0e0; }
                .tbl td { color: #e0e0e0; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.caption("ConText negation used when medspaCy installed; otherwise heuristic.")

# ───────────────────────────────────────────────────────────────────────────────
# Model loader with auto‑download & graceful fallback
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_resource(hash_funcs={Language: id})
def get_pipeline(name: str, gpu: bool) -> Language:
    """Wrapper around nlp_utils.load_pipeline with Streamlit caching."""
    # Streamlit spinner for potentially long model download
    with st.spinner(f"Loading model {name} …"):
        return nlp_load_pipeline(name, gpu=gpu)

COLOUR = {"Positive": "#bef5cb", "Negative": "#ffb3b3", "Neutral": "#ffe5b4"}

def highlight(row):
    colour = COLOUR[row.Classification]
    tip = f"{row.Keyword}|{row.POS}|{row.Dep}|{row.Classification}"
    pat = rf"\\b({re.escape(row.Keyword)})\\b"
    return re.sub(pat,
        rf"<mark style='background:{colour};' title='{tip}'>\\1</mark>",
        row.Sentence, flags=re.I)

# Add comparison of different models
# Add table of metrics 
# Add neutral relation / missing relationship
# More Visualization of results
# Different colors to describe different sentiments
# use colors to highlight terms of interest 



# ───────────────────────────────────────────────────────────────────────────────
# File ingestion helpers
# ───────────────────────────────────────────────────────────────────────────────
# Obtain text source - moved above analysis section
# ───────────────────────────────────────────────────────────────────────────────

# Text source is already obtained in the sidebar above as 'raw_text'

# Create a main content area for text input
with section("1 · Document Text", "Bring in your own clinical note or combine it with imported PubMed articles."):

    # Preset selector and text input
    preset = st.selectbox("Preset case", list(PRESET_CASES.keys()), key="global_preset")

    # Button to load preset text
    if st.button("Load preset text", help="Load the sample text for the selected preset case"):
        st.session_state.preset_text_loaded = PRESET_CASES.get(preset, {}).get("text", "")

    # Initialize preset text if not in session state
    if "preset_text_loaded" not in st.session_state:
        st.session_state.preset_text_loaded = ""

    # Get text source early so we can use it for keyword processing
    def get_text() -> str:
        # Show empty text area by default, unless PubMed articles are available
        if not st.session_state.get("pubmed_export_text"):
            return st.text_area(
                "Document text",
                st.session_state.preset_text_loaded,
                height=220,
                placeholder="Paste your text here, load preset text, or import from PubMed search...",
                label_visibility="collapsed",
            )
        else:
            return ""  # Empty string when PubMed articles are available

    raw_text = get_text()

    st.subheader("Text source handling")

    # Check if PubMed articles are included with improved export handling
    if "pubmed_export_text" in st.session_state and st.session_state.pubmed_export_text:
        selected_count = len(st.session_state.get("selected_articles_for_analysis", []))
        export_format = st.session_state.get('pubmed_export_format', 'Title + Abstract')

        st.success(f"{selected_count} PubMed article(s) imported using '{export_format}' format")

        # Options for handling PubMed text
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            include_pubmed = st.checkbox("Include PubMed articles", value=True, key="include_pubmed_main")
        with col2:
            if st.button("Preview PubMed Text"):
                st.session_state.show_pubmed_detail = not st.session_state.get('show_pubmed_detail', False)
        with col3:
            if st.button("Refresh"):
                st.rerun()
        with col4:
            if st.button("Clear PubMed", help="Remove imported articles"):
                for key in ['pubmed_export_text', 'selected_articles_for_analysis', 'pubmed_export_format', 'show_pubmed_detail']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Show detailed PubMed text if requested
        if st.session_state.get('show_pubmed_detail', False):
            with st.expander("PubMed Export Details", expanded=True):
                st.text_area("Imported PubMed text:", value=st.session_state.pubmed_export_text, height=220, disabled=True)

        # Determine what text to display
        if include_pubmed:
            if raw_text.strip():  # User has uploaded/entered text
                display_text = raw_text + "\n\n" + st.session_state.pubmed_export_text
                text_source = "Combined: Your text + PubMed articles"
            else:  # Only PubMed articles
                display_text = st.session_state.pubmed_export_text
                text_source = "PubMed articles only"
        else:
            display_text = raw_text if raw_text.strip() else ""
            text_source = "Your text only" if raw_text.strip() else "No text selected"
    else:
        include_pubmed = False
        display_text = raw_text if raw_text.strip() else ""
        if raw_text.strip():
            text_source = "Your uploaded/entered text"
        else:
            text_source = "No text available - please import from PubMed or use preset cases"

    # Show text source info
    st.info(f"Text source: {text_source}")

    # Set final_text for analysis
    if display_text:
        final_text = display_text
    else:
        final_text = raw_text if raw_text.strip() else ""

# ───────────────────────────────────────────────────────────────────────────────
# Keyword Configuration - Main Section
# ───────────────────────────────────────────────────────────────────────────────

with section("2 · Keywords to Analyze", "Curate the terminology, colours, and negation rules for this run."):

    # Prepare color palette state
    if "_kw_colors" not in st.session_state:
        st.session_state._kw_colors = {}

    # Use the global preset for default keywords
    current_preset = st.session_state.get("global_preset", "Pneumonia & antibiotics")
    if current_preset and current_preset in PRESET_CASES:
        default_kw = ", ".join(PRESET_CASES[current_preset]["keywords"])[:200]
    else:
        default_kw = "procalcitonin"

    # URL param support for keywords (?k=fever,cough)
    url_kw = _qp.get("k") if hasattr(_qp, "get") else None
    raw_kw_default = url_kw if url_kw else default_kw

    # Main keyword input - prominent and easy to find
    raw_kw = st.text_input(
        "Enter keywords to search for (comma-separated):",
        value=raw_kw_default,
        placeholder="e.g., fever, cough, pneumonia, antibiotic",
        help="Enter the medical terms you want to analyze in your text"
    )

    keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]

    if keywords:
        st.success(f"Will analyze {len(keywords)} keyword(s): {', '.join(keywords)}")
    else:
        st.warning("Please enter at least one keyword to analyze")

    # Advanced keyword options
    with st.expander("Advanced Keyword Options", expanded=False):
        # Seed distinct default colors per keyword
        _palette = [
            "#EF476F", "#06D6A0", "#118AB2", "#FFD166", "#8338EC",
            "#2EC4B6", "#FF9F1C", "#8AC926", "#FF595E", "#1982C4",
        ]
        for i, k in enumerate(sorted(set(keywords))[:16]):
            st.session_state._kw_colors.setdefault(k, _palette[i % len(_palette)])

        # Show keyword colors if there are keywords
        if keywords:
            st.subheader("Keyword Colors")
            col1, col2 = st.columns(2)
            for i, k in enumerate(sorted(set(keywords))[:8]):
                with col1 if i % 2 == 0 else col2:
                    st.session_state._kw_colors[k] = st.color_picker(f"Color for '{k}'", st.session_state._kw_colors.get(k))

        # Custom negation rules
        st.subheader("Custom Negation Rules")
        custom_neg = st.text_area(
            "Custom negation triggers (comma-separated):",
            placeholder="no, denies, without, negative for",
            height=60,
            help="Add custom words that indicate negation in medical text"
        )
        _trigs2 = [t.strip() for t in custom_neg.split(",") if t.strip()]
        st.session_state["custom_neg_triggers"] = _trigs2
        set_custom_negation_triggers(_trigs2)

    if scrub_phi:
        with st.expander("PHI scrubbing preview", expanded=False):
            redacted = nlp_scrub_phi(final_text)
            st.code(redacted)
        final_text = nlp_scrub_phi(final_text)

# ───────────────────────────────────────────────────────────────────────────────
# Extract keyword occurrences
# ───────────────────────────────────────────────────────────────────────────────

# NOTE: The extraction logic has moved to nlp_utils.extract. This wrapper is
# retained for backwards compatibility and to capture feedback for analytics.
def extract(text: str, terms: List[str]):
    raw_results = nlp_extract(text, terms, nlp)
    # Record history for analytics
    st.session_state._hit_history.extend(raw_results)
    # Translate result keys back to the UI's expected format
    translated: List[dict] = []
    for r in raw_results:
        translated.append(
            {
                "Keyword": r.get("keyword"),
                "Sentence": r.get("sentence"),
                "POS": r.get("pos"),
                "Dep": r.get("dep"),
                "Classification": r.get("classification"),
                "SentenceIndex": r.get("sent_index"),
                "TokenIndex": r.get("token_index"),
            }
        )
    return translated

# ───────────────────────────────────────────────────────────────────────────────
# Patch displaCy SVG with token tooltips
# ───────────────────────────────────────────────────────────────────────────────

def patched_svg(doc: Language):  # type: ignore[override]
    """Deprecated wrapper retained for compatibility. Use nlp_utils.render_dependency_svg."""
    return render_dependency_svg(doc.text, nlp)

# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

if st.button("Analyze", use_container_width=True):
    if not keywords:
        st.error("Enter keyword(s)")
        st.stop()

    # Use the final text (which may include PubMed articles and user edits)
    analysis_text = final_text

    if not analysis_text.strip():
        st.error("Text is empty")
        st.stop()

    results = []
    # Optional synonym expansion
    syn_terms = list(set(keywords))
    if expand_syn:
        try:
            from nltk.corpus import wordnet as wn  # type: ignore
            synset = set(syn_terms)
            for term in keywords:
                for s in wn.synsets(term):
                    for l in s.lemma_names():
                        synset.add(l.replace("_", " ").lower())
            syn_terms = sorted(synset)
        except Exception:
            syn_terms = keywords
    for mdl in model_choices:
        nlp = get_pipeline(mdl, use_gpu)
        results.extend(
            nlp_extract(analysis_text, syn_terms, nlp, model_name=mdl)
        )
    if not results:
        st.warning("No occurrences found.")
        st.stop()

    # persist in-session history for analytics
    st.session_state._hit_history.extend(results)

    df = pd.DataFrame(results)
    st.session_state._doc_count += 1

    # Optional: section detection
    if detect_sections:
        headers = nlp_detect_sections(analysis_text) + [{"start": len(analysis_text) + 1, "title": "END"}]
        def locate_section(s: str) -> str:
            pos = analysis_text.find(s[:40])
            if pos < 0:
                return "Unknown"
            prev = "Unknown"
            for h in headers:
                if pos < h["start"]:
                    return prev
                prev = h["title"]
            return prev
        df["section"] = df["sentence"].map(locate_section)

    # Optional: clinical codes (mini dictionary)
    CODE_MAP = {
        "procalcitonin": ("SNOMED", "704427003"),
        "myocardial infarction": ("SNOMED", "22298006"),
        "sepsis": ("SNOMED", "91302008"),
        "pseudomonas aeruginosa": ("SNOMED", "16814004"),
        "neutropenia": ("SNOMED", "165816005"),
        "cisplatin": ("RxNorm", "2551"),
        "fever": ("SNOMED", "386661006"),
        "cough": ("SNOMED", "49727002"),
    }
    if show_codes:
        df["code_system"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[0])
        df["code"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[1])

    # Optional: temporal cues per sentence
    def extract_temporal(text: str) -> str:
        if not show_temporal:
            return ""
        pat = re.compile(r"\b(?:today|yesterday|tomorrow|last\s+\w+|next\s+\w+|\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+(?:ago|prior))\b", re.I)
        return ", ".join(sorted(set(m.group(0) for m in pat.finditer(text))))
    if show_temporal:
        df["temporal"] = df["sentence"].map(extract_temporal)

    # Filters
    filt_cols = st.columns(2)
    with filt_cols[0]:
        sel_keywords = st.multiselect("Filter keywords", sorted(df["keyword"].unique().tolist()), default=sorted(df["keyword"].unique().tolist()))
    with filt_cols[1]:
        sel_classes = st.multiselect("Filter classes", ["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
    df = df[df["keyword"].isin(sel_keywords) & df["classification"].isin(sel_classes)].reset_index(drop=True)

    # Metrics table + stacked bar
    pivot = (
        df.pivot_table(index="model", columns="classification",
                       values="keyword", aggfunc="count", fill_value=0)
          .assign(Total=lambda t: t.sum(1))
          .sort_index()
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    st.subheader("Polarity metrics")
    st.dataframe(
        pivot.style.format().background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
    )
    fig = px.bar(
        pivot.reset_index().melt("model", value_vars=["Positive","Neutral","Negative"], var_name="Classification", value_name="value"),
        x="model", y="value", color="Classification",
        title="Sentiment distribution per model",
        text="value",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Co-occurrence heatmap (keyword x classification)
    st.subheader("Co-occurrence heatmap")
    heat = (
        df.value_counts(["keyword", "classification"]).rename("count").reset_index()
        .pivot(index="keyword", columns="classification", values="count").fillna(0)
    )
    heat = heat.reindex(sorted(heat.index), axis=0).reindex(["Positive","Neutral","Negative"], axis=1, fill_value=0)
    st.dataframe(heat.style.background_gradient(cmap="OrRd"), use_container_width=True)

    # Export functionality
    st.subheader("Export Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export detailed results
        csv_detailed = df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Results (CSV)",
            data=csv_detailed,
            file_name="keyword_analysis_detailed.csv",
            mime="text/csv",
            help="All individual keyword occurrences with classifications"
        )
    
    with col2:
        # Export co-occurrence matrix
        csv_cooccurrence = heat.to_csv()
        st.download_button(
            label="Download Co-occurrence Matrix (CSV)",
            data=csv_cooccurrence,
            file_name="keyword_cooccurrence_matrix.csv",
            mime="text/csv",
            help="Keyword vs classification count matrix"
        )
    
    with col3:
        # Export summary statistics
        summary_stats = pivot.copy()
        csv_summary = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Stats (CSV)",
            data=csv_summary,
            file_name="keyword_analysis_summary.csv",
            mime="text/csv",
            help="Summary statistics by model"
        )
    
    # Store results for manual review
    st.session_state.analysis_results = df.copy()
    st.session_state.cooccurrence_matrix = heat.copy()
    
    # Manual review button
    if st.button("Open Manual Review", type="primary"):
        st.switch_page("pages/manual_review.py")

    # Overview table with highlight & tooltips (adapted to lowercase keys)
    COLOUR = {"Positive": "#bef5cb", "Negative": "#ffb3b3", "Neutral": "#ffe5b4"}
    def highlight_row(row):
        base_colour = COLOUR[row.classification]
        kw = str(row.keyword)
        kw_col = st.session_state._kw_colors.get(kw.lower(), base_colour)
        tip = f"{kw}|{row.pos}|{row.dep}|{row.classification}|{row.get('confidence','')}"
        pat = re.compile(rf"(?<!\w)({re.escape(kw)})(?!\w)", re.I)
        return pat.sub(lambda m: f"<mark style='background:{kw_col};' title='{tip}'>{m.group(1)}</mark>", row.sentence)

    # Ensure highlight is visible across themes
    if not st.session_state.get("_mark_css"):
        st.markdown(
            """
            <style>
              mark { color: #111 !important; padding: 0 2px; border-radius: 3px; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.15); }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["_mark_css"] = True

    # Color legend
    if st.session_state._kw_colors:
        legend = " ".join(
            f"<span style='display:inline-block;margin:0 8px 6px 0;'><span style='display:inline-block;width:12px;height:12px;background:{c};border:1px solid #999;margin-right:6px;vertical-align:middle;'></span><span style='vertical-align:middle'>{k}</span></span>"
            for k, c in st.session_state._kw_colors.items()
        )
        st.markdown(f"<div>{legend}</div>", unsafe_allow_html=True)

    df_disp = df.assign(sentence=df.apply(highlight_row, axis=1))
    display_cols = {"sentence": "Sentence", "keyword": "Keyword", "classification": "Classification", "model": "Model", "confidence": "Confidence"}
    if detect_sections:
        display_cols["section"] = "Section"
    if show_codes:
        display_cols["code_system"] = "CodeSystem"
        display_cols["code"] = "Code"
    if show_temporal:
        display_cols["temporal"] = "Temporal"
    st.markdown(df_disp.rename(columns=display_cols)
                .to_html(escape=False, index=False, classes="tbl"), unsafe_allow_html=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"), "hits.csv", mime="text/csv")
    st.download_button("Download JSON", df.to_json(orient="records").encode("utf-8"), "hits.json", mime="application/json")
    # FHIR Bundle export (minimal Observation/Condition payload per hit)
    def to_fhir(rec: dict) -> dict:
        code = rec.get("code")
        system = rec.get("code_system")
        resource_type = "Observation" if rec.get("classification") != "Negative" else "Condition"
        coding = [{"system": f"http://{system.lower()}.org", "code": code}] if code and system else []
        return {
            "resourceType": resource_type,
            "code": {"coding": coding, "text": rec.get("keyword")},
            "valueString": rec.get("sentence"),
            "note": [{"text": f"classification={rec.get('classification')} confidence={rec.get('confidence')}"}],
        }
    fhir_bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": to_fhir(r)} for r in df.to_dict(orient="records")]}
    st.download_button("Download FHIR Bundle (JSON)", __import__("json").dumps(fhir_bundle, ensure_ascii=False, indent=2).encode("utf-8"), "bundle.json", mime="application/json")

    # Detail view
    st.subheader("Context detail")
    for i, row in df.iterrows():
        with st.expander(f"{row['keyword']} – {row['classification']} [{row.get('model','n/a')}]", expanded=False):
            if detect_sections:
                st.caption(f"Section: {row.get('section','Unknown')}")
            if show_temporal and row.get("temporal"):
                st.caption(f"Temporal cues: {row['temporal']}")
            # Context window view (+/- one sentence)
            try:
                mdl = row.get("model", model_choices[0])
                nlp_ctx = get_pipeline(mdl, use_gpu)
                doc_ctx = nlp_ctx(analysis_text)
                # naive: find current sentence and show neighbors
                sidx = next((i for i, s in enumerate(doc_ctx.sents) if s.text.strip() == row["sentence"].strip()), None)
                if sidx is not None:
                    prev_s = list(doc_ctx.sents)[max(0, sidx-1)].text if sidx-1 >= 0 else ""
                    next_s = list(doc_ctx.sents)[min(len(list(doc_ctx.sents))-1, sidx+1)].text if sidx+1 < len(list(doc_ctx.sents)) else ""
                    st.caption("Context window")
                    st.write(prev_s)
                    st.markdown(highlight_row(row), unsafe_allow_html=True)
                    st.write(next_s)
            except Exception:
                st.markdown(highlight_row(row), unsafe_allow_html=True)
            nlp = get_pipeline(row.get("model", model_choices[0]), use_gpu)
            if "parser" not in nlp.pipe_names:
                st.info(
                    "Dependency graphs unavailable: selected model has no parser. "
                    "Install a model with parser and select it, e.g. run: \n\n"
                    "uv add \"en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl\""
                )
            else:
                doc = nlp(row["sentence"])
                svg = patched_svg(doc)
                cid = f"dep_view_{i}"
                html = f"""
<style>
.viewer {{
  border:1px solid #e0e0e0; background:white; width:100%; height:520px; position:relative; overflow:auto;
}}
.viewer .toolbar {{
  position:absolute; top:8px; right:8px; display:flex; gap:6px; z-index:2;
}}
.viewer .toolbar button, .viewer .toolbar input, .viewer .toolbar a {{
  font-size:12px; padding:6px 10px; border:1px solid #bbb; background:#fafafa; cursor:pointer; border-radius:6px;
}}
.viewer .content {{
  display:inline-block; transform-origin: top left; user-select:none; -webkit-user-drag:none;
}}
</style>
<div id="{cid}" class="viewer">
  <div class="toolbar">
    <button data-act="zoom-out">−</button>
    <button data-act="zoom-in">+</button>
    <button data-act="reset">Reset</button>
    <label style="margin-left:6px;">H: <input type="range" min="240" max="1200" value="520" step="10" data-act="height" /></label>
    <a href="#" data-act="save-svg">SVG</a>
    <a href="#" data-act="save-png">PNG</a>
  </div>
  <div class="content">{svg}</div>
</div>
<div style="margin-top:10px;">
  <strong>PNG preview</strong>
  <div style="border:1px solid #e0e0e0; background:white; padding:6px;">
    <img id="png_{i}" alt="Dependency PNG preview" style="max-width:100%; height:auto; display:block;" />
  </div>
</div>
<script>
(function(){{
  const root = document.getElementById('{cid}');
  const content = root.querySelector('.content');
  const toolbar = root.querySelector('.toolbar');
  const pngEl = document.getElementById('png_{i}');
  let scale = 1.0;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function applyTransform(){{ content.style.transform = `scale(${{scale}})`; }}
  function zoomAt(px, py, factor){{
    const prev = scale; scale = clamp(scale * factor, 0.5, 2.5); const ratio = scale/prev;
    const rect = root.getBoundingClientRect();
    const mx = px - rect.left; const my = py - rect.top;
    root.scrollLeft = (mx + root.scrollLeft) * ratio - mx;
    root.scrollTop  = (my + root.scrollTop) * ratio - my;
    applyTransform();
  }}
  function renderPng(){{
    const svg = content.querySelector('svg');
    if(!svg) return;
    const xml = new XMLSerializer().serializeToString(svg);
    const img = new Image(); const can = document.createElement('canvas'); const ctx = can.getContext('2d');
    const bb = svg.viewBox && svg.viewBox.baseVal ? svg.viewBox.baseVal : null;
    const w = bb && bb.width ? bb.width : svg.clientWidth || 1200;
    const h = bb && bb.height ? bb.height : svg.clientHeight || 400;
    can.width = w; can.height = h;
    img.onload = () => {{ ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,w,h); ctx.drawImage(img,0,0); pngEl.src = can.toDataURL('image/png'); }};
    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
  }}
  root.addEventListener('wheel', (e) => {{ e.preventDefault(); const f = e.deltaY < 0 ? 1.1 : 0.9; zoomAt(e.clientX, e.clientY, f); }}, {{passive:false}});
  let drag=false, lx=0, ly=0; 
  root.addEventListener('mousedown', (e) => {{ if(e.target.closest('.toolbar')) return; drag=true; lx=e.clientX; ly=e.clientY; e.preventDefault(); }});
  window.addEventListener('mouseup', () => drag=false);
  window.addEventListener('mousemove', (e) => {{ if(!drag) return; const dx=e.clientX-lx, dy=e.clientY-ly; root.scrollLeft -= dx; root.scrollTop -= dy; lx=e.clientX; ly=e.clientY; }});
  toolbar.addEventListener('click', (e) => {{
    const act = e.target.getAttribute('data-act'); if(!act) return;
    e.preventDefault();
    if(act==='zoom-in') zoomAt(root.clientWidth/2, root.clientHeight/2, 1.1);
    if(act==='zoom-out') zoomAt(root.clientWidth/2, root.clientHeight/2, 0.9);
    if(act==='reset') {{ scale=1.0; applyTransform(); root.scrollLeft=0; root.scrollTop=0; }}
    if(act==='save-svg') {{
      const svg = content.querySelector('svg'); const blob = new Blob([svg.outerHTML], {{type:'image/svg+xml'}});
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'dependency_{i}.svg'; a.click(); URL.revokeObjectURL(a.href);
    }}
    if(act==='save-png') {{
      const svg = content.querySelector('svg'); const xml = new XMLSerializer().serializeToString(svg);
      const img = new Image(); const can = document.createElement('canvas'); const ctx = can.getContext('2d');
      const bb = svg.viewBox.baseVal; const w = bb && bb.width ? bb.width : svg.clientWidth; const h = bb && bb.height ? bb.height : svg.clientHeight;
      can.width = w; can.height = h; img.onload = () => {{ ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,w,h); ctx.drawImage(img,0,0); const a = document.createElement('a'); a.download='dependency_{i}.png'; a.href = can.toDataURL('image/png'); a.click(); }};
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);
    }}
  }});
  toolbar.addEventListener('input', (e) => {{ if(e.target.getAttribute('data-act')==='height') root.style.height = e.target.value + 'px'; }});
  applyTransform();
  renderPng();
 }})();
</script>
"""
                st.components.v1.html(html, height=540, scrolling=False)

            col1, col2 = st.columns(2)
            if col1.button("Mark correct", key=f"correct_{i}"):
                insert_feedback(
                    keyword=row["keyword"],
                    sentence=row["sentence"],
                    classification=row["classification"],
                    correct_label=True,
                )
                st.success("Feedback recorded as correct")
            if col2.button("Mark incorrect", key=f"incorrect_{i}"):
                insert_feedback(
                    keyword=row["keyword"],
                    sentence=row["sentence"],
                    classification=row["classification"],
                    correct_label=False,
                )
                st.success("Feedback recorded as incorrect")

    # Missing-relationship view (updated keys)
    st.subheader("Sentences without classified polarity")
    nlp_first = get_pipeline(model_choices[0], use_gpu)
    miss = nlp_first(analysis_text)
    miss_sent = [
        s.text for s in miss.sents
        if any(k in s.text.lower() for k in keywords)
           and not any(s.text == r["sentence"] for r in results)
    ]
    if miss_sent:
        with st.expander(f"{len(miss_sent)} unmatched sentences"):
            for s in miss_sent:
                st.markdown(s)
    else:
        st.caption("All keyword sentences classified.")

    # Model disagreement view
    if df["model"].nunique() > 1:
        st.subheader("Model disagreement")
        disag = (
            df.groupby(["sentence", "keyword"]).agg(
                classes=("classification", lambda x: ", ".join(sorted(set(x))))
            ).reset_index()
        )
        disag = disag[disag["classes"].str.contains(",")]
        if not disag.empty:
            st.dataframe(disag, use_container_width=True)
        else:
            st.caption("No disagreements across selected models.")
        # Compare Models: side-by-side counts and diffs
        st.subheader("Compare models")
        cmp = df.pivot_table(index=["keyword","sentence"], columns="model", values="classification", aggfunc=lambda x: ", ".join(sorted(set(x))))
        st.dataframe(cmp, use_container_width=True)

    # Feedback dashboard
    st.subheader("Feedback dashboard")
    try:
        agg, recent = get_feedback_summary()
        if agg:
            agg_df = pd.DataFrame(agg, columns=["keyword","classification","correct","incorrect"]).sort_values(["keyword","classification"])  # type: ignore
            st.dataframe(agg_df, use_container_width=True)
        if recent:
            recent_df = pd.DataFrame([
                {"keyword": r.keyword, "classification": r.classification, "correct": r.correct_label, "time": r.created_at}
                for r in recent
            ])
            st.dataframe(recent_df, use_container_width=True)
    except Exception:
        pass

    # Batch ZIP export of SVGs
    import zipfile, io as _io
    if st.button("Download all dependency graphs (ZIP)"):
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for j, r in df.iterrows():
                mdl = r.get("model", model_choices[0])
                nlp = get_pipeline(mdl, use_gpu)
                if "parser" not in nlp.pipe_names:
                    continue
                svg = render_dependency_svg(r["sentence"], nlp)
                name = f"{j:03d}_{r['keyword']}_{mdl}.svg".replace(" ", "_")
                zf.writestr(name, svg)
        st.download_button("Save ZIP", buf.getvalue(), file_name="dependency_graphs.zip", mime="application/zip")

    # Active learning: review queue (low confidence or disagreement)
    st.subheader("Review queue")
    low_conf = df[df["confidence"] < 0.7]
    # Build disagreement key set
    dis_keys = set()
    if df["model"].nunique() > 1:
        _tmp = df.groupby(["sentence","keyword"])['classification'].nunique().reset_index()
        dis_keys = set((r.sentence, r.keyword) for r in _tmp[_tmp['classification'] > 1].itertuples(index=False))
    queue = []
    for r in df.itertuples(index=False):
        if (r.sentence, r.keyword) in dis_keys or getattr(r, 'confidence', 1.0) < 0.7:
            queue.append(r)
    if queue:
        for idx, r in enumerate(queue[:5]):
            with st.expander(f"Review: {r.keyword} – {r.classification} [{getattr(r,'model','n/a')}]", expanded=False):
                st.markdown(r.sentence)
                c1, c2 = st.columns(2)
                if c1.button("Mark correct", key=f"rev_ok_{idx}"):
                    insert_feedback(keyword=r.keyword, sentence=r.sentence, classification=r.classification, correct_label=True)
                    st.success("Recorded")
                if c2.button("Mark incorrect", key=f"rev_bad_{idx}"):
                    insert_feedback(keyword=r.keyword, sentence=r.sentence, classification=r.classification, correct_label=False)
                    st.success("Recorded")
    else:
        st.caption("No items in the review queue.")

    # Session analytics across all processed documents (updated keys)
    st.subheader("Session Analytics")
    all_results = pd.DataFrame(st.session_state._hit_history)
    if not all_results.empty and "classification" in all_results:
        agg_counts = all_results["classification"].value_counts().reset_index()
        agg_counts.columns = ["Classification", "Count"]
        bar_fig = px.bar(agg_counts, x="Classification", y="Count", title="Aggregate Classification Counts")
        st.plotly_chart(bar_fig, use_container_width=True)
        st.write(f"Total documents processed: {st.session_state._doc_count}")

# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" and not st.runtime.exists():
    import argparse, json as _json
    parser = argparse.ArgumentParser(description="Headless clinical keyword polarity extraction")
    parser.add_argument("model", help="spaCy model name (e.g. en_core_web_sm)")
    parser.add_argument("keywords", help="Comma-separated keywords to search for")
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Emit JSON rather than plain text",
    )
    parser.add_argument("--gpu", action="store_true", dest="gpu", help="Use GPU if available")
    args = parser.parse_args()

    mdl = args.model
    kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
    txt = sys.stdin.read()
    nlp_cli = nlp_load_pipeline(mdl, gpu=args.gpu)
    results = nlp_extract(txt, kws, nlp_cli)
    if args.as_json:
        print(_json.dumps(results, ensure_ascii=False))
    else:
        for r in results:
            print(r)
