"""Clinical Keyword Polarity Analyzer"""

from __future__ import annotations
import sys
import re
import json
import zipfile
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import plotly.express as px
from spacy.language import Language

from src.nlp_utils import (
    load_pipeline,
    extract,
    render_dependency_svg,
    set_custom_negation_triggers,
    detect_sections,
)
from src.database import init_db, insert_feedback, get_feedback_summary

st.set_page_config(page_title="Clinical Keyword Polarity", layout="wide")

MODELS = [
    ("en_core_web_sm", "spaCy small"),
    ("en_core_web_md", "spaCy medium"),
    ("en_core_sci_md", "SciSpaCy clinical"),
]

CODE_MAP = {
    "procalcitonin": ("SNOMED", "704427003"),
    "pneumonia": ("SNOMED", "233604007"),
    "sepsis": ("SNOMED", "91302008"),
    "fever": ("SNOMED", "386661006"),
    "troponin": ("SNOMED", "105000003"),
    "chest pain": ("SNOMED", "29857009"),
    "aspirin": ("RxNorm", "1191"),
    "lactate": ("SNOMED", "83036002"),
    "hypotension": ("SNOMED", "45007003"),
    "antibiotics": ("SNOMED", "255631004"),
}

PRESETS = {
    "Pneumonia management": {
        "text": """CLINICAL PRESENTATION AND HOSPITAL COURSE

A 58-year-old male with a past medical history significant for type 2 diabetes mellitus (HbA1c 7.8%), hypertension, and former tobacco use (30 pack-years, quit 5 years ago) presented to the emergency department with a 3-day history of productive cough with purulent sputum, subjective fevers with measured temperature of 38.9°C at home, and progressive dyspnea on exertion limiting his ability to ambulate more than 50 feet.

PHYSICAL EXAMINATION: Vital signs on presentation were notable for temperature 39.1°C (POSITIVE: fever confirmed), heart rate 108 bpm, respiratory rate 24/min, blood pressure 128/82 mmHg, and oxygen saturation 91% on room air, improving to 96% on 3L nasal cannula. Pulmonary examination revealed decreased breath sounds and crackles in the right lower lung field with dullness to percussion (POSITIVE: crackles present). No wheezing was appreciated (NEGATIVE: wheezing absent). Cardiovascular examination was unremarkable except for tachycardia.

LABORATORY DATA: White blood cell count was 18.4 × 10⁹/L with 89% neutrophils and 8% bands indicating leukocytosis. Procalcitonin was significantly elevated at 2.4 ng/mL (normal <0.1 ng/mL), strongly suggesting bacterial etiology (POSITIVE: procalcitonin elevated). C-reactive protein was 187 mg/L. Basic metabolic panel revealed acute kidney injury with creatinine 1.8 mg/dL (baseline 1.1 mg/dL). Arterial blood gas on 3L nasal cannula showed pH 7.44, pCO2 32 mmHg, pO2 68 mmHg. CURB-65 score was calculated at 2 (confusion absent, uremia present, respiratory rate ≥30 absent, BP low absent, age ≥65 absent).

IMAGING: Posterior-anterior and lateral chest radiograph demonstrated dense right lower lobe consolidation with air bronchograms (POSITIVE: consolidation confirmed), consistent with lobar pneumonia. No evidence of pleural effusion was identified (NEGATIVE: pleural effusion ruled out). No cavitation or lymphadenopathy was observed (NEGATIVE: cavitation absent). No pneumothorax was observed.

MICROBIOLOGY: Blood cultures (2 sets) were obtained prior to antibiotic administration and are pending (NEUTRAL: cultures pending, not yet confirmed). Sputum Gram stain revealed >25 polymorphonuclear cells and <10 epithelial cells per low power field with gram-positive diplococci in pairs. Sputum culture is pending final identification. Legionella urinary antigen was negative (NEGATIVE: Legionella ruled out). Streptococcus pneumoniae urinary antigen returned positive (POSITIVE: S. pneumoniae confirmed).

DIFFERENTIAL DIAGNOSIS: The constellation of findings is most consistent with community-acquired pneumonia. Alternative diagnoses such as pulmonary embolism, lung malignancy, and tuberculosis were considered but deemed unlikely (NEGATIVE: alternative diagnoses excluded). The patient has no history of prior pneumonia episodes (NEUTRAL: historical reference). Family history is notable for a brother who was hospitalized for pneumonia last year (NEUTRAL: family history, not patient).

ASSESSMENT AND PLAN: Community-acquired pneumonia (CAP), CURB-65 score 2, warranting inpatient management (POSITIVE: pneumonia diagnosis confirmed). Given lack of recent hospitalization, no prior antibiotic use in the past 90 days, and no risk factors for Pseudomonas aeruginosa or methicillin-resistant Staphylococcus aureus (MRSA), empiric therapy with ceftriaxone 1g IV q24h plus azithromycin 500mg IV q24h was initiated per IDSA/ATS guidelines (POSITIVE: ceftriaxone and azithromycin administered). Patient reports no known drug allergies to antibiotics (NEGATIVE: allergies absent). Intravenous fluid resuscitation with lactated Ringer's solution at 125 mL/hr was started for volume depletion and acute kidney injury. Venous thromboembolism prophylaxis with subcutaneous heparin 5000 units q8h was ordered. Clinical response will be reassessed in 48-72 hours with consideration for step-down to oral antibiotics upon achievement of clinical stability (afebrile, hemodynamically stable, tolerating oral intake, and improving oxygenation). If fever persists beyond 72 hours, will broaden antibiotic coverage (NEUTRAL: conditional future plan).""",
        "keywords": ["pneumonia", "procalcitonin", "ceftriaxone", "fever", "consolidation", "azithromycin", "crackles", "pleural effusion"],
    },
    "Acute coronary syndrome": {
        "text": """EMERGENCY DEPARTMENT PRESENTATION

A 67-year-old female with a significant cardiovascular risk profile including hypertension (on lisinopril 20mg daily), type 2 diabetes mellitus (on metformin 1000mg twice daily, most recent HbA1c 8.2%), hyperlipidemia (on atorvastatin 40mg daily), and family history of premature coronary artery disease (father with MI at age 52—NEUTRAL: family history, not patient) presented via emergency medical services with acute onset crushing substernal chest pain (POSITIVE: chest pain present) with radiation to the left arm and jaw.

HISTORY OF PRESENT ILLNESS: The patient was in her usual state of health until approximately 2 hours prior to presentation when she developed sudden-onset severe retrosternal chest pain described as "crushing" and "pressure-like," rated 9/10 in intensity (POSITIVE: active chest pain). The pain radiated to her left arm and jaw, associated with diaphoresis (POSITIVE: diaphoresis present), nausea, and dyspnea. She denied any recent cocaine or stimulant use (NEGATIVE: cocaine use denied). She took 325mg aspirin from home prior to EMS arrival (POSITIVE: aspirin administered). Sublingual nitroglycerin administered by paramedics provided minimal relief. No prior history of similar episodes (NEGATIVE: no prior chest pain). No known prior coronary artery disease or cardiac catheterization (NEGATIVE: no prior CAD or catheterization).

PHYSICAL EXAMINATION: On arrival, she appeared pale and diaphoretic (POSITIVE: diaphoresis confirmed on exam). Vital signs: BP 148/92 mmHg in the right arm, HR 96 bpm and regular, RR 20/min, SpO2 97% on room air. Cardiovascular exam revealed normal S1 and S2, no S3 or S4 gallop (NEGATIVE: gallop absent), no murmurs (NEGATIVE: murmurs absent). JVP was not elevated (NEGATIVE: JVP elevation absent). Lungs were clear to auscultation bilaterally with no crackles or wheezing (NEGATIVE: pulmonary edema absent). Lower extremities showed no peripheral edema (NEGATIVE: edema absent). Peripheral pulses were intact and symmetric.

ELECTROCARDIOGRAM: Twelve-lead ECG obtained 8 minutes after arrival demonstrated sinus rhythm at 96 bpm with ST-segment elevation of 3-4mm in leads V2-V4 (POSITIVE: ST-elevation confirmed) with reciprocal ST depression in leads II, III, and aVF. Q waves were not present (NEGATIVE: Q waves absent, suggesting no prior infarct). These findings are consistent with acute anterior ST-elevation myocardial infarction (STEMI) with likely proximal left anterior descending artery (LAD) occlusion (POSITIVE: STEMI diagnosed).

CARDIAC BIOMARKERS: Initial high-sensitivity troponin I was significantly elevated at 4,247 ng/L (normal <14 ng/L in females) (POSITIVE: troponin elevated). Point-of-care troponin at bedside was also markedly positive. NT-proBNP was 892 pg/mL (mildly elevated, suggesting no significant heart failure at this time—NEGATIVE: heart failure excluded).

MANAGEMENT: The STEMI protocol was immediately activated. Loading doses of aspirin 325mg (additional to home dose—POSITIVE: aspirin given), ticagrelor 180mg, and unfractionated heparin 60 units/kg bolus followed by 12 units/kg/hr infusion (POSITIVE: heparin initiated) were administered. Morphine 4mg IV was given for pain. Beta-blocker was held given borderline heart failure concern (NEUTRAL: beta-blocker considered but not given). The interventional cardiology team was emergently consulted, and the patient was taken directly to the cardiac catheterization laboratory within 28 minutes of arrival (door-to-balloon time goal <90 minutes) (POSITIVE: catheterization performed). No absolute contraindications to percutaneous coronary intervention were identified (NEGATIVE: contraindications absent). Fibrinolytic therapy was not pursued given PCI availability (NEGATIVE: fibrinolytics not given).

CARDIAC CATHETERIZATION FINDINGS (preliminary): Complete thrombotic occlusion of the mid-LAD (POSITIVE: LAD occlusion confirmed). Successful primary PCI with drug-eluting stent placement achieved TIMI-3 flow. Second-generation drug-eluting stent deployed. Right coronary artery and left circumflex showed no significant stenosis (NEGATIVE: other vessels patent). Patient transferred to coronary care unit in stable condition for monitoring and initiation of guideline-directed medical therapy including dual antiplatelet therapy, high-intensity statin, ACE inhibitor, and beta-blocker. Will monitor for recurrent chest pain post-intervention (NEUTRAL: future monitoring plan).""",
        "keywords": ["troponin", "st-elevation", "chest pain", "aspirin", "catheterization", "stemi", "diaphoresis", "heparin", "heart failure"],
    },
    "Sepsis evaluation": {
        "text": """INTENSIVE CARE UNIT ADMISSION NOTE

A 72-year-old female nursing home resident with past medical history significant for Alzheimer's dementia (moderate stage, baseline MMSE 18/30), recurrent urinary tract infections (NEUTRAL: historical, prior episodes), type 2 diabetes mellitus, hypertension, and chronic kidney disease stage 3b (baseline creatinine 1.8 mg/dL) was brought to the emergency department by nursing facility staff due to acute altered mental status and decreased oral intake over the past 18 hours.

HISTORY OF PRESENT ILLNESS: Per nursing home documentation, the patient was noted to be more confused than baseline beginning yesterday afternoon, refusing meals, and was found this morning to be febrile to 39.2°C (POSITIVE: fever present) with documented blood pressure of 88/52 mmHg indicating hypotension (POSITIVE: hypotension confirmed). She was unable to provide history due to confusion; collateral obtained from nursing records indicates no recent falls (NEGATIVE: falls absent), no witnessed seizure activity (NEGATIVE: seizures absent), and no new medications. Last documented bowel movement was 2 days ago. She had been treated for a urinary tract infection approximately 3 weeks ago with oral ciprofloxacin (NEUTRAL: historical antibiotic use).

PHYSICAL EXAMINATION: Vital signs on ED arrival: temperature 39.4°C (tympanic), heart rate 118 bpm indicating tachycardia (POSITIVE: tachycardia present), respiratory rate 26/min, blood pressure 82/48 mmHg (POSITIVE: hypotension persistent), SpO2 94% on room air. The patient appeared acutely ill, was somnolent but arousable to voice, oriented to name only. No meningismus or nuchal rigidity on examination (NEGATIVE: meningitis signs absent). Pupils were equal, round, and reactive. Cardiac examination revealed tachycardia, regular rhythm, no murmurs (NEGATIVE: murmurs absent). Pulmonary examination: clear to auscultation without crackles or wheezes (NEGATIVE: pneumonia signs absent). Abdomen was soft, non-distended, with mild suprapubic tenderness; bowel sounds present. Skin was warm and flushed with delayed capillary refill >3 seconds. Foley catheter in place with cloudy, malodorous urine noted in collection bag.

LABORATORY EVALUATION: Complete blood count: WBC 22.3 × 10⁹/L with 92% neutrophils, 4% bands (left shift); hemoglobin 11.2 g/dL; platelets 98 × 10⁹/L (baseline 220) suggesting possible early DIC (NEUTRAL: DIC will need to be monitored). Complete metabolic panel: sodium 132 mEq/L, potassium 5.1 mEq/L, chloride 98 mEq/L, bicarbonate 16 mEq/L, BUN 48 mg/dL, creatinine 3.2 mg/dL (acute-on-chronic kidney injury), glucose 287 mg/dL. Lactate obtained stat was critically elevated at 4.1 mmol/L (normal <2.0 mmol/L) (POSITIVE: lactate elevated), consistent with tissue hypoperfusion.

Arterial blood gas on room air: pH 7.31, pCO2 28 mmHg, pO2 72 mmHg, HCO3 14 mEq/L (partially compensated metabolic acidosis with elevated anion gap of 18). Procalcitonin 8.7 ng/mL (highly suggestive of bacterial infection). Urinalysis: large leukocyte esterase, positive nitrites, >100 WBC/hpf, moderate bacteria (POSITIVE: UTI confirmed). Blood and urine cultures obtained prior to antibiotics (NEUTRAL: cultures pending final results).

SEPSIS CRITERIA: Patient meets Sepsis-3 criteria with suspected infection (urinary source) and SOFA score increase of ≥2 points from baseline (POSITIVE: sepsis criteria met). Septic shock is present given persistent hypotension requiring vasopressor therapy despite adequate fluid resuscitation and serum lactate >2 mmol/L (POSITIVE: septic shock diagnosed). Pneumonia was ruled out based on clear lung exam and chest X-ray (NEGATIVE: pneumonia excluded). No evidence of intra-abdominal source (NEGATIVE: abdominal infection absent).

MANAGEMENT: Hour-1 sepsis bundle was initiated immediately. Broad-spectrum antibiotics (piperacillin-tazobactam 3.375g IV and vancomycin 1.5g IV loading dose) were administered within 45 minutes of arrival (POSITIVE: antibiotics given). Aggressive crystalloid resuscitation with 30 mL/kg lactated Ringer's solution (approximately 2 liters) was infused. Despite fluid resuscitation, mean arterial pressure remained <65 mmHg, and norepinephrine infusion was initiated at 5 mcg/min (POSITIVE: vasopressor/norepinephrine started), titrated to maintain MAP ≥65 mmHg. Epinephrine was not required (NEGATIVE: epinephrine not needed). Central venous catheter was placed in the right internal jugular vein for reliable access and vasopressor administration. Foley catheter output was monitored hourly with goal urine output >0.5 mL/kg/hr. Repeat lactate at 4 hours showed improvement to 2.8 mmol/L. Source control was addressed by removal and replacement of the chronic Foley catheter. Patient was admitted to the medical intensive care unit for close hemodynamic monitoring and management of septic shock secondary to complicated urinary tract infection (POSITIVE: sepsis and septic shock confirmed). Family has been notified and will discuss goals of care if clinical status deteriorates (NEUTRAL: future contingency planning).""",
        "keywords": ["sepsis", "lactate", "hypotension", "antibiotics", "cultures", "vasopressor", "norepinephrine", "tachycardia", "fever", "meningitis"],
    },
    "Stroke assessment": {
        "text": """ACUTE STROKE TEAM EVALUATION

A 78-year-old right-handed male with past medical history of atrial fibrillation (on warfarin, reportedly compliant—NEUTRAL: historical medication), hypertension, hyperlipidemia, and prior transient ischemic attack 3 years ago (NEUTRAL: historical TIA, not current stroke) was brought to the emergency department by his wife after she witnessed sudden onset of left-sided weakness and slurred speech while he was eating dinner.

HISTORY OF PRESENT ILLNESS: Per the patient's wife, they were having dinner at approximately 6:15 PM when she noticed her husband suddenly dropped his fork with his left hand and began speaking with slurred words. She immediately recognized these as stroke symptoms and called 911. Last known well time was 6:10 PM (current time 7:45 PM, approximately 95 minutes since symptom onset). On EMS arrival, the patient was noted to have left facial droop, left arm weakness with drift, and dysarthria (POSITIVE: dysarthria present). Blood glucose in the field was 142 mg/dL (ruling out hypoglycemia—NEGATIVE: hypoglycemia excluded). The patient was transported as a stroke alert.

NEUROLOGICAL EXAMINATION: The patient was awake, alert, and oriented to person and place but not time. Pupils were 3mm bilaterally and reactive. Extraocular movements showed forced gaze deviation to the right with inability to cross midline to the left. Visual fields showed left homonymous hemianopia to confrontation. Left facial droop was present (lower face). Motor examination revealed left upper extremity 0/5 strength with complete plegia, left lower extremity 2/5 strength. Right upper and lower extremities were 5/5 (NEGATIVE: right-sided weakness absent). Left-sided sensory extinction to double simultaneous stimulation. Deep tendon reflexes were 2+ throughout; left Babinski sign was present. Dysarthria was moderate with slurred but comprehensible speech (POSITIVE: dysarthria confirmed). Neglect of left hemispace was evident on visual and tactile testing. No aphasia was noted (NEGATIVE: aphasia absent).

NIHSS ASSESSMENT: 1a (LOC): 0, 1b (LOC questions): 1, 1c (LOC commands): 0, 2 (Best gaze): 2, 3 (Visual fields): 2, 4 (Facial palsy): 2, 5a (Motor arm left): 4, 5b (Motor arm right): 0, 6a (Motor leg left): 3, 6b (Motor leg right): 0, 7 (Limb ataxia): 0, 8 (Sensory): 1, 9 (Best language): 0, 10 (Dysarthria): 2, 11 (Extinction): 2. TOTAL NIHSS SCORE: 19 (severe stroke) (POSITIVE: elevated NIHSS confirmed).

LABORATORY DATA: INR was subtherapeutic at 1.4 (goal 2.0-3.0 for atrial fibrillation) (POSITIVE: subtherapeutic anticoagulation, relevant to stroke etiology). Complete blood count was within normal limits. Comprehensive metabolic panel unremarkable. Point-of-care glucose 138 mg/dL.

NEUROIMAGING: Non-contrast CT head performed within 18 minutes of arrival demonstrated no evidence of acute intracranial hemorrhage (NEGATIVE: hemorrhage ruled out), no established territorial infarction (NEGATIVE: completed infarct absent), and preserved gray-white matter differentiation. ASPECTS score was 9 (early ischemic changes in the right insular ribbon only). CT angiography of the head and neck revealed complete occlusion of the right M1 segment of the middle cerebral artery at its origin with absent distal opacification (POSITIVE: MCA occlusion confirmed). Cervical internal carotid arteries were patent without significant stenosis (NEGATIVE: carotid stenosis absent). CT perfusion demonstrated large area of penumbra in the right MCA territory with core infarct volume estimated at 18 mL and ischemic penumbra (tissue at risk) of 94 mL—favorable mismatch ratio for intervention.

ASSESSMENT: Acute ischemic stroke secondary to right middle cerebral artery (MCA) M1 occlusion (POSITIVE: stroke confirmed), likely cardioembolic etiology in setting of undertreated atrial fibrillation (POSITIVE: atrial fibrillation as risk factor). NIHSS 19 indicating severe deficit. Patient is within the thrombolysis window (last known well <4.5 hours ago) and meets criteria for mechanical thrombectomy. Hemorrhagic stroke was excluded by imaging (NEGATIVE: hemorrhagic stroke ruled out).

TREATMENT: Given presentation within 4.5 hours and no absolute contraindications to intravenous thrombolysis (NEGATIVE: tPA contraindications absent), alteplase (tPA) 0.9 mg/kg was administered (10% bolus, remainder infused over 60 minutes) (POSITIVE: tPA/alteplase given). Total dose 72 mg for 80 kg patient. Blood pressure goal maintained <185/110 mmHg per protocol with labetalol PRN. Patient was then emergently transferred to the neurointerventional suite for mechanical thrombectomy (POSITIVE: thrombectomy performed). Groin puncture was achieved at 8:22 PM. First-pass thrombectomy with Solitaire stent retriever achieved successful recanalization with TICI 2c flow. Post-procedure angiogram confirmed restored MCA flow (NEGATIVE: persistent MCA occlusion no longer present). Patient was admitted to the neurocritical care unit for 24-hour monitoring, serial neurological examinations, and blood pressure management. Post-intervention NIHSS improved to 8 at 6 hours. Dual antiplatelet therapy will be held for 24 hours post-tPA per protocol; anticoagulation for atrial fibrillation to be restarted after hemorrhagic transformation is ruled out on 24-hour follow-up CT (NEUTRAL: future plan contingent on imaging). If hemorrhagic transformation develops, anticoagulation will be held indefinitely (NEUTRAL: conditional future scenario).""",
        "keywords": ["nihss", "tpa", "thrombectomy", "mca occlusion", "hemorrhage", "atrial fibrillation", "alteplase", "dysarthria", "aphasia", "hypoglycemia"],
    },
}

init_db()

st.title("Clinical Keyword Polarity Analyzer")
st.caption("Detect positive, negative, and neutral mentions of clinical terms")

with st.sidebar:
    if st.button("Reset", use_container_width=True):
        for key in ["text", "keywords", "results", "_doc_count", "_hit_history", "_kw_colors"]:
            st.session_state.pop(key, None)
        st.rerun()

# Session state
if "_doc_count" not in st.session_state:
    st.session_state._doc_count = 0
if "_hit_history" not in st.session_state:
    st.session_state._hit_history = []
if "_kw_colors" not in st.session_state:
    st.session_state._kw_colors = {}

# URL params
_qp = st.query_params if hasattr(st, "query_params") else {}


@st.cache_resource
def get_pipeline(name: str) -> Language:
    return load_pipeline(name, gpu=False)


# Preset loader
st.subheader("Quick Start")
col1, col2 = st.columns([3, 1])
with col1:
    preset = st.selectbox("Load a preset case", ["(none)"] + list(PRESETS.keys()))
with col2:
    if st.button("Load", use_container_width=True) and preset != "(none)":
        st.session_state.text = PRESETS[preset]["text"]
        st.session_state.keywords = ", ".join(PRESETS[preset]["keywords"])
        st.rerun()

# Text input
st.subheader("Clinical Text")
text = st.text_area(
    "Paste clinical note",
    value=st.session_state.get("text", ""),
    height=180,
    placeholder="Paste clinical note here...",
    label_visibility="collapsed",
)

# Keywords
st.subheader("Keywords")
url_kw = _qp.get("k", "")
default_kw = st.session_state.get("keywords", url_kw)
keywords_input = st.text_input(
    "Keywords to analyze (comma-separated)",
    value=default_kw,
    placeholder="fever, cough, pneumonia",
    label_visibility="collapsed",
)
keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]

# Settings
st.subheader("Settings")
col1, col2 = st.columns(2)

with col1:
    available_models = [m for m, _ in MODELS]
    url_model = _qp.get("m", "").split(",") if _qp.get("m") else []
    default_models = [m for m in url_model if m in available_models] or ["en_core_web_sm"]
    model_choices = st.multiselect(
        "Models",
        available_models,
        default=default_models,
        format_func=lambda x: dict(MODELS).get(x, x),
        max_selections=3,
    )

with col2:
    c1, c2 = st.columns(2)
    with c1:
        expand_synonyms = st.checkbox("Expand synonyms", help="WordNet expansion")
        show_codes = st.checkbox("Clinical codes", value=True)
    with c2:
        show_sections = st.checkbox("Detect sections", value=True)
        show_temporal = st.checkbox("Temporal extraction", value=True)

# Advanced options
with st.expander("Advanced options"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Keyword colors**")
        palette = ["#EF476F", "#06D6A0", "#118AB2", "#FFD166", "#8338EC"]
        for i, kw in enumerate(keywords[:5]):
            st.session_state._kw_colors.setdefault(kw, palette[i % len(palette)])
            st.session_state._kw_colors[kw] = st.color_picker(
                kw, st.session_state._kw_colors.get(kw, palette[0]), key=f"color_{kw}"
            )
    
    with col2:
        st.markdown("**Custom negation triggers**")
        custom_neg = st.text_area(
            "Additional triggers",
            placeholder="ruled out, unlikely, no evidence of",
            height=100,
            label_visibility="collapsed",
        )
        triggers = [t.strip() for t in custom_neg.split(",") if t.strip()]
        set_custom_negation_triggers(triggers)

st.divider()

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    if not text.strip():
        st.error("Enter text")
        st.stop()
    if not keywords:
        st.error("Enter keywords")
        st.stop()
    if not model_choices:
        st.error("Select at least one model")
        st.stop()
    
    # Synonym expansion
    terms = list(set(keywords))
    if expand_synonyms:
        from nltk.corpus import wordnet as wn
        expanded = set(terms)
        for term in keywords:
            for syn in wn.synsets(term):
                for lemma in syn.lemma_names():
                    expanded.add(lemma.replace("_", " ").lower())
        terms = list(expanded)
    
    results = []
    for model in model_choices:
        with st.spinner(f"Analyzing with {model}..."):
            nlp = get_pipeline(model)
            results.extend(extract(text, terms, nlp, model_name=model))
    
    if not results:
        st.warning("No occurrences found")
        st.stop()
    
    df = pd.DataFrame(results)
    st.session_state._doc_count += 1
    st.session_state._hit_history.extend(results)
    st.session_state.results = df
    
    # Section detection
    if show_sections:
        headers = detect_sections(text) + [{"start": len(text) + 1, "title": "END"}]
        def get_section(sentence: str) -> str:
            pos = text.find(sentence[:40])
            if pos < 0:
                return "Unknown"
            current = "Unknown"
            for h in headers:
                if pos < h["start"]:
                    return current
                current = h["title"]
            return current
        df["section"] = df["sentence"].map(get_section)
    
    # Clinical codes
    if show_codes:
        df["code_system"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[0])
        df["code"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[1])
    
    # Temporal extraction
    if show_temporal:
        temporal_pat = re.compile(
            r"\b(?:today|yesterday|tomorrow|last\s+\w+|next\s+\w+|"
            r"\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|"
            r"\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|prior))\b",
            re.I
        )
        df["temporal"] = df["sentence"].map(
            lambda s: ", ".join(sorted(set(m.group(0) for m in temporal_pat.finditer(s))))
        )
    
    # Results
    st.divider()
    st.subheader("Results")
    
    # Metrics
    cols = st.columns(4)
    cols[0].metric("Total", len(df))
    for i, cls in enumerate(["Positive", "Negative", "Neutral"]):
        cols[i + 1].metric(cls, len(df[df["classification"] == cls]))
    
    # Chart
    pivot = df.pivot_table(
        index="model", columns="classification", values="keyword", aggfunc="count", fill_value=0
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    
    fig = px.bar(
        pivot.reset_index().melt("model", var_name="Classification", value_name="Count"),
        x="model", y="Count", color="Classification",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"},
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Occurrences
    COLORS = {"Positive": "#dcfce7", "Negative": "#fee2e2", "Neutral": "#fef3c7"}
    
    def highlight(row):
        kw = row["keyword"]
        color = st.session_state._kw_colors.get(kw, COLORS[row["classification"]])
        return re.sub(
            rf"(?i)\b({re.escape(kw)})\b",
            rf"<mark style='background:{color};padding:2px;border-radius:3px;'>\1</mark>",
            row["sentence"]
        )
    
    st.subheader("Occurrences")
    for idx, row in df.iterrows():
        with st.expander(f"{row['keyword']} — {row['classification']} ({row['model']})"):
            st.markdown(highlight(row), unsafe_allow_html=True)
            st.caption(f"Confidence: {row['confidence']:.2f}")
            if show_sections:
                st.caption(f"Section: {row.get('section', 'N/A')}")
            if show_codes and row.get("code"):
                st.caption(f"Code: {row['code_system']} {row['code']}")
            if show_temporal and row.get("temporal"):
                st.caption(f"Temporal: {row['temporal']}")
            
            c1, c2 = st.columns(2)
            if c1.button("✓ Correct", key=f"ok_{idx}"):
                insert_feedback(row["keyword"], row["sentence"], row["classification"], True)
                st.toast("Recorded")
            if c2.button("✗ Wrong", key=f"bad_{idx}"):
                insert_feedback(row["keyword"], row["sentence"], row["classification"], False)
                st.toast("Recorded")
    
    # Review queue
    st.subheader("Review Queue")
    low_conf = df[df["confidence"] < 0.7]
    if len(low_conf):
        st.warning(f"{len(low_conf)} items need review (confidence < 0.7)")
        st.dataframe(low_conf[["keyword", "classification", "confidence", "sentence"]])
    else:
        st.success("No low-confidence items")
    
    st.subheader("Feedback Dashboard")
    agg, recent = get_feedback_summary()
    if agg:
        agg_df = pd.DataFrame(agg, columns=["keyword", "classification", "correct", "incorrect"])
        st.dataframe(agg_df)
    else:
        st.caption("No feedback recorded yet")
    
    # Session analytics
    st.subheader("Session Analytics")
    all_hits = pd.DataFrame(st.session_state._hit_history)
    if not all_hits.empty:
        counts = all_hits["classification"].value_counts()
        fig2 = px.pie(values=counts.values, names=counts.index, title="All-session distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Documents analyzed: {st.session_state._doc_count}")
    
    # Exports
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button("CSV", df.to_csv(index=False), "results.csv", mime="text/csv")
    
    with c2:
        def to_fhir(rec):
            return {
                "resourceType": "Observation",
                "code": {"text": rec["keyword"]},
                "valueString": rec["sentence"],
                "interpretation": [{"text": rec["classification"]}],
            }
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": to_fhir(r)} for r in df.to_dict("records")]}
        st.download_button("FHIR Bundle", json.dumps(bundle, indent=2), "bundle.json", mime="application/json")
    
    with c3:
        if st.button("Generate dependency graphs ZIP"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for i, row in df.iterrows():
                    nlp = get_pipeline(row.get("model", model_choices[0]))
                    if "parser" in nlp.pipe_names:
                        svg = render_dependency_svg(row["sentence"], nlp)
                        zf.writestr(f"{i}_{row['keyword']}.svg", svg)
            st.download_button("Download ZIP", buf.getvalue(), "dep_graphs.zip", mime="application/zip")
