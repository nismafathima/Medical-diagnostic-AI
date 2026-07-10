# utils/report_rag.py
# RAG pipeline for generating diagnostic reports
# Uses FAISS + sentence-transformers + Flan-T5 LLM

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import pipeline
from fpdf import FPDF
from datetime import datetime
import tempfile, os

try:
    import streamlit as st
    _cache_resource = st.cache_resource(show_spinner=False)
except Exception:  # pragma: no cover - allows module to be imported outside Streamlit
    def _cache_resource(func):
        return func


# ── Medical knowledge base (embedded at startup) ───────────────────────────────
MEDICAL_KNOWLEDGE = {
    "Pneumonia": """
    Pneumonia is an infection that inflames air sacs in one or both lungs.
    Symptoms include cough with phlegm, fever, chills, and difficulty breathing.
    Bacterial pneumonia is treated with antibiotics. Viral pneumonia may need antivirals.
    Common organisms: Streptococcus pneumoniae, Haemophilus influenzae.
    Diagnosis: Chest X-ray shows consolidation. Blood tests show elevated WBC.
    Risk factors: Elderly, immunocompromised, smokers, chronic lung disease.
    Complications: Sepsis, respiratory failure, pleural effusion.
    Treatment: Amoxicillin for mild cases. Hospitalization if severe (CURB-65 score).
    """,
    "COVID-19": """
    COVID-19 is caused by SARS-CoV-2 coronavirus. Spreads via respiratory droplets.
    Symptoms: Fever, dry cough, loss of taste/smell, breathlessness, fatigue.
    Chest X-ray/CT shows bilateral ground-glass opacities.
    SpO2 below 94% indicates need for oxygen support.
    Treatment: Supportive care. Antivirals (Paxlovid, Remdesivir) for high-risk.
    Severe cases: Dexamethasone reduces mortality. ICU for mechanical ventilation.
    Prevention: Vaccination, masking, social distancing.
    Complications: Long COVID, cytokine storm, organ damage.
    """,
    "Tuberculosis": """
    Tuberculosis (TB) is caused by Mycobacterium tuberculosis. Spreads via air.
    Symptoms: Persistent cough 3+ weeks, night sweats, weight loss, blood in sputum.
    Chest X-ray: Upper lobe infiltrates, cavitations, calcified nodules.
    Diagnosis: Sputum AFB smear, GeneXpert, IGRA blood test, Mantoux test.
    Treatment: 6-month DOTS regimen — Isoniazid, Rifampicin, Pyrazinamide, Ethambutol.
    Drug-resistant TB (MDR-TB) requires longer, more complex treatment.
    Isolation required until 3 negative sputum smears.
    Complications: Miliary TB, TB meningitis, respiratory failure.
    """,
    "Influenza": """
    Influenza is caused by Influenza A or B virus. Seasonal peaks in winter.
    Symptoms: Sudden high fever, body aches, headache, fatigue, dry cough.
    Diagnosis: Rapid influenza test, PCR nasal swab.
    Treatment: Oseltamivir (Tamiflu) within 48 hours of symptom onset.
    Prevention: Annual flu vaccine. Most effective in elderly and children.
    Complications: Pneumonia, encephalitis, myocarditis.
    High-risk: Elderly, pregnant women, immunocompromised.
    """,
    "Normal": """
    No significant pathology detected on imaging.
    Patient vitals are within normal ranges.
    Routine preventive health measures are recommended.
    Annual health check-up advised. Maintain healthy diet and exercise.
    Vaccination schedule should be kept up to date.
    """,
}

_embeddings = None
_vectorstore = None
_llm_pipe = None


@_cache_resource
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


@_cache_resource
def build_medical_kb() -> FAISS:
    """Build FAISS vector store from medical knowledge base."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = []
    for disease, text in MEDICAL_KNOWLEDGE.items():
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"disease": disease}
            ))

    _vectorstore = FAISS.from_documents(docs, get_embeddings())
    return _vectorstore


def retrieve_medical_context(query: str, k: int = 4) -> str:
    """Retrieve relevant medical knowledge for a query."""
    vs = build_medical_kb()
    results = vs.similarity_search(query, k=k)
    return "\n\n".join([r.page_content for r in results])


@_cache_resource
def get_llm():
    """Load Flan-T5 model for report generation.
    Uses small variant for CPU optimization:
    - Small: 77M params, 300MB (3x faster)
    - Base: 250M params, 892MB
    - Still produces high-quality medical summaries
    """
    global _llm_pipe
    if _llm_pipe is None:
        try:
            _llm_pipe = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_new_tokens=350,
                device=-1,
                batch_size=1
            )
        except Exception as e:
            print(f"⚠️ Could not load Flan-T5-small: {e}. Trying base...")
            try:
                _llm_pipe = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    max_new_tokens=350,
                    device=-1,
                    batch_size=1
                )
            except Exception as e2:
                print(f"⚠️ Could not load Flan-T5: {e2}")
                return None
    return _llm_pipe


def generate_diagnostic_report(patient_data: dict) -> str:
    """
    Generate a diagnostic report using RAG.
    patient_data keys: name, age, gender, xray_result, symptom_result, vitals_risk
    With error handling and fallback to template-based reports.
    """
    try:
        llm = get_llm()
        
        if llm is None:
            # Fallback to template-based report
            return generate_template_report(patient_data)

        # Retrieve relevant medical context
        query = f"{patient_data.get('xray_result', '')} {patient_data.get('top_symptom', '')}"
        context = retrieve_medical_context(query)

        xray_confidence = patient_data.get("xray_confidence")
        confidence_str = f"{xray_confidence:.0%}" if isinstance(xray_confidence, (int, float)) else "N/A"

        prompt = f"""You are a medical AI assistant. Write a brief clinical diagnostic summary.

Patient: {patient_data.get('name', 'Unknown')}, Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}
X-ray Finding: {patient_data.get('xray_result', 'N/A')} (Confidence: {confidence_str})
Symptom Analysis: {patient_data.get('top_symptom', 'N/A')}
Vitals Risk: {patient_data.get('vitals_risk', 'N/A')}

Medical Context:
{context[:600]}

Write a 3-4 sentence clinical summary with findings and recommendations.
Always include: this is AI-assisted and requires physician confirmation."""

        result = llm(prompt, max_new_tokens=250, do_sample=False)
        if result and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        else:
            return generate_template_report(patient_data)
    except Exception as e:
        print(f"Error generating report: {e}")
        return generate_template_report(patient_data)


def generate_template_report(patient_data: dict) -> str:
    """Fallback template-based report generation if model fails."""
    xray_result = patient_data.get('xray_result', 'N/A')
    confidence_str = f"{patient_data.get('xray_confidence', 0):.0%}" if patient_data.get('xray_confidence') else "N/A"
    symptom = patient_data.get('top_symptom', 'N/A')
    vitals_risk = patient_data.get('vitals_risk', 'N/A')
    
    report = f"""Clinical Assessment Summary:

X-ray imaging shows {xray_result} with {confidence_str} confidence. Patient reports symptom profile consistent with {symptom}. 
Current vitals assessment indicates {vitals_risk} risk level. 

Recommendation: This is an AI-assisted preliminary assessment and requires physician confirmation before any clinical decisions are made.
Please consult with a qualified healthcare professional for definitive diagnosis and treatment planning."""
    
    return report


def _sanitize_pdf_text(text) -> str:
    """
    Replace common Unicode characters (em/en dashes, smart quotes, emoji, etc.)
    with Latin-1-safe equivalents, then drop anything else Helvetica can't render.
    Prevents FPDF's 'Character ... outside the range of characters supported'
    errors when AI-generated or f-string text contains typographic punctuation.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-",   # em dash —
        "\u2013": "-",   # en dash –
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "-",   # bullet
    }
    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)
    # Drop any remaining characters outside Latin-1 (emoji, other symbols, etc.)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def export_pdf_report(patient_data: dict, report_text: str,
                       xray_result: dict, symptom_result: dict) -> str:
    """Generate a downloadable PDF diagnostic report."""

    class MedReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(10, 60, 120)
            self.set_x(self.l_margin)
            safe_w = self.w - self.r_margin - self.l_margin
            self.multi_cell(safe_w, 10, _sanitize_pdf_text("MedAI Assistant - Diagnostic Report"), align="C")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(120, 120, 120)
            self.set_x(self.l_margin)
            self.multi_cell(safe_w, 6, _sanitize_pdf_text(
                f"Generated: {datetime.now().strftime('%d %b %Y %I:%M %p')} | AI-Assisted - Not a substitute for professional medical advice"),
                align="C")
            self.ln(4)
            self.set_draw_color(10, 60, 120)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.set_x(self.l_margin)
            safe_w = self.w - self.r_margin - self.l_margin
            self.multi_cell(safe_w, 10, _sanitize_pdf_text(f"Page {self.page_no()} | MedAI - Educational Purposes Only"), align="C")

    pdf = MedReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Helper for safe multi_cell width
    safe_w = pdf.w - pdf.r_margin - pdf.l_margin

    # Patient info
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 8, "Patient Information")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    info_rows = [
        ("Name", patient_data.get("name", "N/A")),
        ("Age", str(patient_data.get("age", "N/A"))),
        ("Gender", patient_data.get("gender", "N/A")),
        ("Report Date", datetime.now().strftime("%d %b %Y")),
    ]
    for label, val in info_rows:
        pdf.set_x(pdf.l_margin)
        pdf.cell(50, 6, _sanitize_pdf_text(f"  {label}:"), border=0)
        pdf.multi_cell(safe_w - 50, 6, _sanitize_pdf_text(val), wrapmode="CHAR")
    pdf.ln(4)

    # X-ray findings
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(10, 60, 120)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 8, _sanitize_pdf_text("X-Ray Analysis (CNN - EfficientNet-B0)"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(f"  Predicted Condition: {xray_result.get('predicted', 'N/A')}"), wrapmode="CHAR")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(f"  Confidence: {xray_result.get('confidence', 0):.1%}"), wrapmode="CHAR")
    pdf.ln(3)

    # Symptom analysis
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(10, 60, 120)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 8, _sanitize_pdf_text("Symptom Analysis (NLP - Zero-Shot Classification)"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(f"  Top Prediction: {symptom_result.get('top_prediction', 'N/A')}"), wrapmode="CHAR")
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(f"  Confidence: {symptom_result.get('top_confidence', 0):.1%}"), wrapmode="CHAR")
    detected = ", ".join(symptom_result.get("detected_symptoms", [])) or "None detected"
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(f"  Detected Symptoms: {detected}"), wrapmode="CHAR")
    pdf.ln(3)

    # Vitals
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(10, 60, 120)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 8, _sanitize_pdf_text(f"Vitals Risk Assessment: {patient_data.get('vitals_risk', 'N/A')}"), wrapmode="CHAR")
    pdf.ln(3)

    # AI Report
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(10, 60, 120)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 8, _sanitize_pdf_text("AI Diagnostic Summary (RAG - Flan-T5 + FAISS)"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 6, _sanitize_pdf_text(report_text), wrapmode="CHAR")
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 60, 60)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(safe_w, 5, _sanitize_pdf_text(
        "DISCLAIMER: This report is generated by an AI system for educational and research purposes only. "
        "It is NOT a substitute for professional medical diagnosis or treatment. "
        "Always consult a qualified healthcare professional."), wrapmode="CHAR")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name
    pdf.output(path)
    return path