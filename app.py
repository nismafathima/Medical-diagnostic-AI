# ────────────────────────────────────────────────────────────────────────────
# 🏥 MedAI Assistant - Gradio Version (CPU-Optimized)
# ────────────────────────────────────────────────────────────────────────────
# Multimodal AI-Powered Medical Diagnostic System
# Features: X-Ray Analysis, Symptom Checking, Vitals Forecasting, Report Gen,
#           Voice Input (Whisper), PDF Report Download
# ────────────────────────────────────────────────────────────────────────────

import gradio as gr
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import tempfile
import os
import re
import traceback
warnings.filterwarnings("ignore")

from fpdf import FPDF

# Import optimized modules
from utils.xray_classifier import build_model, classify_xray
from utils.symptom_analyzer import analyze_symptoms
from utils.vitals_forecaster import forecast_vital
from utils.report_rag import generate_diagnostic_report, retrieve_medical_context
from utils.openai_report import generate_diagnostic_report_openai, resolve_api_key, OpenAIReportError
from utils.speech_handler import transcribe as speech_transcribe

# ────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL CACHE
# ────────────────────────────────────────────────────────────────────────────
_xray_model = None

def get_xray_model():
    global _xray_model
    if _xray_model is None:
        try:
            _xray_model = build_model()
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return None
    return _xray_model

# ────────────────────────────────────────────────────────────────────────────
# PATIENT DATA STORAGE (In-Memory)
# ────────────────────────────────────────────────────────────────────────────
patient_storage = {
    "name": "Unknown",
    "age": 0,
    "gender": "Unknown",
    "xray_result": None,
    "symptoms": "",
    "symptom_result": None,
    "vitals_data": {},
    "last_report_text": None
}

# ────────────────────────────────────────────────────────────────────────────
# PDF / EMOJI HELPERS
# ────────────────────────────────────────────────────────────────────────────
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0001F900-\U0001F9FF"
    "]+"
)

def sanitize_for_pdf(text: str) -> str:
    """Strip emoji/unicode that fpdf's core fonts can't render."""
    if not text:
        return ""
    text = EMOJI_PATTERN.sub("", text)
    text = text.replace("━", "-").replace("•", "-")
    return text.encode("latin-1", "ignore").decode("latin-1")

def generate_pdf_file(report_text):
    """Convert the diagnostic report text into a downloadable PDF file.
    Raises gr.Error with the real reason instead of failing silently."""
    if not report_text or not report_text.strip():
        raise gr.Error("No report text to export. Generate a report first.")

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        clean_text = sanitize_for_pdf(report_text)
        
        # FIX: Pre-calculate safe width once
        safe_width = pdf.w - pdf.r_margin - pdf.l_margin

        for raw_line in clean_text.split("\n"):
            line = raw_line.rstrip()
            if not line:
                pdf.ln(4)
                continue
            
            # FIX: Reset x to left margin before each line to prevent drift
            pdf.set_x(pdf.l_margin)
            
            if line.strip().startswith("**") and line.strip().endswith("**"):
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(safe_width, 7, line.replace("**", ""), wrapmode="CHAR")
                pdf.set_font("Helvetica", size=11)
            else:
                pdf.multi_cell(safe_width, 6, line.replace("**", ""), wrapmode="CHAR")

        out_dir = os.path.join(os.getcwd(), "generated_reports")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"medai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(out_path)

        if not os.path.exists(out_path):
            raise gr.Error("PDF file was not created on disk.")

        return out_path

    except gr.Error:
        raise
    except Exception as e:
        print("⚠️ PDF generation error:")
        traceback.print_exc()
        raise gr.Error(f"PDF generation failed: {str(e)}")

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: ANALYZE X-RAY
# ────────────────────────────────────────────────────────────────────────────
def analyze_xray(image, patient_name="Unknown", patient_age=0, patient_gender="Unknown"):
    if image is None:
        return (
            "❌ **ERROR:** Please upload an X-ray image first",
            "No image", "0%", pd.DataFrame()
        )
    try:
        patient_storage["name"] = patient_name if patient_name else "Unknown"
        patient_storage["age"] = int(patient_age) if patient_age else 0
        patient_storage["gender"] = patient_gender if patient_gender else "Unknown"

        model = get_xray_model()
        if model is None:
            return (
                "❌ **ERROR:** Could not load X-ray model. Check internet connection.",
                "Error", "0%", pd.DataFrame()
            )

        result = classify_xray(image, model)
        patient_storage["xray_result"] = result

        if "error" in result:
            return (
                f"❌ **ANALYSIS FAILED:** {result.get('error', 'Unknown error')}",
                "Error", "0%", pd.DataFrame()
            )

        predicted = result["predicted"]
        confidence = f"{result['confidence']:.1%}"

        probs_df = pd.DataFrame({
            "Disease": list(result['probabilities'].keys()),
            "Probability (%)": [p * 100 for p in result['probabilities'].values()]
        })

        output_text = f"""
✅ **X-RAY ANALYSIS COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **Predicted Condition:** {predicted}
📊 **Confidence Score:** {confidence}

📈 **Class Probabilities:**
{probs_df.to_string(index=False)}

⚠️ {result['disclaimer']}

🔬 **Model:** MobileNetV2 (CPU-Optimized)
⏱️ **Inference Time:** ~100-200ms
"""
        return output_text, predicted, confidence, probs_df

    except Exception as e:
        return f"❌ **ERROR:** {str(e)}", "Error", "0%", pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: TRANSCRIBE VOICE INPUT
# ────────────────────────────────────────────────────────────────────────────
def transcribe_audio_gradio(audio_path):
    if not audio_path:
        return "❌ No audio recorded. Please record or upload audio first."
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        text = speech_transcribe(audio_bytes)
        if not text or text.startswith("[Error") or text.startswith("[Transcription failed"):
            return text or "❌ Transcription returned empty result."
        return text
    except Exception as e:
        return f"❌ Transcription error: {str(e)}"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: ANALYZE SYMPTOMS
# ────────────────────────────────────────────────────────────────────────────
def analyze_symptoms_gradio(symptoms_text):
    if not symptoms_text or len(symptoms_text.strip()) == 0:
        return (
            "❌ **ERROR:** Please enter symptoms (comma-separated)",
            pd.DataFrame(), "No symptoms"
        )
    try:
        result = analyze_symptoms(symptoms_text)
        patient_storage["symptoms"] = symptoms_text
        patient_storage["symptom_result"] = result

        if "error" in result:
            return (
                f"⚠️ **WARNING:** {result.get('error')}. Using fallback method.",
                pd.DataFrame(), result.get("top_prediction", "Unknown")
            )

        output_text = f"""
✅ **SYMPTOM ANALYSIS COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **Top Prediction:** {result['top_prediction']}
📊 **Confidence:** {result['top_confidence']:.1%}

📊 **TOP 3 DISEASE CANDIDATES:**
"""
        for i, item in enumerate(result['top3'], 1):
            output_text += f"\n{i}. **{item['disease']}** - {item['confidence']:.1%}"

        if result.get('detected_symptoms'):
            output_text += f"\n\n🔍 **Detected Symptoms:**\n"
            for symptom in result['detected_symptoms']:
                output_text += f"  • {symptom}\n"

        if result.get('keyword_matches'):
            output_text += f"\n📌 **Symptom-Disease Matches:**"
            for disease, keywords in result['keyword_matches'].items():
                output_text += f"\n  • **{disease}:** {', '.join(keywords)}"

        output_text += f"""

⚠️ AI-assisted preliminary assessment only — consult a healthcare professional.

🔬 **Model:** DistilBERT (CPU-Optimized)
"""

        chart_df = pd.DataFrame({
            "Disease": [item['disease'] for item in result['top3']],
            "Confidence (%)": [item['confidence'] * 100 for item in result['top3']]
        })

        return output_text, chart_df, result['top_prediction']

    except Exception as e:
        return f"❌ **ERROR:** {str(e)}", pd.DataFrame(), "Error"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: FORECAST VITALS
# ────────────────────────────────────────────────────────────────────────────
def forecast_vitals_gradio(vital_type, vital_readings):
    if not vital_readings or len(vital_readings.strip()) == 0:
        return (
            "❌ **ERROR:** Please enter vital readings (comma-separated numbers)",
            pd.DataFrame(), "No data"
        )
    try:
        values_str = vital_readings.replace(" ", "").split(",")
        values = [float(v) for v in values_str if v]

        if len(values) < 2:
            return (
                "❌ **ERROR:** Please enter at least 2 vital readings",
                pd.DataFrame(), "Insufficient data"
            )

        dates = pd.date_range(end=datetime.now(), periods=len(values), freq="D")
        df = pd.DataFrame({"ds": dates, "y": values})

        result = forecast_vital(df, periods=3)
        patient_storage["vitals_data"][vital_type] = result

        output_text = f"""
✅ **VITALS FORECAST COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **Vital Type:** {vital_type}
📈 **Trend:** {result['trend'].capitalize()}
🎯 **Next Predicted Value:** {result['avg_forecast']:.1f}
📉 **Change:** {result['change']:.2f}

📋 **Last Actual Reading:** {result['last_actual']:.1f}

📊 **3-Day Forecast:**
"""
        for i, val in enumerate(result['next_values'], 1):
            output_text += f"\nDay {i}: {val:.1f}"

        if "note" in result:
            output_text += f"\n\n⚠️ {result['note']}"

        output_text += f"\n\n🔬 **Model:** Facebook Prophet (CPU-Optimized)\n"

        if 'forecast_df' in result:
            chart_df = result['forecast_df'].copy()
            if 'ds' in chart_df.columns:
                chart_df['ds'] = pd.to_datetime(chart_df['ds'])
                chart_df = chart_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                chart_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        else:
            chart_df = pd.DataFrame()

        return output_text, chart_df, vital_type

    except Exception as e:
        return f"❌ **ERROR:** {str(e)}", pd.DataFrame(), "Error"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: GENERATE REPORT (trimmed — one disclaimer only)
# ────────────────────────────────────────────────────────────────────────────
def generate_report_gradio(use_openai=False):
    try:
        if not patient_storage["xray_result"]:
            report_text = (
                "❌ **INCOMPLETE ANALYSIS**\n"
                "Please complete X-Ray Analysis and Symptom Analysis first."
            )
            patient_storage["last_report_text"] = report_text
            return report_text

        if not patient_storage["symptoms"]:
            report_text = (
                "❌ **INCOMPLETE ANALYSIS**\n"
                "Please complete Symptom Analysis first."
            )
            patient_storage["last_report_text"] = report_text
            return report_text

        report_data = {
            "name": patient_storage.get("name", "Patient"),
            "age": patient_storage.get("age", 0),
            "gender": patient_storage.get("gender", "Unknown"),
            "xray_result": patient_storage["xray_result"].get("predicted", "Unknown"),
            "xray_confidence": patient_storage["xray_result"].get("confidence", 0),
            "top_symptom": patient_storage["symptoms"],
            "vitals_risk": "Moderate"
        }

        engine_used = "Flan-T5-small (CPU-Optimized, local)"
        if use_openai:
            api_key = resolve_api_key()
            try:
                query = f"{report_data.get('xray_result', '')} {report_data.get('top_symptom', '')}"
                context = retrieve_medical_context(query)
                report = generate_diagnostic_report_openai(report_data, context, api_key)
                engine_used = "OpenAI GPT (gpt-4o-mini)"
            except OpenAIReportError as e:
                report = generate_diagnostic_report(report_data)
                engine_used = f"Flan-T5-small (local) — fallback: {str(e)}"
        else:
            report = generate_diagnostic_report(report_data)

        # ── Single, short disclaimer instead of three overlapping blocks ──
        output = f"""
📄 **DIAGNOSTIC REPORT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 Name: {report_data['name']}  |  Age: {report_data['age']}  |  Gender: {report_data['gender']}

🔬 **CLINICAL FINDINGS**
X-Ray: {report_data['xray_result']} ({report_data['xray_confidence']:.1%} confidence)
Symptoms: {report_data['top_symptom']}
Vitals Risk: {report_data['vitals_risk']}

📋 **AI-GENERATED SUMMARY**
{report}

⚠️ **Disclaimer:** Educational use only — not a substitute for professional medical diagnosis. Always consult a licensed healthcare provider.

🔬 Model: {engine_used} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        patient_storage["last_report_text"] = output
        return output

    except Exception as e:
        error_out = f"❌ **REPORT GENERATION FAILED**\nError: {str(e)}"
        patient_storage["last_report_text"] = error_out
        return error_out

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 6: DOWNLOAD REPORT AS PDF
# ────────────────────────────────────────────────────────────────────────────
def download_report_pdf(report_text):
    if not report_text or not report_text.strip():
        raise gr.Error("No report to export — generate a report first.")
    if "INCOMPLETE ANALYSIS" in report_text or "REPORT GENERATION FAILED" in report_text:
        raise gr.Error("Complete X-Ray and Symptom analysis, then generate a valid report before exporting.")
    return generate_pdf_file(report_text)

# ────────────────────────────────────────────────────────────────────────────
# BUILD GRADIO INTERFACE
# ────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="🏥 MedAI Assistant - Medical Diagnostic AI",
    theme=gr.themes.Soft(),
    css="""
    .header { text-align: center; }
    .footer { text-align: center; color: gray; padding: 20px; }
    """
) as demo:

    gr.Markdown("""
    # 🏥 MedAI Assistant
    ## Medical Diagnostic AI System
    **CPU-Optimized Edition v2.2** | Powered by PyTorch + Gradio
    """)

    gr.Markdown("""
    Multimodal AI system for medical analysis:
    - 🖼️ **X-Ray Analysis** - Chest X-ray classification (MobileNetV2)
    - 🗣️ **Symptom Checker** - Disease prediction from symptoms (DistilBERT) + Voice Input
    - 📈 **Vitals Forecasting** - Time-series vital sign prediction (Prophet)
    - 📄 **Report Generator** - Comprehensive diagnostic summaries (Flan-T5) + PDF Download
    """)

    gr.Markdown("---")

    with gr.Tabs():

        # ═══════════════════════════ TAB 1: X-RAY ═══════════════════════════
        with gr.TabItem("🖼️ X-Ray Analysis", id="xray"):
            gr.Markdown("### Analyze Chest X-Ray Images")
            gr.Markdown("Upload a chest X-ray for AI-powered classification")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Patient Information")
                    xray_patient_name = gr.Textbox(label="Patient Name", placeholder="Enter patient name", value="Unknown")
                    xray_patient_age = gr.Number(label="Age", minimum=0, maximum=150, value=0, precision=0)
                    xray_patient_gender = gr.Dropdown(choices=["Male", "Female", "Other"], label="Gender", value="Unknown")

                    gr.Markdown("#### Step 2: Upload X-Ray")
                    xray_image = gr.Image(label="X-Ray Image (JPG/PNG)", type="pil")

                    xray_analyze_btn = gr.Button("🧠 Analyze X-Ray", size="lg", variant="primary", scale=1)

                with gr.Column():
                    gr.Markdown("#### Results")
                    xray_result_text = gr.Textbox(label="Analysis Results", lines=12, interactive=False)

                    with gr.Row():
                        xray_prediction = gr.Textbox(label="Prediction", interactive=False, scale=1)
                        xray_confidence = gr.Textbox(label="Confidence", interactive=False, scale=1)

                    xray_chart = gr.DataFrame(label="Probability Distribution", interactive=False)

            xray_analyze_btn.click(
                fn=analyze_xray,
                inputs=[xray_image, xray_patient_name, xray_patient_age, xray_patient_gender],
                outputs=[xray_result_text, xray_prediction, xray_confidence, xray_chart]
            )

        # ═══════════════════════ TAB 2: SYMPTOM ANALYSIS ═══════════════════
        with gr.TabItem("🗣️ Symptom Checker", id="symptoms"):
            gr.Markdown("### Analyze Patient Symptoms")
            gr.Markdown("Enter symptoms to predict potential diseases, or record your voice")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Input Symptoms")

                    gr.Markdown("**Option A — Voice Input:**")
                    symptoms_audio = gr.Audio(
                        label="Record or Upload Symptoms Audio",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                    transcribe_btn = gr.Button("🎙️ Transcribe to Text", variant="secondary")

                    gr.Markdown("**Option B — Type Symptoms (comma-separated):**")
                    symptoms_input = gr.Textbox(
                        label="Symptoms",
                        placeholder="E.g., fever, cough, fatigue, shortness of breath",
                        lines=6
                    )

                    symptoms_analyze_btn = gr.Button("🧠 Analyze Symptoms", size="lg", variant="primary")

                with gr.Column():
                    gr.Markdown("#### Results")
                    symptoms_result_text = gr.Textbox(label="Analysis Results", lines=12, interactive=False)
                    symptoms_chart = gr.DataFrame(label="Top 3 Predictions", interactive=False)
                    symptoms_prediction = gr.Textbox(label="Top Prediction", interactive=False)

            transcribe_btn.click(
                fn=transcribe_audio_gradio,
                inputs=[symptoms_audio],
                outputs=[symptoms_input]
            )

            symptoms_analyze_btn.click(
                fn=analyze_symptoms_gradio,
                inputs=[symptoms_input],
                outputs=[symptoms_result_text, symptoms_chart, symptoms_prediction]
            )

        # ═══════════════════════ TAB 3: VITALS FORECASTING ═════════════════
        with gr.TabItem("📈 Vitals Forecasting", id="vitals"):
            gr.Markdown("### Forecast Vital Signs")
            gr.Markdown("Predict next 3 days of patient vital signs")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Select Vital & Enter Data")
                    vitals_type = gr.Dropdown(
                        choices=["Heart Rate (bpm)", "Blood Pressure (mmHg)", "SpO2 (%)", "Temperature (°C)"],
                        label="Vital Sign Type", value="Heart Rate (bpm)"
                    )

                    gr.Markdown("Enter daily readings (comma-separated)")
                    vitals_input = gr.Textbox(label="Daily Readings", placeholder="E.g., 72, 75, 78, 80, 82", lines=8)

                    vitals_forecast_btn = gr.Button("📊 Forecast Vitals", size="lg", variant="primary")

                with gr.Column():
                    gr.Markdown("#### Forecast Results")
                    vitals_result_text = gr.Textbox(label="Forecast Results", lines=12, interactive=False)
                    vitals_chart = gr.DataFrame(label="Forecast Data", interactive=False)
                    vitals_summary = gr.Textbox(label="Vital Type", interactive=False)

            vitals_forecast_btn.click(
                fn=forecast_vitals_gradio,
                inputs=[vitals_type, vitals_input],
                outputs=[vitals_result_text, vitals_chart, vitals_summary]
            )

        # ═══════════════════════ TAB 4: GENERATE REPORT ════════════════════
        with gr.TabItem("📄 Generate Report", id="report"):
            gr.Markdown("### Generate Diagnostic Report")
            gr.Markdown("Create comprehensive AI-powered diagnostic report")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Requirements")
                    gr.Markdown("""
                    ✅ X-Ray Analysis (REQUIRED)
                    ✅ Symptom Analysis (REQUIRED)
                    ⚠️ Vitals Forecasting (Optional)
                    """)

                    use_openai_checkbox = gr.Checkbox(
                        label="🔑 Use OpenAI (GPT) for reports",
                        value=False,
                        info="Requires OPENAI_API_KEY set in your .env file."
                    )

                    report_generate_btn = gr.Button("🧠 Generate Diagnostic Report", size="lg", variant="primary")

                    gr.Markdown("#### Download")
                    download_pdf_btn = gr.Button("⬇️ Download Report as PDF", variant="secondary")
                    report_pdf_file = gr.File(label="Diagnostic Report (PDF)")

                with gr.Column():
                    gr.Markdown("#### Generated Report")
                    report_output = gr.Textbox(
                        label="Diagnostic Report", lines=20, interactive=False, show_copy_button=True
                    )

            report_generate_btn.click(
                fn=generate_report_gradio,
                inputs=[use_openai_checkbox],
                outputs=[report_output]
            )

            download_pdf_btn.click(
                fn=download_report_pdf,
                inputs=[report_output],
                outputs=[report_pdf_file]
            )

    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <h3>🏥 MedAI Assistant v2.2 - CPU Optimized Edition (Gradio)</h3>
        <p><b>⚠️ DISCLAIMER:</b> For educational purposes only. Not a substitute for professional medical diagnosis.</p>
        <p><b>Models:</b> MobileNetV2 | DistilBERT | Prophet | Flan-T5-small | Whisper-tiny</p>
        <p><small>Built with PyTorch, Transformers, Gradio | © 2024 | MIT License</small></p>
    </div>
    """)

if __name__ == "__main__":
    print("🚀 Starting MedAI Assistant (Gradio)...")
    print("📍 Open your browser to: http://localhost:7860")
    print("⏳ First run will download models (~1.5GB)...\n")

    demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False,
    show_error=True,
    show_api=False,
    quiet=False
)