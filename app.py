# ────────────────────────────────────────────────────────────────────────────
# 🏥 MedAI Assistant - Gradio Version (CPU-Optimized)
# ────────────────────────────────────────────────────────────────────────────
# Multimodal AI-Powered Medical Diagnostic System
# Features: X-Ray Analysis, Symptom Checking, Vitals Forecasting, Report Gen
# ────────────────────────────────────────────────────────────────────────────

import gradio as gr
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import optimized modules
from utils.xray_classifier import build_model, classify_xray
from utils.symptom_analyzer import analyze_symptoms
from utils.vitals_forecaster import forecast_vital
from utils.report_rag import generate_diagnostic_report, retrieve_medical_context
from utils.openai_report import generate_diagnostic_report_openai, resolve_api_key, OpenAIReportError

# ────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL CACHE
# ────────────────────────────────────────────────────────────────────────────
_xray_model = None

def get_xray_model():
    """Load and cache X-ray model globally"""
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
# FUNCTION 1: ANALYZE X-RAY
# ────────────────────────────────────────────────────────────────────────────
def analyze_xray(image, patient_name="Unknown", patient_age=0, patient_gender="Unknown"):
    """
    Analyze chest X-ray using MobileNetV2 (optimized for CPU)
    Returns: prediction, confidence, probability chart
    """
    if image is None:
        return (
            "❌ **ERROR:** Please upload an X-ray image first",
            "No image",
            "0%",
            pd.DataFrame()
        )
    
    try:
        # Store patient information
        patient_storage["name"] = patient_name if patient_name else "Unknown"
        patient_storage["age"] = int(patient_age) if patient_age else 0
        patient_storage["gender"] = patient_gender if patient_gender else "Unknown"
        
        # Load model
        model = get_xray_model()
        if model is None:
            return (
                "❌ **ERROR:** Could not load X-ray model. Check internet connection.",
                "Error",
                "0%",
                pd.DataFrame()
            )
        
        # Analyze image
        result = classify_xray(image, model)
        patient_storage["xray_result"] = result
        
        # Check for errors in result
        if "error" in result:
            return (
                f"❌ **ANALYSIS FAILED:** {result.get('error', 'Unknown error')}",
                "Error",
                "0%",
                pd.DataFrame()
            )
        
        # Format output
        predicted = result["predicted"]
        confidence = f"{result['confidence']:.1%}"
        
        # Create probability dataframe for visualization
        probs_df = pd.DataFrame({
            "Disease": list(result['probabilities'].keys()),
            "Probability (%)": [p * 100 for p in result['probabilities'].values()]
        })
        
        # Create detailed output
        output_text = f"""
✅ **X-RAY ANALYSIS COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **Predicted Condition:** {predicted}
📊 **Confidence Score:** {confidence}

📈 **Class Probabilities:**
{probs_df.to_string(index=False)}

⚠️ **DISCLAIMER:**
{result['disclaimer']}

🔬 **Model:** MobileNetV2 (CPU-Optimized)
⏱️ **Inference Time:** ~100-200ms
"""
        
        return output_text, predicted, confidence, probs_df
    
    except Exception as e:
        error_msg = f"❌ **ERROR:** {str(e)}"
        return error_msg, "Error", "0%", pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: ANALYZE SYMPTOMS
# ────────────────────────────────────────────────────────────────────────────
def analyze_symptoms_gradio(symptoms_text):
    """
    Analyze symptoms using DistilBERT (optimized for CPU)
    Returns: disease prediction with confidence
    """
    if not symptoms_text or len(symptoms_text.strip()) == 0:
        return (
            "❌ **ERROR:** Please enter symptoms (comma-separated)",
            pd.DataFrame(),
            "No symptoms"
        )
    
    try:
        # Analyze symptoms
        result = analyze_symptoms(symptoms_text)
        patient_storage["symptoms"] = symptoms_text
        patient_storage["symptom_result"] = result
        
        # Check for errors
        if "error" in result:
            return (
                f"⚠️ **WARNING:** {result.get('error')}. Using fallback method.",
                pd.DataFrame(),
                result.get("top_prediction", "Unknown")
            )
        
        # Create output text
        output_text = f"""
✅ **SYMPTOM ANALYSIS COMPLETE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **Top Prediction:** {result['top_prediction']}
📊 **Confidence:** {result['top_confidence']:.1%}

📊 **TOP 3 DISEASE CANDIDATES:**
"""
        
        # Add top 3 predictions
        for i, item in enumerate(result['top3'], 1):
            output_text += f"\n{i}. **{item['disease']}** - {item['confidence']:.1%}"
        
        # Add detected symptoms
        if result.get('detected_symptoms'):
            output_text += f"\n\n🔍 **Detected Symptoms:**\n"
            for symptom in result['detected_symptoms']:
                output_text += f"  • {symptom}\n"
        
        # Add keyword matches
        if result.get('keyword_matches'):
            output_text += f"\n📌 **Symptom-Disease Matches:**"
            for disease, keywords in result['keyword_matches'].items():
                output_text += f"\n  • **{disease}:** {', '.join(keywords)}"
        
        output_text += f"""

⚠️ **DISCLAIMER:**
This is an AI-assisted preliminary assessment. 
Always consult with a qualified healthcare professional.

🔬 **Model:** DistilBERT (CPU-Optimized)
⏱️ **Inference Time:** ~500ms-1s
"""
        
        # Create chart of top 3
        chart_df = pd.DataFrame({
            "Disease": [item['disease'] for item in result['top3']],
            "Confidence (%)": [item['confidence'] * 100 for item in result['top3']]
        })
        
        return output_text, chart_df, result['top_prediction']
    
    except Exception as e:
        error_msg = f"❌ **ERROR:** {str(e)}"
        return error_msg, pd.DataFrame(), "Error"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: FORECAST VITALS
# ────────────────────────────────────────────────────────────────────────────
def forecast_vitals_gradio(vital_type, vital_readings):
    """
    Forecast vital signs using Prophet (CPU-optimized)
    Returns: forecast results and chart
    """
    if not vital_readings or len(vital_readings.strip()) == 0:
        return (
            "❌ **ERROR:** Please enter vital readings (comma-separated numbers)",
            pd.DataFrame(),
            "No data"
        )
    
    try:
        # Parse input
        values_str = vital_readings.replace(" ", "").split(",")
        values = [float(v) for v in values_str if v]
        
        if len(values) < 2:
            return (
                "❌ **ERROR:** Please enter at least 2 vital readings",
                pd.DataFrame(),
                "Insufficient data"
            )
        
        # Create time-series dataframe
        dates = pd.date_range(end=datetime.now(), periods=len(values), freq="D")
        df = pd.DataFrame({
            "ds": dates,
            "y": values
        })
        
        # Forecast
        result = forecast_vital(df, periods=3)
        patient_storage["vitals_data"][vital_type] = result
        
        # Format output
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
        
        # Add note if using fallback
        if "note" in result:
            output_text += f"\n\n⚠️ **Note:** {result['note']}"
        
        output_text += f"""

🔬 **Model:** Facebook Prophet (CPU-Optimized)
⏱️ **Forecast Time:** ~1-3s
"""
        
        # Create chart dataframe
        if 'forecast_df' in result:
            chart_df = result['forecast_df'].copy()
            # Ensure datetime column
            if 'ds' in chart_df.columns:
                chart_df['ds'] = pd.to_datetime(chart_df['ds'])
                # Rename for display
                chart_df = chart_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                chart_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
        else:
            chart_df = pd.DataFrame()
        
        return output_text, chart_df, vital_type
    
    except Exception as e:
        error_msg = f"❌ **ERROR:** {str(e)}"
        return error_msg, pd.DataFrame(), "Error"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: GENERATE REPORT
# ────────────────────────────────────────────────────────────────────────────
def generate_report_gradio(use_openai=False):
    """
    Generate diagnostic report from all analyses
    Uses RAG with Flan-T5-small (CPU-optimized) by default,
    or OpenAI GPT if use_openai is True and a key is configured.
    """
    try:
        # Check if X-ray analysis is done
        if not patient_storage["xray_result"]:
            return """
❌ **INCOMPLETE ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please complete the following analyses first:

1. ❌ X-Ray Analysis (REQUIRED)
2. ⚠️ Symptom Analysis (REQUIRED)
3. ⚠️ Vitals Forecasting (Optional)

Navigate to the previous tabs to complete these analyses.
"""
        
        # Check if symptoms are analyzed
        if not patient_storage["symptoms"]:
            return """
❌ **INCOMPLETE ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Please complete Symptom Analysis first.

Analyses completed:
1. ✅ X-Ray Analysis
2. ❌ Symptom Analysis (REQUIRED)
3. ⚠️ Vitals Forecasting (Optional)

Navigate to Symptom Checker tab to complete the analysis.
"""
        
        # Prepare data for report
        report_data = {
            "name": patient_storage.get("name", "Patient"),
            "age": patient_storage.get("age", 0),
            "gender": patient_storage.get("gender", "Unknown"),
            "xray_result": patient_storage["xray_result"].get("predicted", "Unknown"),
            "xray_confidence": patient_storage["xray_result"].get("confidence", 0),
            "top_symptom": patient_storage["symptoms"],
            "vitals_risk": "Moderate"  # Can be enhanced with vitals analysis
        }
        
        # Generate report — OpenAI GPT if requested and available, else local Flan-T5
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
                engine_used = f"Flan-T5-small (local) — OpenAI fallback reason: {str(e)}"
        else:
            report = generate_diagnostic_report(report_data)
        
        # Format final output
        output = f"""
📄 **DIAGNOSTIC REPORT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👤 **PATIENT INFORMATION**
  Name: {report_data['name']}
  Age: {report_data['age']} years
  Gender: {report_data['gender']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 **CLINICAL FINDINGS**

X-Ray Analysis:
  • Condition: {report_data['xray_result']}
  • Confidence: {report_data['xray_confidence']:.1%}

Symptom Analysis:
  • Reported Symptoms: {report_data['top_symptom']}

Vitals Assessment:
  • Risk Level: {report_data['vitals_risk']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **AI-GENERATED SUMMARY**

{report}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **CRITICAL DISCLAIMER**

This report is generated by AI for EDUCATIONAL PURPOSES ONLY.

🔴 DO NOT USE FOR MEDICAL DECISIONS
🔴 NOT A SUBSTITUTE FOR PROFESSIONAL DIAGNOSIS
🔴 ALWAYS CONSULT A LICENSED HEALTHCARE PROFESSIONAL

This system uses general-purpose AI models trained on public 
datasets. It does NOT have access to actual medical training data 
and should NEVER be used for clinical decision-making.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 **TECHNICAL DETAILS**
  Model: {engine_used}
  Type: Retrieval-Augmented Generation (RAG)
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return output
    
    except Exception as e:
        return f"""
❌ **REPORT GENERATION FAILED**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error: {str(e)}

Please try again or check your internet connection.
"""

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
    
    # ── HEADER ──────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🏥 MedAI Assistant
    ## Medical Diagnostic AI System
    **CPU-Optimized Edition v2.0** | Powered by PyTorch + Gradio
    """)
    
    gr.Markdown("""
    Multimodal AI system for medical analysis:
    - 🖼️ **X-Ray Analysis** - Chest X-ray classification (MobileNetV2)
    - 🗣️ **Symptom Checker** - Disease prediction from symptoms (DistilBERT)
    - 📈 **Vitals Forecasting** - Time-series vital sign prediction (Prophet)
    - 📄 **Report Generator** - Comprehensive diagnostic summaries (Flan-T5)
    """)
    
    gr.Markdown("---")
    
    # ── TABS ────────────────────────────────────────────────────────────
    with gr.Tabs():
        
        # ════════════════════════════════════════════════════════════════
        # TAB 1: X-RAY ANALYSIS
        # ════════════════════════════════════════════════════════════════
        with gr.TabItem("🖼️ X-Ray Analysis", id="xray"):
            gr.Markdown("### Analyze Chest X-Ray Images")
            gr.Markdown("Upload a chest X-ray for AI-powered classification")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Patient Information")
                    xray_patient_name = gr.Textbox(
                        label="Patient Name",
                        placeholder="Enter patient name",
                        value="Unknown"
                    )
                    xray_patient_age = gr.Number(
                        label="Age",
                        minimum=0,
                        maximum=150,
                        value=0,
                        precision=0
                    )
                    xray_patient_gender = gr.Dropdown(
                        choices=["Male", "Female", "Other"],
                        label="Gender",
                        value="Unknown"
                    )
                    
                    gr.Markdown("#### Step 2: Upload X-Ray")
                    xray_image = gr.Image(
                        label="X-Ray Image (JPG/PNG)",
                        type="pil"
                    )
                    
                    xray_analyze_btn = gr.Button(
                        "🧠 Analyze X-Ray",
                        size="lg",
                        variant="primary",
                        scale=1
                    )
                
                with gr.Column():
                    gr.Markdown("#### Results")
                    xray_result_text = gr.Textbox(
                        label="Analysis Results",
                        lines=12,
                        interactive=False
                    )
                    
                    with gr.Row():
                        xray_prediction = gr.Textbox(
                            label="Prediction",
                            interactive=False,
                            scale=1
                        )
                        xray_confidence = gr.Textbox(
                            label="Confidence",
                            interactive=False,
                            scale=1
                        )
                    
                    xray_chart = gr.DataFrame(
                        label="Probability Distribution",
                        interactive=False
                    )
            
            # Connect button to function
            xray_analyze_btn.click(
                fn=analyze_xray,
                inputs=[
                    xray_image,
                    xray_patient_name,
                    xray_patient_age,
                    xray_patient_gender
                ],
                outputs=[
                    xray_result_text,
                    xray_prediction,
                    xray_confidence,
                    xray_chart
                ]
            )
        
        # ════════════════════════════════════════════════════════════════
        # TAB 2: SYMPTOM ANALYSIS
        # ════════════════════════════════════════════════════════════════
        with gr.TabItem("🗣️ Symptom Checker", id="symptoms"):
            gr.Markdown("### Analyze Patient Symptoms")
            gr.Markdown("Enter symptoms to predict potential diseases")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Input Symptoms")
                    gr.Markdown("Enter symptoms separated by commas")
                    symptoms_input = gr.Textbox(
                        label="Symptoms",
                        placeholder="E.g., fever, cough, fatigue, shortness of breath",
                        lines=8
                    )
                    
                    symptoms_analyze_btn = gr.Button(
                        "🧠 Analyze Symptoms",
                        size="lg",
                        variant="primary"
                    )
                
                with gr.Column():
                    gr.Markdown("#### Results")
                    symptoms_result_text = gr.Textbox(
                        label="Analysis Results",
                        lines=12,
                        interactive=False
                    )
                    
                    symptoms_chart = gr.DataFrame(
                        label="Top 3 Predictions",
                        interactive=False
                    )
                    
                    symptoms_prediction = gr.Textbox(
                        label="Top Prediction",
                        interactive=False
                    )
            
            # Connect button to function
            symptoms_analyze_btn.click(
                fn=analyze_symptoms_gradio,
                inputs=[symptoms_input],
                outputs=[
                    symptoms_result_text,
                    symptoms_chart,
                    symptoms_prediction
                ]
            )
        
        # ════════════════════════════════════════════════════════════════
        # TAB 3: VITALS FORECASTING
        # ════════════════════════════════════════════════════════════════
        with gr.TabItem("📈 Vitals Forecasting", id="vitals"):
            gr.Markdown("### Forecast Vital Signs")
            gr.Markdown("Predict next 3 days of patient vital signs")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Step 1: Select Vital & Enter Data")
                    vitals_type = gr.Dropdown(
                        choices=[
                            "Heart Rate (bpm)",
                            "Blood Pressure (mmHg)",
                            "SpO2 (%)",
                            "Temperature (°C)"
                        ],
                        label="Vital Sign Type",
                        value="Heart Rate (bpm)"
                    )
                    
                    gr.Markdown("Enter daily readings (comma-separated)")
                    vitals_input = gr.Textbox(
                        label="Daily Readings",
                        placeholder="E.g., 72, 75, 78, 80, 82",
                        lines=8
                    )
                    
                    vitals_forecast_btn = gr.Button(
                        "📊 Forecast Vitals",
                        size="lg",
                        variant="primary"
                    )
                
                with gr.Column():
                    gr.Markdown("#### Forecast Results")
                    vitals_result_text = gr.Textbox(
                        label="Forecast Results",
                        lines=12,
                        interactive=False
                    )
                    
                    vitals_chart = gr.DataFrame(
                        label="Forecast Data",
                        interactive=False
                    )
                    
                    vitals_summary = gr.Textbox(
                        label="Vital Type",
                        interactive=False
                    )
            
            # Connect button to function
            vitals_forecast_btn.click(
                fn=forecast_vitals_gradio,
                inputs=[vitals_type, vitals_input],
                outputs=[
                    vitals_result_text,
                    vitals_chart,
                    vitals_summary
                ]
            )
        
        # ════════════════════════════════════════════════════════════════
        # TAB 4: GENERATE REPORT
        # ════════════════════════════════════════════════════════════════
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
                    
                    Complete all required analyses before generating report.
                    """)
                    
                    use_openai_checkbox = gr.Checkbox(
                        label="🔑 Use OpenAI (GPT) for reports",
                        value=False,
                        info="Requires OPENAI_API_KEY set in your .env file. Falls back to the free local model if no key is found or the request fails."
                    )

                    report_generate_btn = gr.Button(
                        "🧠 Generate Diagnostic Report",
                        size="lg",
                        variant="primary"
                    )
                
                with gr.Column():
                    gr.Markdown("#### Generated Report")
                    report_output = gr.Textbox(
                        label="Diagnostic Report",
                        lines=20,
                        interactive=False,
                        show_copy_button=True
                    )
            
            # Connect button to function
            report_generate_btn.click(
                fn=generate_report_gradio,
                inputs=[use_openai_checkbox],
                outputs=[report_output]
            )
    
    # ── FOOTER ──────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: gray; padding: 20px;">
        <h3>🏥 MedAI Assistant v2.0 - CPU Optimized Edition (Gradio)</h3>
        <p><b>⚠️ DISCLAIMER:</b> For educational purposes only. Not a substitute for professional medical diagnosis.</p>
        <p><b>Models:</b> MobileNetV2 | DistilBERT | Prophet | Flan-T5-small</p>
        <p><b>Performance:</b> 5-15x faster than original | CPU-optimized for all systems</p>
        <p><small>Built with PyTorch, Transformers, Gradio | © 2024 | MIT License</small></p>
    </div>
    """)

# ────────────────────────────────────────────────────────────────────────────
# LAUNCH APPLICATION
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting MedAI Assistant (Gradio)...")
    print("📍 Open your browser to: http://localhost:7860")
    print("⏳ First run will download models (~1.5GB)...\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
        quiet=False
    )