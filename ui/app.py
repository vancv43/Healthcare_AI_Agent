import gradio as gr
from core.explainer import AIExplainer
from core.memory import generate_patient_id, update_patient_memory, get_patient_history
from main import MedicalKnowledgeBase, DoctorRecommender, InsurancePredictor


# =========================
# INIT SERVICES
# =========================
diagnostic_engine = MedicalKnowledgeBase()
doctor_recommender = DoctorRecommender()
insurance_estimator = InsurancePredictor()
ai_explainer = AIExplainer()


# =========================
# HELPERS
# =========================
def calculate_bmi(height_cm, weight_kg):
    height_m = float(height_cm) / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    bmi = float(weight_kg) / (height_m ** 2)
    return round(bmi, 2)


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def format_doctors(doctors):
    if not doctors or isinstance(doctors, str):
        return "No doctors found."

    lines = []
    for idx, doc in enumerate(doctors, start=1):
        lines.append(
            f"{idx}. {doc.get('name', 'Unknown')}\n"
            f"   Specialty: {doc.get('specialty', 'Unknown')}\n"
            f"   City: {doc.get('city', 'Unknown')}\n"
            f"   Rating: {doc.get('rating', 'N/A')}\n"
        )
    return "\n".join(lines)


def format_history(history):
    if not history:
        return "No previous patient history found."

    visits = history.get("history", [])
    if not visits:
        return "No previous patient history found."

    demographics = history.get("demographics", {})

    lines = []
    lines.append("Patient Visit Timeline")
    lines.append("=" * 46)

    for idx, visit in enumerate(visits, start=1):
        diag = visit.get("diagnosis", {})
        lines.append(f"Visit {idx}")
        lines.append(f"Timestamp: {visit.get('timestamp', 'Unknown')}")
        lines.append(f"Symptoms: {visit.get('symptoms', 'Unknown')}")
        lines.append(f"Condition: {diag.get('condition', 'Unknown')}")
        lines.append(f"Specialty: {diag.get('specialty', 'Unknown')}")
        lines.append(f"Insurance estimate: {visit.get('insurance_price', 'Unknown')}")
        if demographics.get("bmi") is not None:
            lines.append(f"Stored BMI: {demographics.get('bmi')}")
        lines.append("-" * 46)

    return "\n".join(lines)


def build_hero():
    return """
    <div class="hero-wrap">
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-title">Healthcare AI Agent</div>
            <div class="hero-subtitle">
                A modern clinical decision-support demo that transforms patient inputs into
                actionable health insight, doctor recommendation, insurance estimation, and AI explanation.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">Patient Intake</span>
                <span class="hero-badge">AI Insight Layer</span>
                <span class="hero-badge">Doctor Matching</span>
                <span class="hero-badge">Insurance Estimation</span>
                <span class="hero-badge">Patient Memory</span>
            </div>
        </div>
    </div>
    """


def build_status_html(condition, specialty, advice, triage="standard"):
    triage_map = {
        "standard": "#22c55e",
        "urgent": "#f59e0b",
        "critical": "#ef4444"
    }
    triage_color = triage_map.get(str(triage).lower(), "#22c55e")

    return f"""
    <div class="status-card">
        <div class="status-header">
            <div>
                <div class="status-caption">Decision Support Snapshot</div>
                <div class="status-title">Clinical Triage Overview</div>
            </div>
            <div class="status-pill" style="background:{triage_color};">{str(triage).upper()}</div>
        </div>

        <div class="status-grid">
            <div class="metric-card">
                <div class="metric-label">Matched Condition</div>
                <div class="metric-value">{condition}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recommended Specialty</div>
                <div class="metric-value">{specialty}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Advice</div>
                <div class="metric-value">{advice}</div>
            </div>
        </div>
    </div>
    """


def build_bmi_html(height_cm, weight_kg):
    try:
        bmi = calculate_bmi(height_cm, weight_kg)
        category = bmi_category(bmi)
        return f"""
        <div class="mini-stat-card">
            <div class="mini-stat-label">BMI</div>
            <div class="mini-stat-number">{bmi}</div>
            <div class="mini-stat-sub">{category}</div>
        </div>
        """
    except Exception:
        return """
        <div class="mini-stat-card">
            <div class="mini-stat-label">BMI</div>
            <div class="mini-stat-number">--</div>
            <div class="mini-stat-sub">Invalid input</div>
        </div>
        """


def build_insurance_html(cost_text):
    return f"""
    <div class="mini-stat-card">
        <div class="mini-stat-label">Insurance Estimate</div>
        <div class="mini-stat-number small">{cost_text}</div>
        <div class="mini-stat-sub">Predicted support output</div>
    </div>
    """


def analyze(name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai):
    try:
        if not str(name).strip() or not str(email).strip() or not str(symptoms).strip():
            msg = "Please fill in name, email, and symptoms."
            return (
                build_status_html("N/A", "N/A", msg, "standard"),
                build_insurance_html("N/A"),
                build_bmi_html(height_cm, weight_kg),
                msg,
                msg,
                msg,
                msg,
                msg
            )

        patient_id = generate_patient_id(name, email)

        # 1. Diagnosis
        result = diagnostic_engine.search_symptoms(symptoms)

        # 2. Doctor recommendation
        doctors = doctor_recommender.find_doctor(result["specialty"], city)

        # 3. BMI + insurance
        bmi = calculate_bmi(height_cm, weight_kg)
        smoker_val = 1 if smoker == "yes" else 0
        region_code = 1
        est_cost = insurance_estimator.estimate_cost(age, bmi, smoker_val, region_code, int(children))
        insurance_text = f"${est_cost}"

        # 4. AI explainer
        top_diagnosis = {
            "diagnosis": result["condition"],
            "specialty": result["specialty"],
            "advice": result["advice"],
            "triage_level": "standard",
            "recommended_tests": "N/A"
        }

        if use_ai and not ai_explainer.model_loaded:
            ai_explainer.enable_local_model()

        payload = {
            "top_diagnosis": top_diagnosis,
            "doctors": doctors if isinstance(doctors, list) else [],
            "insurance_text": insurance_text,
            "history_summary": "",
        }
        ai_text = ai_explainer.explain(payload, use_ai=use_ai)

        # 5. Memory
        demographics = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "city": city
        }

        update_patient_memory(
            patient_id=patient_id,
            demographics=demographics,
            symptoms=symptoms,
            diagnosis_result=result,
            insurance_price=est_cost,
            doctors=doctors if isinstance(doctors, list) else []
        )

        history = get_patient_history(patient_id)

        detailed_analysis = (
            f"Matched condition: {result['condition']}\n"
            f"Recommended specialty: {result['specialty']}\n"
            f"Advice: {result['advice']}\n"
            f"Triage level: standard\n"
            f"Recommended tests: N/A"
        )

        summary = f"Insight: {result['condition']} | Specialty: {result['specialty']}"
        doctor_text = format_doctors(doctors)
        history_text = format_history(history)

        return (
            build_status_html(result["condition"], result["specialty"], result["advice"], "standard"),
            build_insurance_html(insurance_text),
            build_bmi_html(height_cm, weight_kg),
            summary,
            detailed_analysis,
            ai_text,
            doctor_text,
            history_text
        )

    except Exception as e:
        msg = f"Error during analysis: {e}"
        return (
            build_status_html("N/A", "N/A", msg, "standard"),
            build_insurance_html("N/A"),
            build_bmi_html(height_cm, weight_kg),
            msg,
            msg,
            msg,
            msg,
            msg
        )


def show_history(name, email):
    try:
        if not str(name).strip() or not str(email).strip():
            return "Please enter both name and email."
        patient_id = generate_patient_id(name, email)
        history = get_patient_history(patient_id)
        return format_history(history)
    except Exception as e:
        return f"Error while loading history: {e}"


def refresh_bmi(height_cm, weight_kg):
    return build_bmi_html(height_cm, weight_kg)


# =========================
# THEME
# =========================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    spacing_size="md",
    text_size="md"
).set(
    body_background_fill="#eef4f8",
    body_background_fill_dark="#0f172a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#111827",
    block_border_width="1px",
    block_border_color="#dbe7f0",
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e8f0ff",
    button_secondary_background_fill_hover="#dbeafe",
    button_secondary_text_color="#0f172a",
    input_background_fill="#ffffff",
    input_border_color="#d6e0ea",
    input_border_color_focus="#60a5fa"
)


custom_css = """
body {
    background: linear-gradient(180deg, #eef4f8 0%, #f7fbff 100%);
}

.gradio-container {
    max-width: 1420px !important;
    margin: 0 auto !important;
    padding-top: 26px !important;
    padding-bottom: 28px !important;
}

.hero-wrap {
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    min-height: 240px;
    padding: 36px 42px;
    background:
        linear-gradient(120deg, rgba(4, 15, 40, 0.92) 0%, rgba(10, 74, 89, 0.90) 42%, rgba(50, 157, 211, 0.92) 100%);
    box-shadow: 0 18px 55px rgba(15, 23, 42, 0.12);
    margin-bottom: 22px;
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at top right, rgba(255,255,255,0.12), transparent 28%),
        radial-gradient(circle at bottom left, rgba(255,255,255,0.08), transparent 30%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    color: #ffffff !important;
    font-size: 56px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.18);
}

.hero-subtitle {
    color: rgba(255,255,255,0.96) !important;
    font-size: 24px;
    line-height: 1.55;
    max-width: 1280px;
    margin-bottom: 26px;
    text-shadow: 0 1px 4px rgba(0,0,0,0.14);
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}

.hero-badge {
    color: #ffffff !important;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 12px 18px;
    font-size: 15px;
    font-weight: 700;
    backdrop-filter: blur(6px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
}

.panel-card {
    border-radius: 24px !important;
    border: 1px solid #dde8f2 !important;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}

.section-heading {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

.section-text {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 10px;
}

.status-card {
    border-radius: 24px;
    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid #d9e7f4;
    padding: 22px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.04);
}

.status-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}

.status-caption {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 4px;
}

.status-title {
    font-size: 22px;
    color: #0f172a;
    font-weight: 800;
}

.status-pill {
    color: #ffffff !important;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.metric-card {
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    background: #ffffff;
    padding: 16px;
}

.metric-label {
    font-size: 12px;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 18px;
    color: #0f172a;
    font-weight: 800;
    line-height: 1.45;
}

.mini-stat-card {
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
    border: 1px solid #d9e7f4;
    padding: 20px;
    min-height: 130px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.04);
}

.mini-stat-label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 10px;
}

.mini-stat-number {
    font-size: 34px;
    color: #0f172a;
    font-weight: 800;
    line-height: 1.1;
}

.mini-stat-number.small {
    font-size: 24px;
}

.mini-stat-sub {
    margin-top: 10px;
    font-size: 14px;
    color: #475569;
    line-height: 1.5;
}

.note-box {
    font-size: 13px;
    line-height: 1.7;
    color: #475569;
    background: #f8fbfe;
    border: 1px dashed #c9d8e6;
    border-radius: 18px;
    padding: 14px 16px;
    margin-top: 10px;
}

textarea, input, .gr-dropdown {
    border-radius: 16px !important;
}

button {
    border-radius: 16px !important;
    font-weight: 800 !important;
}

.gr-tab-nav button {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

@media (max-width: 1100px) {
    .hero-title {
        font-size: 42px;
    }

    .hero-subtitle {
        font-size: 18px;
    }

    .status-grid {
        grid-template-columns: 1fr;
    }
}
"""


# =========================
# UI
# =========================
with gr.Blocks(
    title="Healthcare AI Agent",
    theme=theme,
    css=custom_css,
    fill_width=True
) as demo:

    gr.HTML(build_hero())

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Patient Intake")
                gr.Markdown("Enter patient profile and symptom details to generate health insight.")

                with gr.Row():
                    name = gr.Textbox(label="Full Name", placeholder="Enter full name")
                    email = gr.Textbox(label="Email", placeholder="Enter email address")

                with gr.Row():
                    age = gr.Number(label="Age", value=35, precision=0)
                    sex = gr.Dropdown(choices=["male", "female"], value="male", label="Sex")
                    city = gr.Dropdown(
                        choices=["Bangkok", "Seoul", "Hanoi", "Chiang Mai"],
                        value="Bangkok",
                        label="City"
                    )

                with gr.Row():
                    height_cm = gr.Number(label="Height (cm)", value=170)
                    weight_kg = gr.Number(label="Weight (kg)", value=70)
                    children = gr.Number(label="Children", value=1, precision=0)

                with gr.Row():
                    smoker = gr.Dropdown(choices=["yes", "no"], value="no", label="Smoker")
                    region = gr.Dropdown(
                        choices=["northeast", "northwest", "southeast", "southwest"],
                        value="southeast",
                        label="Region"
                    )
                    use_ai = gr.Checkbox(label="Enable local AI explanation", value=False)

                symptoms = gr.Textbox(
                    label="Describe Symptoms",
                    lines=5,
                    placeholder="Example: fever, cough, sore throat, chest discomfort..."
                )

                with gr.Row():
                    analyze_button = gr.Button("Analyze Patient", variant="primary", size="lg")
                    history_button = gr.Button("Show Patient History", variant="secondary", size="lg")

                gr.HTML("""
                <div class="note-box">
                    <strong>Disclaimer:</strong> This system is for educational and AI decision-support demonstration only.
                    It is not a substitute for diagnosis, emergency response, or treatment by a licensed healthcare professional.
                </div>
                """)

        with gr.Column(scale=4):
            with gr.Row():
                bmi_box = gr.HTML(build_bmi_html(170, 70))
                insurance_box = gr.HTML(build_insurance_html("N/A"))

            status_box = gr.HTML(
                build_status_html(
                    "Not analyzed yet",
                    "N/A",
                    "Please submit patient information to begin analysis.",
                    "standard"
                )
            )

    with gr.Tabs():
        with gr.Tab("Summary"):
            summary_output = gr.Textbox(
                label="Health Insight Summary",
                lines=4,
                interactive=False
            )

        with gr.Tab("Full Analysis"):
            detailed_output = gr.Textbox(
                label="Detailed Analysis",
                lines=10,
                interactive=False
            )

        with gr.Tab("AI Explanation"):
            ai_output = gr.Textbox(
                label="Explanation Layer Output",
                lines=12,
                interactive=False
            )

        with gr.Tab("Doctor Recommendations"):
            doctor_output = gr.Textbox(
                label="Recommended Doctors",
                lines=12,
                interactive=False
            )

        with gr.Tab("Patient History"):
            history_output = gr.Textbox(
                label="Patient Visit Timeline",
                lines=14,
                interactive=False
            )

    height_cm.change(
        refresh_bmi,
        inputs=[height_cm, weight_kg],
        outputs=bmi_box
    )

    weight_kg.change(
        refresh_bmi,
        inputs=[height_cm, weight_kg],
        outputs=bmi_box
    )

    analyze_button.click(
        fn=analyze,
        inputs=[name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai],
        outputs=[status_box, insurance_box, bmi_box, summary_output, detailed_output, ai_output, doctor_output, history_output]
    )

    history_button.click(
        fn=show_history,
        inputs=[name, email],
        outputs=history_output
    )


if __name__ == "__main__":
    demo.launch()