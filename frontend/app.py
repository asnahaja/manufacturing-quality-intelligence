import streamlit as st
import requests
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Manufacturing Quality Intelligence",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Manufacturing Quality Intelligence System")
st.caption(
    "AI-powered predictive maintenance, anomaly detection "
    "and manufacturing quality prediction"
)

st.divider()

st.subheader("Machine Information")
st.write("Enter the machine readings below and run the AI predictions.")

with st.form("machine_input_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        machine_id = st.text_input("Machine ID", value="M001")
        production_line = st.selectbox(
            "Production Line", ["Line A", "Line B", "Line C"]
        )
        machine_type = st.selectbox(
            "Machine Type", ["Type A", "Type B", "Type C"]
        )
        machine_age_years = st.number_input(
            "Machine Age (years)", min_value=0.0,
            value=3.0, step=0.5
        )
        hours_since_maintenance = st.number_input(
            "Hours Since Maintenance", min_value=0.0,
            value=120.0
        )
        maintenance_event = st.selectbox(
            "Maintenance Event", [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:
        load_factor = st.number_input(
            "Load Factor", min_value=0.0, value=0.75
        )
        temperature_c = st.number_input(
            "Temperature (°C)", value=65.0
        )
        vibration_mm_s = st.number_input(
            "Vibration (mm/s)", min_value=0.0, value=2.5
        )
        pressure_psi = st.number_input(
            "Pressure (psi)", min_value=0.0, value=100.0
        )
        current_a = st.number_input(
            "Current (A)", min_value=0.0, value=10.0
        )
        rpm = st.number_input(
            "RPM", min_value=0.0, value=1500.0
        )

    with col3:
        cycle_time_sec = st.number_input(
            "Cycle Time (seconds)", min_value=0.1,
            value=30.0
        )
        tool_wear_pct = st.number_input(
            "Tool Wear (%)", min_value=0.0,
            max_value=100.0, value=35.0
        )
        machine_health_index = st.number_input(
            "Machine Health Index", value=0.80
        )
        quality_score = st.number_input(
            "Quality Score", value=85.0
        )
        reading_date = st.date_input(
            "Reading Date", value=datetime.now().date()
        )
        reading_time = st.time_input(
            "Reading Time", value=datetime.now().time().replace(
                second=0, microsecond=0
            )
        )

    submitted = st.form_submit_button(
        "Run All Predictions", type="primary", use_container_width=True
    )

if submitted:

    timestamp = (
        f"{reading_date.strftime('%d-%m-%Y')} "
        f"{reading_time.strftime('%H:%M')}"
    )

    payload = {
        "machine_id": machine_id,
        "production_line": production_line,
        "machine_type": machine_type,
        "machine_age_years": machine_age_years,
        "hours_since_maintenance": hours_since_maintenance,
        "maintenance_event": maintenance_event,
        "load_factor": load_factor,
        "temperature_c": temperature_c,
        "vibration_mm_s": vibration_mm_s,
        "pressure_psi": pressure_psi,
        "current_a": current_a,
        "rpm": rpm,
        "cycle_time_sec": cycle_time_sec,
        "tool_wear_pct": tool_wear_pct,
        "machine_health_index": machine_health_index,
        "quality_score": quality_score,
        "timestamp": timestamp
    }

    endpoints = [
        ("Predictive Maintenance", "/predict/failure"),
        ("Anomaly Detection", "/predict/anomaly"),
        ("Quality Prediction", "/predict/quality")
    ]

    st.divider()
    st.subheader("Prediction Results")

    try:
        with st.spinner("Analysing machine readings..."):

            results = []

            for title, endpoint in endpoints:
                response = requests.post(
                    f"{API_URL}{endpoint}",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                results.append((title, response.json()))

        columns = st.columns(3)

        for column, (title, result) in zip(columns, results):
            with column:
                st.subheader(title)

                label = result.get("label", "Unknown")
                probability = result.get("probability")

                if title == "Predictive Maintenance":
                    st.metric("Prediction", label)
                elif title == "Anomaly Detection":
                    st.metric("Machine Status", label)
                else:
                    st.metric("Quality Status", label)

                if probability is not None:
                    st.progress(
                        max(0.0, min(1.0, float(probability)))
                    )
                    st.caption(
                        f"Model probability: {float(probability):.1%}"
                    )

                st.caption(
                    f"Predicted class: {result.get('prediction', 'N/A')}"
                )

        st.success("All three predictions completed successfully.")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Start the backend on port 8000 "
            "and try again."
        )

    except requests.exceptions.HTTPError as error:
        st.error(f"The API returned an error: {error}")
        if error.response is not None:
            st.code(error.response.text)

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

st.divider()
st.caption(
    "Academic project demonstration • Predictions depend on the "
    "trained models and supplied input data."
)
