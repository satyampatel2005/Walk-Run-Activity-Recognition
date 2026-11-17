import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Walk / Run AI", page_icon="🏃", layout="wide")

FEATURES = [
    "wrist",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
LABELS = {0: "Walking", 1: "Running"}

@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("walkrun.csv")

bundle = load_bundle()
model = bundle["model"]
feature_names = bundle.get("feature_names", FEATURES)
accuracy = bundle.get("test_accuracy")
f1 = bundle.get("test_f1_score")

st.title("🏃 Walk / Run Activity Recognition")
st.caption("A machine-learning application that classifies activity from wearable-device motion sensor readings.")

with st.sidebar:
    st.header("About this project")
    st.write(
        "This application uses an XGBoost classification model trained on "
        "accelerometer and gyroscope readings. It predicts one of two classes: "
        "walking (0) or running (1)."
    )
    st.markdown("---")
    st.write("**Input features**")
    st.write(", ".join(FEATURES))
    st.caption("Wrist is retained as its original encoded value (0 or 1); its physical-side mapping is not documented in the supplied data.")

tab_predict, tab_batch, tab_data, tab_info = st.tabs(
    ["Single prediction", "CSV batch prediction", "Dataset overview", "Model information"]
)

with tab_predict:
    st.subheader("Enter sensor readings")
    st.write("Enter one set of sensor values. These are example starting values; replace them with real readings from a compatible sensor.")
    # Use training-data medians as sensible initial values.
    medians = bundle.get("feature_medians", {name: 0.0 for name in FEATURES})
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            wrist = st.selectbox("Wrist (encoded)", options=[0, 1], index=0,
                                 help="The dataset encodes wrist as 0 or 1. The supplied source does not define which value means left or right.")
            ax = st.number_input("Acceleration X", value=float(medians["acceleration_x"]), format="%.6f")
            ay = st.number_input("Acceleration Y", value=float(medians["acceleration_y"]), format="%.6f")
            az = st.number_input("Acceleration Z", value=float(medians["acceleration_z"]), format="%.6f")
        with c2:
            gx = st.number_input("Gyroscope X", value=float(medians["gyro_x"]), format="%.6f")
            gy = st.number_input("Gyroscope Y", value=float(medians["gyro_y"]), format="%.6f")
            gz = st.number_input("Gyroscope Z", value=float(medians["gyro_z"]), format="%.6f")
        submitted = st.form_submit_button("Predict activity", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "wrist": wrist,
            "acceleration_x": ax,
            "acceleration_y": ay,
            "acceleration_z": az,
            "gyro_x": gx,
            "gyro_y": gy,
            "gyro_z": gz,
        }], columns=feature_names)
        prediction = int(model.predict(row)[0])
        title = LABELS.get(prediction, str(prediction))
        st.markdown("---")
        if prediction == 0:
            st.info(f"Predicted activity: **{title}** 🚶")
        else:
            st.success(f"Predicted activity: **{title}** 🏃")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            class_probs = {int(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)}
            confidence = class_probs.get(prediction, max(class_probs.values()))
            st.metric("Model confidence for this prediction", f"{confidence:.2%}")
            st.caption("Confidence is the model's estimated class probability, not a guarantee of correctness.")

with tab_batch:
    st.subheader("Predict multiple records from a CSV")
    st.write("Upload a CSV containing all seven feature columns. The original dataset CSV is also accepted; extra columns such as date, time, username and activity are ignored.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [col for col in FEATURES if col not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                input_data = batch[FEATURES].copy()
                if input_data.isnull().any().any():
                    st.error("The uploaded data contains missing values in the required sensor columns. Please clean them before prediction.")
                else:
                    batch_predictions = model.predict(input_data)
                    result = batch.copy()
                    result["predicted_activity"] = [LABELS.get(int(p), str(p)) for p in batch_predictions]
                    if hasattr(model, "predict_proba"):
                        result["prediction_confidence"] = model.predict_proba(input_data).max(axis=1)
                    st.success(f"Predictions generated for {len(result):,} records.")
                    st.dataframe(result.head(100), use_container_width=True)
                    st.download_button(
                        "Download predictions CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="walk_run_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

with tab_data:
    st.subheader("Dataset overview")
    try:
        data = load_dataset()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Records", f"{len(data):,}")
        col_b.metric("Columns", str(data.shape[1]))
        col_c.metric("Missing values", f"{int(data.isna().sum().sum()):,}")
        if "activity" in data.columns:
            counts = data["activity"].map(LABELS).value_counts()
            st.write("Activity distribution")
            st.bar_chart(counts)
        st.write("Sample records")
        st.dataframe(data.head(10), use_container_width=True)
    except Exception:
        st.warning("Dataset preview is unavailable. The prediction model can still be used.")

with tab_info:
    st.subheader("Model information")
    st.write("**Algorithm:** XGBoost Classifier")
    st.write("**Problem type:** Binary classification")
    st.write("**Target classes:** 0 = Walking, 1 = Running")
    st.write("**Train-test split:** 80:20, random_state=45")
    if accuracy is not None:
        st.metric("Hold-out test accuracy", f"{accuracy:.2%}")
    if f1 is not None:
        st.metric("Hold-out test F1-score", f"{f1:.2%}")
    st.warning(
        "Evaluation note: the original notebook applies IQR-based outlier replacement before the random train-test split. "
        "Sensor records may be temporally related, so a chronological or session-based evaluation is recommended before "
        "claiming real-world performance. This app demonstrates predictions from individual sensor rows; it is not live smartwatch tracking."
    )
    st.markdown("**Feature order used by the saved model**")
    st.code("\\n".join(feature_names))

st.markdown("---")
st.caption("Educational project • Predictions depend on the quality and format of the supplied sensor readings.")
