
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HIKARI Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "hikari_best_model.pkl"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "Trained model not found! "
            "Please run train_model.py first."
        )
        st.stop()

    return joblib.load(MODEL_PATH)


model_package = load_model()

model = model_package["model"]
required_features = model_package["features"]
best_algorithm = model_package["algorithm"]

# ==========================================
# 3. HEADER
# ==========================================

st.title("🛡️ HIKARI Network Traffic Detection")

st.write(
    "Machine Learning Based Network Intrusion "
    "Detection System"
)

st.success(
    f"Active Detection Algorithm: {best_algorithm}"
)

st.info(
    "Detect whether network traffic is Benign or Attack "
    "using the trained machine learning model."
)

st.divider()

# ==========================================
# 4. DATA PREPROCESSING FUNCTION
# ==========================================

def prepare_data(data):

    X = data.copy()

    X = X.drop(
        columns=[
            'uid',
            'originh',
            'originp',
            'responh',
            'responp',
            'traffic_category',
            'Label',
            'Unnamed: 0',
            'Unnamed: 0.1',
            'Unnamed: 0.2'
        ],
        errors='ignore'
    )

    # Check required model features

    missing_features = [
        col for col in required_features
        if col not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # Keep the exact training feature order

    X = X[required_features]

    # Convert to numeric

    X = X.apply(
        pd.to_numeric,
        errors='coerce'
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


# ==========================================
# 5. PREDICTION FUNCTION
# ==========================================

def predict_traffic(X):

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = np.full(len(predictions), np.nan)

    return predictions, probabilities


# ==========================================
# 6. SELECT DETECTION METHOD
# ==========================================

st.header("Network Traffic Analysis")

tab1, tab2 = st.tabs([
    "🔍 Search by UID",
    "📁 Upload CSV Dataset"
])


# ==========================================
# TAB 1: UID-BASED DETECTION
# ==========================================

with tab1:

    st.subheader("Detect Traffic Using UID")

    st.write(
        "Enter a network flow UID to check whether "
        "the traffic is benign or malicious."
    )

    uid = st.text_input(
        "Enter Network Flow UID",
        placeholder="Example: Cg61Jch3vdz9DBptj"
    )

    dataset_file = st.file_uploader(
        "Upload HIKARI CSV Dataset for UID Search",
        type=["csv"],
        key="uid_dataset"
    )

    if st.button(
        "Check UID",
        use_container_width=True
    ):

        if not uid.strip():

            st.warning("Please enter a UID.")

        elif dataset_file is None:

            st.warning(
                "Please upload the HIKARI dataset "
                "to search for the UID."
            )

        else:

            try:

                data = pd.read_csv(
                    dataset_file,
                    low_memory=False
                )

                if "uid" not in data.columns:
                    st.error(
                        "The uploaded CSV does not contain "
                        "a uid column."
                    )

                else:

                    matching_rows = data[
                        data["uid"].astype(str).str.strip()
                        == uid.strip()
                    ]

                    if matching_rows.empty:

                        st.error(
                            "UID not found in the uploaded dataset."
                        )

                    else:

                        # Use the first matching row

                        row = matching_rows.iloc[[0]]

                        X = prepare_data(row)

                        predictions, probabilities = predict_traffic(X)

                        prediction = int(predictions[0])

                        st.divider()

                        st.subheader("Detection Result")

                        if prediction == 1:

                            st.error(
                                "🚨 ATTACK DETECTED"
                            )

                            st.write(
                                "The selected network flow "
                                "has been classified as malicious."
                            )

                        else:

                            st.success(
                                "✅ BENIGN TRAFFIC"
                            )

                            st.write(
                                "The selected network flow "
                                "has been classified as benign."
                            )

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "UID",
                            uid
                        )

                        col2.metric(
                            "Prediction",
                            "Attack" if prediction == 1 else "Benign"
                        )

                        if not np.isnan(probabilities[0]):

                            st.metric(
                                "Predicted Attack Probability",
                                f"{probabilities[0] * 100:.2f}%"
                            )

                        st.caption(
                            f"Detection Algorithm: {best_algorithm}"
                        )

            except Exception as e:

                st.error(
                    f"Error during prediction: {e}"
                )


# ==========================================
# TAB 2: CSV-BASED DETECTION
# ==========================================

with tab2:

    st.subheader("Upload Network Traffic Dataset")

    uploaded_file = st.file_uploader(
        "Choose HIKARI CSV File",
        type=["csv"],
        key="batch_dataset"
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

            st.success("Dataset uploaded successfully!")

            st.write("Dataset Preview")

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            if st.button(
                "Predict Network Traffic",
                use_container_width=True
            ):

                X = prepare_data(data)

                predictions, probabilities = predict_traffic(X)

                results = data.copy()

                results["Prediction"] = predictions

                results["Traffic Status"] = [
                    "Attack" if p == 1 else "Benign"
                    for p in predictions
                ]

                results["Attack Probability"] = probabilities

                benign = int(
                    np.sum(predictions == 0)
                )

                attack = int(
                    np.sum(predictions == 1)
                )

                st.divider()

                st.header("Prediction Dashboard")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Records",
                    len(predictions)
                )

                col2.metric(
                    "Benign Traffic",
                    benign
                )

                col3.metric(
                    "Attack Traffic",
                    attack
                )

                st.subheader("Traffic Distribution")

                chart_data = pd.DataFrame({
                    "Traffic": ["Benign", "Attack"],
                    "Records": [benign, attack]
                })

                st.bar_chart(
                    chart_data.set_index("Traffic")
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    csv,
                    "hikari_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error processing dataset: {e}"
            )


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "HIKARI 2021 | Machine Learning Based "
    "Network Intrusion Detection System"
)
