# ============================================================
# EcoType: Forest Cover Type Prediction - FINAL FIXED APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("forest_cover_model.pkl")
    selected_features = joblib.load("selected_features.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, selected_features, label_encoder

model, selected_features, label_encoder = load_artifacts()

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------

st.set_page_config(
    page_title="EcoType | Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    """
    <h1 style="text-align:center;">🌲 EcoType</h1>
    <h4 style="text-align:center;">Forest Cover Type Prediction System</h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab1, tab2 = st.tabs(
    ["🔮 Single Prediction",  "ℹ️ About Project"]
)

# ============================================================
# TAB 1: Single Prediction
# ============================================================

with tab1:
    st.subheader("Enter Environmental & Cartographic Features")

    # --------------------------------------------------------
    # Auto-fill example values (FIXED)
    # --------------------------------------------------------

    if st.button("✨ Auto-Fill Example Values"):
        example_values = {}

        for feature in selected_features:
            if "Elevation" in feature:
                example_values[feature] = float(np.random.randint(500, 3500))
            elif "Distance" in feature:
                example_values[feature] = float(np.random.randint(0, 8000))
            elif "Hillshade" in feature:
                example_values[feature] = float(np.random.randint(50, 250))
            else:
                example_values[feature] = float(np.random.randint(0, 3000))

        st.session_state["example_values"] = example_values

    input_data = {}
    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Input fields with validation (ALL FLOATS)
    # --------------------------------------------------------

    for i, feature in enumerate(selected_features):
        default_val = st.session_state.get(
            "example_values", {}
        ).get(feature, 0.0)

        if "Elevation" in feature:
            min_val, max_val = 0.0, 4000.0
        elif "Distance" in feature:
            min_val, max_val = 0.0, 10000.0
        elif "Hillshade" in feature:
            min_val, max_val = 0.0, 255.0
        else:
            min_val, max_val = 0.0, 5000.0

        if i % 3 == 0:
            input_data[feature] = col1.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=float(default_val)
            )
        elif i % 3 == 1:
            input_data[feature] = col2.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=float(default_val)
            )
        else:
            input_data[feature] = col3.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=float(default_val)
            )

    input_df = pd.DataFrame([input_data])

    st.markdown("### 🔍 Input Preview")
    st.dataframe(input_df, use_container_width=True)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button("🚀 Predict Forest Cover Type"):
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100

        st.success(f"🌳 **Predicted Forest Cover Type:** {predicted_label}")
        st.info(f"📈 **Prediction Confidence:** {confidence:.2f}%")

        prob_df = pd.DataFrame(
            probabilities,
            columns=label_encoder.classes_
        ).T

        st.markdown("### 📊 Class Probability Distribution")
        st.bar_chart(prob_df)


# ============================================================
# TAB 2: About
# ============================================================

with tab2:
    st.subheader("About EcoType")

    st.markdown(
        """
        **EcoType** is an end-to-end machine learning system for predicting
        forest cover types using cartographic and environmental data.

        **Key Highlights**
        - Automatic preprocessing & skewness handling  
        - SMOTE for class imbalance  
        - Feature selection via Random Forest importance  
        - Optimized Random Forest model  
        - Advanced Streamlit deployment  

        **Applications**
        - Forest management  
        - Wildfire risk analysis  
        - Land-use planning  
        - Environmental research  
        """
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🌲 EcoType | Built with Streamlit</p>",
    unsafe_allow_html=True
)
