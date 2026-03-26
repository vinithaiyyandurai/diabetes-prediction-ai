
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model  = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥"
)

st.title("🏥 Diabetes Risk Prediction")
st.markdown("*Powered by Gradient Boosting + SHAP*")
st.divider()

st.sidebar.header("Patient Details")
glucose  = st.sidebar.slider("Glucose Level",    0, 200, 120)
bmi      = st.sidebar.slider("BMI",              10.0, 60.0, 25.0)
age      = st.sidebar.slider("Age",              18, 90, 30)
bp       = st.sidebar.slider("Blood Pressure",   0, 130, 70)
insulin  = st.sidebar.slider("Insulin",          0, 900, 80)
skin     = st.sidebar.slider("Skin Thickness",   0, 100, 20)
dpf      = st.sidebar.slider("Diabetes Pedigree",0.0, 2.5, 0.5)
preg     = st.sidebar.slider("Pregnancies",      0, 17, 1)

input_data   = np.array([[preg, glucose, bp,
                           skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)
prediction   = model.predict(input_scaled)[0]
probability  = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error(f"⚠️ High Diabetes Risk!")
    else:
        st.success(f"✅ Low Diabetes Risk!")
    st.metric("Risk Probability", f"{probability*100:.1f}%")
    st.progress(float(probability))

with col2:
    import shap
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    feature_names = ['Pregnancies','Glucose','BloodPressure',
                     'SkinThickness','Insulin','BMI',
                     'DiabetesPedigree','Age']
    fig, ax = plt.subplots(figsize=(8,4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_data[0],
            feature_names = feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    st.caption("Why this prediction? — SHAP")
