import streamlit as st
import numpy as np
import tensorflow as tf
import joblib 
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="NeuroDiagnose", page_icon="🏥", layout="wide")

# 2. Load Assets (Cached)
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("models/diabetes_model.h5")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

# Load them once at the start
model, scaler = load_assets()

# ------------------------------------
# 📄 PAGE: HOME
# ------------------------------------
def page_home():
    st.title("🏥 NeuroDiagnose General Hospital")
    st.markdown("### Welcome to the AI Diagnostic Center")
    st.write("Please select a department to begin:")
    
    st.divider()
    
    col1, col2 = st.columns(2)

    with col1:
        st.info("🩸 **Endocrinology**")
        st.write("Diabetes risk assessment using patient vitals.")
        if st.button("Go to Endocrinology (Diabetes)"):
            st.session_state['page'] = 'diabetes'
            st.rerun()
    
    with col2:
        st.info("🩻 **Radiology**")
        st.write("Pneumonia detection using Chest X-Rays.")
        if st.button("Go to Radiology (X-Ray)"):
            st.session_state['page'] = 'radiology'
            st.rerun()

# ------------------------------------
# 📄 PAGE: DIABETES
# ------------------------------------
def page_diabetes():
    st.title("🩸 Endocrinology Department")
    st.caption("AI-Powered Diabetes Risk Assessment")
    
    # Check if assets loaded correctly
    if model is None or scaler is None:
        st.error("⚠️ System Error: Model files not found. Please check your setup.")
        return

    st.divider()

    # --- INPUT GRID ---
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 100)
        bp = st.number_input("Blood Pressure (mmHg)", 0, 150, 70)
        skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin (mu U/mL)", 0, 900, 0)
        bmi = st.number_input("BMI Index", 0.0, 75.0, 25.0)
        pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (Years)", 0, 120, 30)

    st.divider()

    # --- PREDICTION ENGINE ---
    # We move the technical details toggle here for better flow
    col_action, col_debug = st.columns([1, 1])
    
    with col_action:
        analyze_btn = st.button("🔍 Analyze Patient Risk", type="primary") # Primary makes it red/bold
    
    with col_debug:
        show_details = st.checkbox("Show Developer/Technical View")

    if analyze_btn:
        # 1. Prepare Data
        raw_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        scaled_data = scaler.transform(raw_data)
        
        # 2. Predict
        prediction = model.predict(scaled_data)
        probability = prediction[0][0]

        # 3. Output
        if probability > 0.5:
            st.error(f"⚠️ **High Risk Detected** (Confidence: {probability:.2%})")
            st.markdown("The AI model indicates signs consistent with diabetes.")
        else:
            st.success(f"✅ **Low Risk Detected** (Confidence: {(1-probability):.2%})")
            st.markdown("The AI model indicates the patient is likely healthy.")

        # 4. Debug View
        if show_details:
            st.warning("🛠️ **Technical Details**")
            st.json({
                "Model Probability": float(probability),
                "Z-Scores Input": scaled_data.tolist()
            })

# ------------------------------------
# 📄 PAGE: RADIOLOGY (Placeholder)
# ------------------------------------
@st.cache_resource
def load_xray_model():
    return tf.keras.models.load_model("models/xray_model.h5")

def page_radiology():
    st.title("🩻 Radiology Department")
    st.caption("Pneumonia Detection System (Under Construction)")
    
    st.info("Please upload a Chest X-Ray (JPEG/PNG)")
    
    try:
        xray_model = load_xray_model()
    except Exception as e:
        st.error(f"Failed to load X-Ray model: {str(e)}")
        return
    
    uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded X-Ray", width=250)

        with col2:
            st.write("Analyzing image...")

            # --- IMAGE PROCESSING --- 
            img = Image.open(uploaded_file)

            img = img.convert('L') # L due to 1 channel

            img = img.resize((150, 150))

            img_array = np.array(img)

            # Reshape image (1, 150, 150, 1)
            img_input = img_array.reshape(1, 150, 150, 1)

            # --- PREDICTION ENGINE ---
            prediction = xray_model.predict(img_input)
            probability = prediction[0][0]

            st.divider()



            if probability > 0.5:
                st.error(f"⚠️ **Pneumonia Detected** (Confidence: {probability:.2%})")
                st.markdown("The AI model indicates signs consistent with pneumonia.")
            else:
                st.success(f"✅ **Normal Chest X-Ray** (Confidence: {(1-probability):.2%})")
                st.markdown("The AI model indicates the patient is likely healthy.")

                st.progress(int((1-probability)*100))
            
            

# ------------------------------------
# 🧭 MAIN ROUTER & SIDEBAR
# ------------------------------------
def main():
    # Initialize Session State
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    # --- SIDEBAR NAVIGATION ---
    # This persists across all pages!
    with st.sidebar:
        st.title("NeuroDiagnose 🏥")
        st.write("System Navigation")
        
        # We use a radio button for cleaner switching than a dropdown
        page_selection = st.radio(
            "Go to:",
            ["Home", "Endocrinology", "Radiology"],
            index=["home", "diabetes", "radiology"].index(st.session_state['page'])
        )

        # Update state if changed via Sidebar
        if page_selection == "Home" and st.session_state['page'] != 'home':
            st.session_state['page'] = 'home'
            st.rerun()
        elif page_selection == "Endocrinology" and st.session_state['page'] != 'diabetes':
            st.session_state['page'] = 'diabetes'
            st.rerun()
        elif page_selection == "Radiology" and st.session_state['page'] != 'radiology':
            st.session_state['page'] = 'radiology'
            st.rerun()
            
        st.divider()
        st.caption("v1.0.0 | NeuroDiagnose AI")

    # --- PAGE ROUTING ---
    if st.session_state['page'] == 'home':
        page_home()
    elif st.session_state['page'] == 'diabetes':
        page_diabetes()
    elif st.session_state['page'] == 'radiology':
        page_radiology()

if __name__ == "__main__":
    main()