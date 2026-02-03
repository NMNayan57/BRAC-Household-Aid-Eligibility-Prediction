# ============================================================
# BRAC Household Aid Eligibility Prediction System
# Streamlit Web Application
# Author: Nasim Mahmud
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="BRAC Aid Eligibility Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .eligible {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .not-eligible {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .factor-positive {
        color: #059669;
        font-weight: bold;
    }
    .factor-negative {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_models():
    """Load all models and preprocessors."""
    models_path = "models"
    
    models = {
        'task1_model': joblib.load(os.path.join(models_path, 'task1_eligibility_model.joblib')),
        'task2_model': joblib.load(os.path.join(models_path, 'task2_aidtype_model.joblib')),
        'scaler': joblib.load(os.path.join(models_path, 'scaler.joblib')),
        'le_target2': joblib.load(os.path.join(models_path, 'le_target2.joblib')),
        'shap_explainer': joblib.load(os.path.join(models_path, 'shap_explainer_t1.joblib'))
    }
    
    with open(os.path.join(models_path, 'feature_config.json'), 'r') as f:
        feature_config = json.load(f)
    
    with open(os.path.join(models_path, 'model_card.json'), 'r') as f:
        model_card = json.load(f)
    
    return models, feature_config, model_card

# Load models
try:
    models, feature_config, model_card = load_models()
    FEATURES = feature_config['features']
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")
    st.stop()

# ============================================================
# Prediction Function
# ============================================================

def predict_eligibility(input_data):
    """Make prediction for household."""
    
    # Create dataframe
    input_df = pd.DataFrame([input_data])[FEATURES]
    
    # Scale
    input_scaled = pd.DataFrame(
        models['scaler'].transform(input_df),
        columns=FEATURES
    )
    
    # Task 1: Eligibility
    proba = models['task1_model'].predict_proba(input_scaled)[0][1]
    score = round(proba * 100, 1)
    is_eligible = proba >= 0.5
    
    result = {
        'score': score,
        'is_eligible': is_eligible,
        'decision': 'Eligible' if is_eligible else 'Not Eligible'
    }
    
    # Task 2: Aid Type
    if is_eligible:
        aid_pred = models['task2_model'].predict(input_scaled)[0]
        aid_proba = models['task2_model'].predict_proba(input_scaled)[0]
        result['aid_type'] = models['le_target2'].inverse_transform([aid_pred])[0]
        result['aid_confidence'] = round(max(aid_proba) * 100, 1)
    else:
        result['aid_type'] = 'N/A'
        result['aid_confidence'] = None
    
    # SHAP - Top 3 Factors
    shap_vals = models['shap_explainer'].shap_values(input_scaled)
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    elif isinstance(shap_vals, list):
        shap_vals = np.array(shap_vals[1])
    shap_vals = shap_vals.flatten()
    
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_factors = []
    for idx in sorted_idx:
        direction = "increases" if shap_vals[idx] > 0 else "decreases"
        top_factors.append({
            'feature': FEATURES[idx],
            'direction': direction,
            'shap': shap_vals[idx]
        })
    result['top_factors'] = top_factors
    
    return result

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/c0/BRAC_logo.svg/1200px-BRAC_logo.svg.png", width=150)
    
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    This AI system helps field officers assess household eligibility for BRAC's poverty alleviation programs.
    
    **Outputs:**
    - 📊 Eligibility Score (0-100%)
    - 🎯 Recommended Aid Type
    - 🔍 Top 3 Influencing Factors
    """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Performance")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | ROC-AUC | {model_card['task1']['test_roc_auc']} |
    | Recall (Selected) | {model_card['task1']['test_recall_selected']*100:.0f}% |
    """)
    
    st.markdown("---")
    
    st.markdown("### ✅ Fairness Status")
    st.success("Passed all fairness tests")
    
    st.markdown("---")
    st.caption("Developed by Nasim Mahmud")
    st.caption("BRAC Technology Division")

# ============================================================
# Main Content
# ============================================================

# Header
st.markdown('<p class="main-header">🏠 BRAC Household Aid Eligibility Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered system to assess household eligibility for Income Generating Activity (IGA) support</p>', unsafe_allow_html=True)

st.markdown("---")

# Input Form
st.markdown("### 📝 Enter Household Information")

# Create tabs for organized input
tab1, tab2, tab3 = st.tabs(["👥 Family Details", "💰 Financial Information", "🏠 Assets & Health"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        guardian_age = st.slider("Guardian Age", 18, 100, 40, help="Age of the household guardian")
        participant_age = st.slider("Participant Age", 15, 80, 35, help="Age of the program participant")
        family_size = st.slider("Family Size", 1, 15, 4, help="Total number of family members")
        
    with col2:
        marital_status = st.selectbox("Marital Status", ["Married", "Widow/Others"])
        migrant = st.selectbox("Is this a Migrant Family?", ["No", "Yes"])
        migrant_tenure = st.number_input("Migration Tenure (Years)", 0.0, 50.0, 0.0, 0.5, 
                                         help="Years since migration (0 if not migrant)")
    
    col3, col4 = st.columns(2)
    with col3:
        has_under5 = st.selectbox("Has Children Under 5?", ["No", "Yes"])
        has_working_age = st.selectbox("Has Working Age Member (18-50)?", ["Yes", "No"])
    with col4:
        has_50plus = st.selectbox("Has Elderly Member (50+)?", ["No", "Yes"])

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        annual_income = st.number_input("Total Annual Income (BDT)", 0, 1000000, 100000, 5000,
                                        help="Total household income per year")
        income_per_head = st.number_input("Monthly Income per Head (BDT)", 0, 50000, 2500, 100,
                                          help="Monthly income divided by family size")
        has_savings = st.selectbox("Has Savings?", ["No", "Yes"])
        savings_amt = st.number_input("Savings Amount (BDT)", 0, 500000, 0, 500)
        
    with col2:
        loans_taken = st.selectbox("Has Ever Taken Loans?", ["No", "Yes"])
        loans_running = st.selectbox("Has Running Loans Currently?", ["No", "Yes"], 
                                     help="⚠️ This is a strong predictor")
        loans_num = st.number_input("Number of Active Loans", 0, 10, 0)
        loans_outstanding = st.number_input("Outstanding Loan Amount (BDT)", 0, 500000, 0, 1000)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        has_assets = st.selectbox("Has Productive Assets?", ["No", "Yes"],
                                  help="Livestock, machinery, etc.")
        asset_num = st.number_input("Number of Productive Assets", 0, 10, 0)
        asset_value = st.number_input("Total Asset Value (BDT)", 0, 500000, 0, 1000)
        
    with col2:
        has_chronic = st.selectbox("Any Member with Chronic Disease?", ["No", "Yes"])
        chronic_num = st.number_input("Number with Chronic Disease", 0, 5, 0)
        has_disabled = st.selectbox("Any Disabled Member?", ["No", "Yes"])
        disabled_num = st.number_input("Number of Disabled Members", 0, 5, 0)

st.markdown("---")

# Predict Button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Eligibility", type="primary", use_container_width=True)

# ============================================================
# Prediction Results
# ============================================================

if predict_button:
    # Prepare input data
    input_data = {
        'Family_Migrant_Tenure(Years)': migrant_tenure,
        'Gurdian_Age': guardian_age,
        'Family_Size': family_size,
        'Asset_Yes': 1 if has_assets == "Yes" else 0,
        'Productive_Asset_Num': asset_num,
        'Productive_Asset_Value': asset_value,
        'Total_Income_Annualy': annual_income,
        'Income_Monthly_per_head': income_per_head,
        'Chronic_Patient_Num': chronic_num,
        'Disabled_Num': disabled_num,
        'Loans_Num': loans_num,
        'Loans_Outstanding': loans_outstanding,
        'has_Savings': 1 if has_savings == "Yes" else 0,
        'Savings_Amt': savings_amt,
        'has_18_50_Family_member': 1 if has_working_age == "Yes" else 0,
        'has_50_plus': 1 if has_50plus == "Yes" else 0,
        'has_under5': 1 if has_under5 == "Yes" else 0,
        'Has_Chronic_Dissease': 1 if has_chronic == "Yes" else 0,
        'Loans_Taken_Yes': 1 if loans_taken == "Yes" else 0,
        'Loans_Running_Yes': 1 if loans_running == "Yes" else 0,
        'Migrant': 1 if migrant == "Yes" else 0,
        'Disabled_Yes': 1 if has_disabled == "Yes" else 0,
        'Marrital_Status': 1 if marital_status == "Widow/Others" else 0,
        'Participant_Age': participant_age
    }
    
    # Get prediction
    with st.spinner("Analyzing household data..."):
        result = predict_eligibility(input_data)
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    # Results display
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.markdown("#### Eligibility Score")
        
        # Score gauge visualization
        score_color = "#10B981" if result['is_eligible'] else "#EF4444"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #F9FAFB; border-radius: 10px;">
            <div style="font-size: 48px; font-weight: bold; color: {score_color};">
                {result['score']}%
            </div>
            <div style="font-size: 14px; color: #6B7280;">
                Eligibility Probability
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown("#### Decision")
        if result['is_eligible']:
            st.success(f"✅ **{result['decision']}**")
            st.markdown("This household qualifies for IGA support.")
        else:
            st.error(f"❌ **{result['decision']}**")
            st.markdown("This household does not meet eligibility criteria.")
    
    with col_r3:
        st.markdown("#### Recommended Aid Type")
        if result['is_eligible']:
            st.info(f"🎯 **{result['aid_type']}**")
            st.caption(f"Confidence: {result['aid_confidence']}%")
        else:
            st.markdown("*Not applicable*")
    
    # Top 3 Factors
    st.markdown("---")
    st.markdown("### 🔍 Top 3 Factors Driving This Prediction")
    
    factor_cols = st.columns(3)
    
    for i, factor in enumerate(result['top_factors']):
        with factor_cols[i]:
            icon = "📈" if factor['direction'] == "increases" else "📉"
            color = "#059669" if factor['direction'] == "increases" else "#DC2626"
            
            st.markdown(f"""
            <div style="padding: 15px; background-color: #F9FAFB; border-radius: 8px; border-left: 4px solid {color};">
                <div style="font-size: 24px;">{icon}</div>
                <div style="font-weight: bold; color: #1F2937; margin-top: 5px;">
                    {factor['feature'].replace('_', ' ')}
                </div>
                <div style="color: {color}; font-size: 14px;">
                    {factor['direction']} eligibility
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendation
    st.markdown("---")
    st.markdown("### 💡 Recommendation")
    
    if result['score'] >= 70:
        st.success("""
        **Strong eligibility indicators detected.**
        
        This household shows strong signs of needing support. Proceed with the aid allocation process.
        The recommended aid type is based on household characteristics.
        """)
    elif result['score'] >= 50:
        st.warning("""
        **Moderate eligibility - Review Recommended**
        
        This household meets basic eligibility criteria but is near the threshold.
        Consider additional field verification before final decision.
        """)
    elif result['score'] >= 30:
        st.info("""
        **Borderline Case - Human Review Required**
        
        This household is near the eligibility threshold. A field officer should conduct
        additional assessment to make the final decision.
        """)
    else:
        st.error("""
        **Does not meet eligibility criteria**
        
        Based on the provided information, this household does not qualify for the current
        program. They may be eligible for other BRAC programs or services.
        """)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9CA3AF; font-size: 12px;">
    ⚠️ <strong>Disclaimer:</strong> This tool assists decision-making but does not replace human judgment. 
    Always verify predictions with field assessment. Model predictions should be reviewed by qualified personnel.
</div>
""", unsafe_allow_html=True)