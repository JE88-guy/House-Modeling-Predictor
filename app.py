import streamlit as st
import joblib
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Housing Estate AI", page_icon="🏘️", layout="wide")

# 2. Robust Model Loading
path = 'Models/Population_Model.pkl'
if os.path.exists(path):
    model = joblib.load(path)
    try:
        # This allows the app to 'learn' what the model needs automatically
        model_features = model.feature_names_in_.tolist()
    except AttributeError:
        st.error("Model format error. Please ensure the model was trained with feature names.")
        st.stop()
else:
    st.error(f"Model not found at {path}. Check your GitHub 'Models' folder!")
    st.stop()

# 3. UI Header
st.title("🏘️ Housing Estate Development Strategy")
st.markdown("""
    **Project Phase 6: Operational Deployment** This tool uses a Decision Tree model to forecast population shifts and recommend the optimal 
    housing development tier for specific Philippine regions.
""")
st.divider()

# 4. Input Sidebar
with st.sidebar:
    st.header("📍 Location & Time")
    target_year = st.selectbox("Forecast Year", [2025, 2026, 2027])
    
    # Dynamically extract region names from the model's own memory
    available_regions = [f.replace('Region_', '') for f in model_features if f.startswith('Region_')]
    selected_region = st.selectbox("Select Target Region", sorted(available_regions))

# 5. Market Data Inputs
st.subheader("📊 Market Indicators")
col1, col2 = st.columns(2)

with col1:
    pop_prev = st.number_input("Current Population (Latest Census)", value=1200000, step=1000)
    growth_prev = st.slider("Current Annual Growth Rate (%)", 0.0, 5.0, 1.5)

with col2:
    pop_prev2 = st.number_input("Population (2 Years Prior)", value=1150000, step=1000)
    growth_prev2 = st.slider("Previous Annual Growth Rate (%)", 0.0, 5.0, 1.4)

# 6. Analysis Logic
if st.button("Generate Development Strategy", key="deploy_btn"):
    # Everything below this must be indented by 4 spaces
    input_dict = {col: [0.0] for col in model_features}
    
    # Map Numerical Data (Forcing Float conversion)
    if 'Year' in input_dict: input_dict['Year'] = [float(target_year)]
    if 'Prev_Population' in input_dict: input_dict['Prev_Population'] = [float(pop_prev)]
    if 'Prev2_Population' in input_dict: input_dict['Prev2_Population'] = [float(pop_prev2)]
    if 'Prev_Growth' in input_dict: input_dict['Prev_Growth'] = [float(growth_prev)]
    if 'Prev2_Growth' in input_dict: input_dict['Prev2_Growth'] = [float(growth_prev2)]
    
    # Recalculate Rolling Growth
    if 'Rolling Growth' in input_dict:
        input_dict['Rolling Growth'] = [(float(growth_prev) + float(growth_prev2)) / 2.0]
    
    # Map Region
    region_col = f"Region_{selected_region}"
    if region_col in input_dict:
        input_dict[region_col] = [1.0]
    
    # Final Dataframe
    feature_df = pd.DataFrame(input_dict)[model_features]
    
    # --- THIS WAS THE ERROR AREA ---
    try:
        # Treat model output as 'Projected Increase'
        projected_increase = model.predict(feature_df)[0]
        future_total = pop_prev + projected_increase
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Predicted New Residents", 
                      value=f"{int(projected_increase):,}",
                      delta=f"{((projected_increase/pop_prev)*100):.2f}% Growth")
            
            st.metric(label=f"Total {target_year} Population", 
                      value=f"{int(future_total):,}")
        
        with res_col2:
            if projected_increase > 400000:
                st.success("### 🏙️ TIER 1: MASSIVE DEMAND")
                st.write("**Strategy:** High-rise Vertical Development")
            elif projected_increase > 150000:
                st.success("### 🏡 TIER 2: SUSTAINED GROWTH")
                st.write("**Strategy:** Suburban Gated Communities")
            else:
                st.warning("### 🏗️ TIER 3: SLOW/STABLE MARKET")
                st.write("**Strategy:** Niche or Socialized Housing")
                
    except Exception as e:
        st.error(f"Operational Error: {e}")
# Footer for your project submission
st.divider()
st.caption("Final Model Deployment - CRISP-DM Phase 6 - Housing Estate Analysis Framework")
