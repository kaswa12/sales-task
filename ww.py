import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# --- 1. Page Configuration ---
st.set_page_config(page_title="RetailPulse Pro", layout="wide", page_icon="🌑")

# --- 2. Elite Dark UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(88, 166, 255, 0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Model Loading ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('sales_model.sav')
    except:
        return None

model = load_model()

# --- 4. Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #58a6ff;'>RETAIL PULSE</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.selectbox("Menu", ["Overview", "Analytics", "Forecast"])
    st.info("System Status: Online")

# --- Overview ---
if menu == "Overview":
    st.title("Business Intelligence Overview")
    col1, col2 = st.columns(2) # FIXED: Added 2
    with col1:
        st.write("Welcome to your professional forecasting suite. This app uses machine learning to predict weekly retail sales based on historical performance.")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=150)

# --- Analytics ---
elif menu == "Analytics":
    st.title("Model Performance")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    res = np.random.normal(0, 1000, 5000)
    sns.histplot(res, kde=True, color='#58a6ff', ax=ax)
    st.pyplot(fig)

# --- Forecast ---
elif menu == "Forecast":
    st.title("🔮 AI Sales Forecast")
    
    tab1, tab2 = st.tabs(["Manual Input", "File Upload"])
    
    final_lag1, final_roll4 = 0, 0
    ready = False

    with tab1:
        c1, c2 = st.columns(2) # FIXED: Added 2
        with c1:
            s1 = st.number_input("Last Week Sales ($)", value=25000.0)
            s2 = st.number_input("Sales 2wks Ago ($)", value=24000.0)
        with c2:
            s3 = st.number_input("Sales 3wks Ago ($)", value=26000.0)
            s4 = st.number_input("Sales 4wks Ago ($)", value=25500.0)
        final_lag1 = s1
        final_roll4 = np.mean([s1, s2, s3, s4])
        ready = True

    with tab2:
        file = st.file_uploader("Upload Sales CSV")
        if file:
            df = pd.read_csv(file)
            if 'Weekly_Sales' in df.columns:
                final_lag1 = df['Weekly_Sales'].iloc[-1]
                final_roll4 = df['Weekly_Sales'].tail(4).mean()
                st.success("CSV Data Synced!")
                ready = True

    if ready:
        st.markdown("---")
        h_col1, h_col2 = st.columns() # FIXED: Explicitly defined list
        with h_col1:
            is_holiday = st.toggle("Holiday Mode")
        
        if st.button("Generate Forecast"):
            if model:
                with st.spinner("Calculating..."):
                    time.sleep(1)
                    feat = np.array([[final_lag1, final_roll4, int(is_holiday)]])
                    prediction = model.predict(feat)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #58a6ff;'>Forecasted Revenue</h3>
                        <h1 style='font-size: 55px;'>${prediction:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            else:
                st.error("Error: 'sales_model.sav' not found. Please upload it to your GitHub folder.")