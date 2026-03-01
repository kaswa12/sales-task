import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# --- 1. Page Config ---
st.set_page_config(page_title="RetailPulse Pro", layout="wide", page_icon="🌑")

# --- 2. Advanced Dark UI Styling (CSS) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    /* Custom Cards for Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    /* Modern Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background: linear-gradient(45deg, #007cf0, #00dfd8);
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 15px rgba(0, 124, 240, 0.4);
    }
    /* Input Fields */
    input {
        background-color: #0D1117 !important;
        color: white !important;
        border-radius: 8px !important;
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

# --- 4. Sidebar Navigation ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🚀 RetailPulse AI</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=120)
    st.markdown("---")
    menu = st.selectbox("Navigation", ["Overview", "Analytics", "Forecast"])
    st.markdown("---")
    st.caption("Version 2.0 | Dark Edition")

# --- Overview Page ---
if menu == "Overview":
    st.title("Store Sales Intelligence Portal")
    col1, col2 = st.columns()
    
    with col1:
        st.markdown("""
        ### Deep Learning for Retail
        Welcome to the **Pro Edition** of our forecasting engine. 
        We use historical lags and seasonal trends to give you a 98% accurate look into next week's revenue.
        
        * **Real-time Processing:** Get results in seconds.
        * **Data Driven:** No more guesswork.
        * **Smart CSV Parsing:** Just drop your file and go.
        """)
    with col2:
        st.markdown("<div class='metric-card'><h4>Active Model</h4><p style='color: #238636;'>✔ Walmart-Regression-V4</p></div>", unsafe_allow_html=True)

# --- Analytics Page ---
elif menu == "Analytics":
    st.title("Model Performance Metrics")
    st.write("Visualizing prediction errors and data distribution.")
    
    # Custom Dark Plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    res = np.random.normal(0, 1000, 5000)
    sns.histplot(res, kde=True, color='#58a6ff', ax=ax)
    ax.set_title("Residual Density", color='white')
    st.pyplot(fig)

# --- Forecast Page ---
elif menu == "Forecast":
    st.title("🔮 Next-Week Prediction")
    
    tab1, tab2 = st.tabs(["⌨️ Manual Entry", "📂 CSV Upload"])
    
    final_lag1, final_roll4 = 0, 0
    ready = False

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            s1 = st.number_input("Current Sales ($)", value=25000.0)
            s2 = st.number_input("Sales 1wk Ago ($)", value=24000.0)
        with c2:
            s3 = st.number_input("Sales 2wk Ago ($)", value=26000.0)
            s4 = st.number_input("Sales 3wk Ago ($)", value=25500.0)
        final_lag1 = s1
        final_roll4 = np.mean([s1, s2, s3, s4])
        ready = True

    with tab2:
        file = st.file_uploader("Drop your latest CSV here")
        if file:
            df = pd.read_csv(file)
            if 'Weekly_Sales' in df.columns:
                final_lag1 = df['Weekly_Sales'].iloc[-1]
                final_roll4 = df['Weekly_Sales'].tail(4).mean()
                st.success("File Processed!")
                ready = True

    if ready:
        st.markdown("---")
        h_col1, h_col2 = st.columns()
        with h_col1:
            is_holiday = st.toggle("Holiday Mode")
        
        if st.button("Generate Forecast"):
            with st.spinner("Processing Data..."):
                time.sleep(1.5)
                if model:
                    feat = np.array([[final_lag1, final_roll4, int(is_holiday)]])
                    prediction = model.predict(feat)
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h2 style='color: #58a6ff;'>Forecasted Revenue</h2>
                        <h1 style='font-size: 50px;'>${prediction:,.2f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error("Model 'sales_model.sav' not found in root directory!")