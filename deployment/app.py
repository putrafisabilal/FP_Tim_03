import streamlit as st

st.set_page_config(page_title="PPEye: Solusi Cerdas PT Mayora untuk Otomatisasi Pemantauan Kepatuhan APD Sesuai Standar GMP & ISO 9001", layout="wide")

import eda
import appProfile
import inference

st.sidebar.title("PPE")
page = st.sidebar.radio("Pilihan menu:)", ("Perkenalan", "EDA", "Prediksi"), index=0)


st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Team Member:**

1. Gregorius Yoseph Radityo 
2. Putra Fisabil Muhammad
3. Rafi Arya Siregar
4. Rifqi Munif Imanullah
""")

if page == "EDA":
    eda.app()
elif page == "Prediksi":
    inference.app()
else:
    appProfile.app()