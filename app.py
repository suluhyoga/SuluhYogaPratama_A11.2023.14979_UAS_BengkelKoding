import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Analisis & Prediksi Churn Pelanggan",
    page_icon="📉",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #1A2036;
        color: #E0E6ED;
    }

    h1 {
        color: #3498DB !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .stMarkdown p, .stMarkdown li {
        color: #B2BEC3;
    }
    
    h3, h4, h5 {
        color: #E0E6ED !important;
    }

    .custom-guide-box {
        background-color: #151929;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #3498DB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }
    .custom-guide-title {
        color: #3498DB;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151929;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #B2BEC3;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3498DB20;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3498DB;
        color: #FFFFFF !important;
        font-weight: bold;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 1.1em;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.5);
    }

    .status-banner {
        padding: 12px 15px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 20px;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .status-safe {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ECC71;
        border: 1px solid #2ECC71;
    }
    .status-churn {
        background-color: rgba(231, 76, 60, 0.15);
        color: #E74C3C;
        border: 1px solid #E74C3C;
    }

    .stExpander {
        background-color: #151929;
        border: 1px solid #2C3E50;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model_dir = 'model'
    model   = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    scaler  = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(model_dir, 'encoder.pkl'))

    feat_path_json = os.path.join(model_dir, 'selected_features.json')
    feat_path_pkl  = os.path.join(model_dir, 'selected_features.pkl')
    if os.path.exists(feat_path_json):
        with open(feat_path_json) as f:
            features = json.load(f)
    else:
        features = joblib.load(feat_path_pkl)

    with open(os.path.join(model_dir, 'cat_cols.json')) as f:
        cat_cols = json.load(f)

    return model, scaler, encoder, features, cat_cols

model, scaler, encoder, selected_features, cat_cols = load_artifacts()

st.markdown("<h1 style='text-align: center;'>Analisis & Prediksi Churn Pelanggan</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.05em; color: #B2BEC3; max-width: 600px; margin: 0 auto;'>Aplikasi ini digunakan untuk mendeteksi apakah pelanggan Anda berpotensi pergi (berhenti menggunakan layanan) atau tetap setia. Sistem otomatis menganalisis data profil dan aktivitas harian mereka agar bisa mengambil langkah pencegahan dengan cepat.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="custom-guide-box">
    <div class="custom-guide-title">
        <span>📖 Panduan Fitur</span>
    </div>
    <ul style='list-style-type: none; padding-left: 0; margin-bottom: 0; font-size: 0.95em; color: #B2BEC3;'>
        <li style='margin-bottom: 8px;'>🔹 <b style="color: #E0E6ED;">Kepuasan :</b> Skor tingkat kepuasan pelanggan terhadap layanan keseluruhan (skala 1–5).</li>
        <li style='margin-bottom: 8px;'>🔹 <b style="color: #E0E6ED;">Total Pengeluaran :</b> Akumulasi total uang yang sudah dibelanjakan pelanggan.</li>
        <li style='margin-bottom: 8px;'>🔹 <b style="color: #E0E6ED;">Tiket Support :</b> Akumulasi pengajuan keluhan atau tiket bantuan layanan pelanggan.</li>
        <li style='margin-bottom: 8px;'>🔹 <b style="color: #E0E6ED;">Keterlambatan Pengiriman :</b> Jumlah hari rata-rata keterlambatan pengiriman pesanan pelanggan.</li>
        <li style='margin-bottom: 0px;'>🔹 <b style="color: #E0E6ED;">Lainnya :</b> Tipe perangkat, negara, kota, metode pembayaran, dan channel akuisisi pelanggan.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.subheader("📝 Formulir Data Pelanggan")

col1, col2 = st.columns(2)
with col1:
    satisfaction_score  = st.slider("Skor Kepuasan (1–5)", 1.0, 5.0, 3.0, step=0.5)
    total_spent          = st.number_input("Total Pengeluaran ($)", 0.0, 5000.0, 500.0, step=10.0)
    support_tickets       = st.number_input("Jumlah Tiket Support", 0, 50, 2)
    device_type           = st.selectbox("Tipe Perangkat", ["Mobile", "Desktop", "Tablet"])
    country                = st.selectbox("Negara", ["India", "Germany", "USA", "UK", "Bangladesh"])
with col2:
    payment_method        = st.selectbox("Metode Pembayaran", ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"])
    acquisition_channel   = st.selectbox("Channel Akuisisi", ["Email", "Social Media", "Referral", "Paid Ads", "Organic"])
    city                    = st.selectbox("Kota", ["Berlin", "Mumbai", "London", "Hamburg", "New York", "Dhaka", "Chicago"])
    delivery_delay         = st.number_input("Keterlambatan Pengiriman (hari)", 0, 30, 2)

st.markdown("<br>", unsafe_allow_html=True)

st.button("Jalankan Analisis Prediksi", key="predict_btn", type="primary", use_container_width=True)

if st.session_state.get("predict_btn"):
    # Hanya 9 fitur dari form (sesuai selected_features.json) yang diisi user.
    form_values = {
        "satisfaction_score" : satisfaction_score,
        "total_spent"        : total_spent,
        "support_tickets"    : support_tickets,
        "device_type"        : device_type,
        "country"            : country,
        "payment_method"     : payment_method,
        "acquisition_channel": acquisition_channel,
        "city"               : city,
        "delivery_delay_days": delivery_delay,
    }
    input_data = pd.DataFrame([form_values])

    # Kolom lain yang dibutuhkan scaler (hasil training) tapi tidak ditampilkan
    # di form karena bukan bagian dari fitur terpilih -> diisi otomatis.
    # - Kolom numerik diisi 0.
    # - Kolom kategorikal HARUS diisi salah satu kategori valid (bukan 0),
    #   agar encoder.transform() tidak menghasilkan NaN saat di-scale.
    scaler_cols = list(scaler.feature_names_in_)
    for i, col in enumerate(scaler_cols):
        if col not in input_data.columns:
            if col in cat_cols:
                cat_idx = cat_cols.index(col)
                input_data[col] = encoder.categories_[cat_idx][0]
            else:
                input_data[col] = 0

    cols_to_encode = [c for c in cat_cols if c in input_data.columns]
    if cols_to_encode:
        input_data[cols_to_encode] = encoder.transform(input_data[cols_to_encode])

    input_data = input_data[scaler_cols]
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    input_scaled_all = scaler.transform(input_data)
    input_scaled_df  = pd.DataFrame(input_scaled_all, columns=scaler_cols)
    input_df = input_scaled_df[selected_features]

    pred       = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    prob_churn = pred_proba[1] * 100

    st.divider()

    st.markdown("<h3 style='text-align: center;'>📊 Profil Risiko Pelanggan</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class="status-banner status-churn">
            🚨 STATUS: CHURN
        </div>
        <div style='color: #E74C3C; font-weight: bold; font-size: 2.2em; text-align: center; margin-bottom: 8px;'>
            {prob_churn:.1f}%
        </div>
        <p style='color: #B2BEC3; text-align: center; margin-bottom: 25px;'>Pelanggan ini terindikasi memiliki risiko tinggi untuk berhenti berlangganan. Diperlukan tindakan penanganan retensi prioritas.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-banner status-safe">
            ✅ STATUS: TIDAK CHURN
        </div>
        <div style='color: #2ECC71; font-weight: bold; font-size: 2.2em; text-align: center; margin-bottom: 8px;'>
        {pred_proba[0]*100:.1f}%
        </div>
        <p style='color: #B2BEC3; text-align: center; margin-bottom: 25px;'>Kondisi loyalitas pelanggan stabil. Pertahankan performa layanan dan program engagement berkala.</p>
        """, unsafe_allow_html=True)

    v_col1, v_col2 = st.columns([1, 1])

    fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob_churn,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Metrik Risiko Churn", 'font': {'color': '#64748B', 'size': 18}},
    number={'suffix': "%", 'font': {'color': '#E0E6ED', 'size': 48}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#34495E"},
        'bar': {'color': "#3498DB", 'thickness': 0.35},
        'bgcolor': "#151929",
        'borderwidth': 1,
        'bordercolor': "#2C3E50",
        'steps': [
            {'range': [0, 40], 'color': "rgba(46, 204, 113, 0.05)"},
            {'range': [40, 70], 'color': "rgba(241, 196, 15, 0.05)"},
            {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.05)"}
        ],
    }
))

    fig_gauge.update_layout(
        paper_bgcolor='#1A2036',
        plot_bgcolor='#1A2036',
        height=400,  # dari 270 jadi 400
        margin=dict(l=50, r=50, t=60, b=40)
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    st.markdown("<h4>Rincian Probabilitas</h4>", unsafe_allow_html=True)
    met_col1, met_col2 = st.columns(2)
    
    with met_col1:
        st.markdown(f"""
        <div style="margin-bottom: 6px;">
            <span style="color: #64748B; font-size: 0.9em; font-weight: 500;">Tidak Churn</span>
            <div style="color: #2ECC71; font-size: 1.8em; font-weight: bold; margin-top: -2px;">{pred_proba[0]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(pred_proba[0]))
        
    with met_col2:
        st.markdown(f"""
        <div style="margin-bottom: 6px;">
            <span style="color: #64748B; font-size: 0.9em; font-weight: 500;">Churn</span>
            <div style="color: #E74C3C; font-size: 1.8em; font-weight: bold; margin-top: -2px;">{pred_proba[1]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(pred_proba[1]))

    st.markdown("<br>", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.85em;'>Suluh Yoga Pratama | A11.2023.14979 | UAS Bengkel Koding</p>", unsafe_allow_html=True)