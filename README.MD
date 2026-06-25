# 📉 Prediksi Customer Churn - Sales and Marketing Dataset

Aplikasi web untuk memprediksi kemungkinan **customer churn** (pelanggan berhenti berlangganan/menggunakan layanan) berdasarkan data perilaku dan transaksi pelanggan. Dibangun menggunakan **Streamlit** dan model **Random Forest Classifier**.

Proyek ini dibuat sebagai pemenuhan Ujian Akhir Semester mata kuliah **Bengkel Koding Data Science**.

**Nama  :** Suluh Yoga Pratama  
**NIM   :** A11.2023.14979  
**Kelas :** DS-01

🔗 **Demo Aplikasi:** [suluhyogapratama-prediksi-churn.streamlit.app](https://suluhyogapratama-prediksi-churn.streamlit.app/)

---

## 📁 Struktur Folder

```
.
├── model/
│   ├── best_model.pkl          # Model terbaik (Random Forest Classifier)
│   ├── scaler.pkl              # StandardScaler untuk normalisasi fitur numerik
│   ├── encoder.pkl             # OrdinalEncoder untuk fitur kategorikal
│   ├── selected_features.json  # 9 fitur hasil feature selection
│   └── cat_cols.json           # 8 kolom kategorikal yang digunakan saat training
├── notebook/
│   └── A11_2023_14979_SuluhYogaPratama_UAS_BengkelKoding.ipynb   # Notebook lengkap proses EDA s.d. eksport model
├── app.py                      # Aplikasi web Streamlit
├── requirements.txt            # Daftar dependensi Python
└── runtime.txt                 # Versi Python yang digunakan saat deployment
```

---

## 🧠 Tentang Proyek

### 9 Fitur Terpilih (Feature Importance)

| No | Fitur | Keterangan |
|----|-------|------------|
| 1  | `satisfaction_score` | Skor kepuasan pelanggan (skala 1–5) |
| 2  | `total_spent` | Total pengeluaran pelanggan |
| 3  | `support_tickets` | Jumlah tiket layanan pelanggan |
| 4  | `device_type` | Jenis perangkat yang digunakan |
| 5  | `country` | Negara asal pelanggan |
| 6  | `payment_method` | Metode pembayaran |
| 7  | `acquisition_channel` | Channel akuisisi pelanggan |
| 8  | `city` | Kota asal pelanggan |
| 9  | `delivery_delay_days` | Rata-rata keterlambatan pengiriman (hari) |

`satisfaction_score` merupakan fitur dengan kontribusi tertinggi terhadap prediksi churn berdasarkan analisis *impurity-based feature importance* pada Random Forest.

---

## 🚀 Tech Stack

- **Python 3.12**
- **Streamlit**: antarmuka aplikasi web
- **Scikit-Learn**: pemodelan machine learning (Random Forest Classifier, OrdinalEncoder, StandardScaler)
- **Pandas & NumPy**: manipulasi dan pemrosesan data
- **Plotly**: visualisasi data interaktif
- **Joblib**: penyimpanan dan pemuatan model (`.pkl`)

---

## 📊 Model

| Artefak | Deskripsi |
|---|---|
| `best_model.pkl` | Random Forest Classifier (model terbaik hasil evaluasi) |
| `scaler.pkl` | StandardScaler, dilatih pada seluruh fitur numerik sebelum feature selection |
| `encoder.pkl` | OrdinalEncoder untuk 8 kolom kategorikal |
| `selected_features.json` | Daftar 9 fitur final yang digunakan model untuk prediksi |
| `cat_cols.json` | Daftar 8 kolom kategorikal yang di-encode |

---

Proyek ini dibuat untuk keperluan akademik (Ujian Akhir Semester) mata kuliah Bengkel Koding.
