import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import google.generativeai as genai
from datetime import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# MongoDB connection string
mongo_uri = "mongodb+srv://IotHatch:hatchiot123@iothatch.5kday.mongodb.net/?appName=IotHatch"

# --- Load Data from MongoDB ---
@st.cache_data
def load_data_from_mongo():
    # Setup MongoDB connection
    client = MongoClient(mongo_uri)
    db = client["IotHatch"]
    collection = db["aquasync"]

    # Retrieve data from MongoDB
    data = list(collection.find({}, {"_id": 0}))  # Exclude the _id field
    df = pd.DataFrame(data)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

# --- Setup ---
st.set_page_config(page_title="AquaSync Dashboard", layout="wide")

# Load data from MongoDB
df = load_data_from_mongo()

if df.empty or "timestamp" not in df.columns:
    st.warning("Tidak ada data untuk ditampilkan.")
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

# --- Filter Tanggal ---
st.sidebar.header("📅 Filter Waktu")
min_date = df["date"].min() if not df.empty else datetime.today().date()
max_date = df["date"].max() if not df.empty else datetime.today().date()

start_date = st.sidebar.date_input("Tanggal mulai", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("Tanggal akhir", min_value=min_date, max_value=max_date, value=max_date)

filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

st.title("💧 AquaSync Dashboard")

# --- Insight Otomatis ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("💡 Insight Otomatis dari Data")

if not filtered_df.empty:
    max_day = filtered_df.groupby("date")["debit"].sum().idxmax()
    max_val = filtered_df.groupby("date")["debit"].sum().max()
    trend = "naik" if filtered_df["debit"].iloc[-1] > filtered_df["debit"].iloc[0] else "turun"
    
    insight = f"""
    - Hari dengan penggunaan tertinggi: *{max_day}* sebesar *{max_val:.2f} liter*
    - Tren penggunaan air selama periode ini: *{trend}*
    """
    st.markdown(insight)
else:
    st.info("Belum cukup data untuk insight.")

# --- Ringkasan ---
if not filtered_df.empty:
    total_liter = filtered_df["debit"].sum()
    avg_liter_per_use = filtered_df["debit"].mean()

    col1, col2 = st.columns(2)
    col1.metric("🔢 Total Air Digunakan", f"{total_liter:.2f} Liter")
    col2.metric("📏 Rata-rata per Penggunaan", f"{avg_liter_per_use:.2f} Liter")
else:
    st.warning("Tidak ada data untuk rentang waktu yang dipilih.")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tabs untuk berbagai analisis ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
    "📊 Total Penggunaan Air",
    "📈 Rata-rata per Hari",
    "📌 Proporsi Penggunaan",
    "🚨 Anomali",
    "🔮 Prediksi"
    ])

# --- Tab 1: Total Penggunaan Air per Hari ---
with tab1:
    st.subheader("📊 Total Penggunaan Air per Hari")

    if 'filtered_df' in locals() and not filtered_df.empty:
        total_per_day = filtered_df.groupby("date")["debit"].sum().reset_index()

        fig = px.bar(
            total_per_day,
            x="date",
            y="debit",
            labels={"date": "Tanggal", "debit": "Total Liter"},
            title="Total Penggunaan Air per Hari",
            color_discrete_sequence=["#1f77b4"],
            hover_data={"debit": True, "date": True}
        )

        fig.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="white",
            title_font_size=18,
            xaxis_tickangle=-45,
        )

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.2)')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada data untuk ditampilkan.")
            
# --- Tab 2: Rata-rata Volume Air per Hari ---
with tab2:
    st.subheader("📈 Rata-rata Volume Air per Hari")
    if not filtered_df.empty:
        avg_per_day = filtered_df.groupby("date")["debit"].mean()
        avg_df = avg_per_day.reset_index()
        st.line_chart(avg_df, x="date", y="debit", use_container_width=True)
    else:
        st.info("Tidak ada data rata-rata.")

# --- Tab 3: Proporsi Penggunaan Air per Hari ---
with tab3:
    st.subheader("📌 Proporsi Penggunaan Air per Hari")
    if not filtered_df.empty:
        pie_data = filtered_df.groupby("date")["debit"].sum().reset_index()
        fig = px.pie(
            pie_data, values="debit", names="date",
            title="Persentase Penggunaan Air", hole=0.4,
        )
        st.plotly_chart(fig)
    else:
        st.info("Belum ada data proporsi.")

# --- Tab 4: Anomali Penggunaan Air ---
with tab4:
    st.subheader("🚨 Anomali Penggunaan Air")
    if not filtered_df.empty:
        anomalies = filtered_df[filtered_df["debit"] > 1.0]
        if not anomalies.empty:
            st.error("Terdeteksi anomali penggunaan air lebih dari 1 liter:")
            st.dataframe(anomalies[["timestamp", "debit"]])
        else:
            st.success("Tidak ada anomali terdeteksi.")
    else:
        st.info("Belum ada data untuk pengecekan anomali.")

# --- Tab 5: Prediksi Penggunaan Air (Linear Regression) ---
with tab5:
    st.subheader("📈 Prediksi Penggunaan Air (Linear Regression)")

    if len(filtered_df) >= 3:
        pred_df = filtered_df.groupby("date")["debit"].sum().reset_index()
        pred_df["date_ordinal"] = pd.to_datetime(pred_df["date"]).map(datetime.toordinal)

        X = pred_df["date_ordinal"].values.reshape(-1, 1)
        y = pred_df["debit"].values

        model = LinearRegression()
        model.fit(X, y)

        last_date = pred_df["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        predictions = model.predict(future_ordinals)

        future_df = pd.DataFrame({
            "date": future_dates,
            "predicted_liter": predictions
        })

        st.write("Prediksi penggunaan air 3 hari ke depan:")
        st.dataframe(future_df)

        combined = pd.concat([
            pred_df[["date", "debit"]].rename(columns={"debit": "value"}).assign(type="Aktual"),
            future_df.rename(columns={"predicted_liter": "value"}).assign(type="Prediksi")
        ])

        fig = px.line(combined, x="date", y="value", color="type", markers=True,
                      title="Aktual vs Prediksi Penggunaan Air")
        st.plotly_chart(fig)
    else:
        st.info("Data kurang untuk membuat prediksi.")

# --- Chatbot Gemini ---

# Konfigurasi Gemini
genai.configure(api_key="AIzaSyAKkHEPR8_UhFwOa41bITZMzSKXnbSKIFg")
model = genai.GenerativeModel("models/gemini-2.0-flash")

# --- AI Assistant di Sidebar  ---
with st.sidebar:
    st.markdown("## 🤖 AI AquaSync Assistant")

    # Inisialisasi state untuk tampung jawaban terakhir
    if "last_ai_answer" not in st.session_state:
        st.session_state.last_ai_answer = ""

    # Form untuk tanya AI (Enter langsung submit)
    with st.form("ai_form"):
        user_prompt = st.text_input("Tanya sesuatu tentang data air (tekan Enter):")
        submit = st.form_submit_button("Tanya AI")

        if submit and user_prompt.strip():
            with st.spinner("AI sedang berpikir..."):
                try:
                    response = model.generate_content(f"""
                    Kamu adalah AI di sistem IoT bernama AquaSync.
                    Berikut data penggunaan air per hari: {filtered_df.to_dict(orient='records')}
                    Jawab pertanyaan ini: {user_prompt}
                    Berikan jawaban yang akurat dan saran jika diperlukan.
                    """)
                    st.session_state.last_ai_answer = response.text
                except Exception as e:
                    st.session_state.last_ai_answer = f"❌ Error: {e}"

    # Tampilkan jawaban jika ada
    if st.session_state.last_ai_answer:
        st.markdown("---")
        st.markdown("💡 Jawaban AI:")
        st.write(st.session_state.last_ai_answer)
