# app.py — Landing + Form Preferensi + Kartu Hasil
# Versi ini mengganti "rating" menjadi "price category" (free/low/medium/high/blank)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from reco_utils import load_data, build_cbf_index

st.set_page_config(page_title="Sistem Rekomendasi Wisata", page_icon="🧭", layout="wide")

# ============ STYLE ============
st.markdown("""
<style>
.main .block-container {max-width: 1100px; padding-top: 1.5rem;}
.hero {text-align:center; padding: 3.0rem 0 2.0rem 0;}
.hero h1 {font-size: 3rem; margin: 0 0 .5rem 0;}
.hero p  {font-size: 1.1rem; color: #5a5f69;}
.hero .btn {display:inline-block; padding:.8rem 1.6rem; background:#2f6df6; color:#fff;
            border-radius: .6rem; text-decoration:none; font-weight:600;}
.section-title {font-size: 2rem; font-weight: 800; margin: 1.8rem 0 .8rem 0;}
.card {border:1px solid #EEE; border-radius: 12px; padding: 16px; margin-bottom: 16px; background:#fff;}
.card h3{margin:.1rem 0 .4rem 0;}
.card small{color:#6b7280;}
div[data-baseweb="select"] {min-width: 240px;}
</style>
""", unsafe_allow_html=True)

# ============ DATA ============
try:
    places, ratings = load_data()
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

# TF-IDF index untuk keyword matching
places_cbf, vectorizer, X_tfidf, cos_df = build_cbf_index(places)

# ---- Helper parsing harga & membuat kategori harga ----
def _price_to_float(s: pd.Series) -> pd.Series:
    num = (s.astype(str)
             .str.replace(r"[^0-9.,]", "", regex=True)
             .str.replace(".", "", regex=False)
             .str.replace(",", ".", regex=False))
    return pd.to_numeric(num, errors="coerce")

price_num = _price_to_float(places["Price"])
# Ambil kuantil agar adaptif ke dataset (≈ 33% dan 66%)
q1 = price_num.dropna().quantile(0.33) if price_num.notna().any() else np.nan
q2 = price_num.dropna().quantile(0.66) if price_num.notna().any() else np.nan

def _price_cat(v: float) -> str | None:
    if pd.isna(v):
        return None
    if v == 0:
        return "free"
    if pd.isna(q1) or pd.isna(q2):
        # fallback kalau semua harga NaN → tanpa kategori
        return None
    if v <= q1:
        return "low"
    if v <= q2:
        return "medium"
    return "high"

places["Price_Num"] = price_num
places["Price_Cat"] = price_num.apply(_price_cat)

# Pilihan unik
CITIES = ["Semua"] + sorted(places["City"].dropna().unique().tolist())
CATS   = ["Semua"] + sorted(places["Category"].dropna().unique().tolist())
PRICE_CATS = ["", "free", "low", "medium", "high"]  # "" = blank (abaikan)

# ============ HELPER REKOMENDASI ============
def preference_score(df: pd.DataFrame, keywords: str, cat: str, city: str) -> pd.Series:
    """Skor gabungan: keyword (TF-IDF) + match kategori + match kota."""
    # 1) keyword via TF-IDF
    q = keywords.strip()
    if q:
        q_vec = vectorizer.transform([q])
        sim_kw = cosine_similarity(X_tfidf, q_vec).ravel()
        sim_kw = (sim_kw - sim_kw.min()) / (sim_kw.max() - sim_kw.min()) if sim_kw.max() > sim_kw.min() else np.zeros_like(sim_kw)
    else:
        sim_kw = np.zeros(len(df))

    # 2) kategori (1/0)
    if cat and cat != "Semua":
        cat_score = (df["Category"].astype(str).str.lower() == cat.lower()).astype(float).to_numpy()
    else:
        cat_score = np.zeros(len(df))

    # 3) kota (1/0)
    if city and city != "Semua":
        city_score = (df["City"].astype(str).str.lower() == city.lower()).astype(float).to_numpy()
    else:
        city_score = np.zeros(len(df))

    # bobot sederhana
    w_kw, w_cat, w_city = 0.6, 0.25, 0.15
    return (w_kw*sim_kw + w_cat*cat_score + w_city*city_score)

def make_cards(df: pd.DataFrame):
    """Render grid 2 kolom berisi kartu tempat."""
    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        price_text = row.get("Price", "")
        price_cat  = row.get("Price_Cat", None)
        price_line = f"Harga: {price_cat}" if pd.notna(price_cat) and price_cat else f"Harga: {price_text}"
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="card">
                  <h3>{row['Place_Name']}</h3>
                  <small>Kategori: {row['Category']}</small><br>
                  <small>Lokasi: {row['City']}</small><br>
                  <small>{price_line}</small><br>
                  <small><a href="https://www.google.com/maps/search/?api=1&query={row['Place_Name']} {row['City']}"
                            target="_blank">Lihat di Google Maps</a></small>
                </div>
                """,
                unsafe_allow_html=True
            )

# ============ HERO ============
st.markdown(
    """
<div class="hero">
  <h1>Sistem Rekomendasi Wisata</h1>
  <p>Temukan destinasi wisata sesuai preferensi Anda.</p>
  <a class="btn" href="#form">Mulai</a>
</div>
""",
    unsafe_allow_html=True
)

# ============ FORM PREFERENSI ============
st.markdown('<div id="form"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Masukkan Preferensi</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    pilih_kategori = st.selectbox("Kategori", CATS, index=0)
    price_cat_input = st.selectbox(
        "Enter the price category (free/low/medium/high or blank)",
        PRICE_CATS, index=0, help="Pilih kosong (blank) untuk mengabaikan harga"
    )
with c2:
    pilih_lokasi = st.selectbox("Lokasi", CITIES, index=0)
    keywords = st.text_input("Kata Kunci (opsional)", placeholder="contoh: alam hijau air terjun instagramable")

topn = st.slider("Jumlah rekomendasi", min_value=4, max_value=24, value=10, step=2)
go = st.button("Dapatkan Rekomendasi", type="primary")

# ============ HASIL ============
st.markdown('<div class="section-title">Hasil Rekomendasi</div>', unsafe_allow_html=True)

if go:
    df = places.copy()

    # Filter kategori & lokasi (opsional)
    if pilih_kategori != "Semua":
        df = df[df["Category"] == pilih_kategori]
    if pilih_lokasi != "Semua":
        df = df[df["City"] == pilih_lokasi]

    # Filter kategori harga (free/low/medium/high) jika diisi
    if price_cat_input:
        df = df[df["Price_Cat"].fillna("") == price_cat_input]

    # Hitung skor preferensi (keyword + match cat & city)
    df["Pref_Score"] = preference_score(df, keywords, pilih_kategori, pilih_lokasi)

    # fallback kalau kosong → longgarkan harga dulu, lalu kota/kategori
    if df.empty and price_cat_input:
        df = places.copy()
        if pilih_kategori != "Semua":
            df = df[df["Category"] == pilih_kategori]
        if pilih_lokasi != "Semua":
            df = df[df["City"] == pilih_lokasi]
        df["Pref_Score"] = preference_score(df, keywords, pilih_kategori, pilih_lokasi)

    if df.empty:
        df = places.copy()
        df["Pref_Score"] = preference_score(df, keywords, "Semua", "Semua")

    out = df.sort_values("Pref_Score", ascending=False).head(topn)

    if out.empty:
        st.info("Tidak ada hasil dengan preferensi saat ini. Coba kosongkan kategori harga atau longgarkan pilihan.")
    else:
        make_cards(out)
        csv = out[["Place_Id","Place_Name","Category","City","Price","Price_Cat","Pref_Score"]].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download hasil (.csv)", csv, "rekomendasi.csv", "text/csv")
else:
    st.info("Isi preferensi di atas lalu klik **Dapatkan Rekomendasi**.")
