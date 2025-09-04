import streamlit as st
from reco_utils import load_data, build_cbf_index, try_load_cf, build_cf_maps, \
                       get_cbf_recs, get_cf_recs, get_hybrid_recs

st.set_page_config(page_title="Hybrid Tourism Recommender", page_icon="🧭", layout="wide")
st.title("🧭 Hybrid Tourism Recommender (CBF + optional CF)")

# Safety: show helpful message if CSVs missing
try:
    places, ratings = load_data()
except Exception as e:
    st.error(f"⚠️ {e}")
    st.markdown("""
**Cara pakai:**
1. Pastikan di root repo ada dua file:
   - `tourism_with_id.csv` (kolom wajib: Place_Id, Place_Name, Category, City, Description, Price)
   - `tourism_rating.csv` (kolom wajib: User_Id, Place_Id, Place_Ratings)
2. Commit & push ke GitHub, lalu redeploy app.
    """)
    st.stop()

@st.cache_resource
def _cbf():
    return build_cbf_index(places)

@st.cache_resource
def _cf():
    model = try_load_cf("artifacts/cf_model.h5")  # optional
    u2e,p2e,_,_ = build_cf_maps(ratings)
    return model,u2e,p2e

places_cbf, vec, X, cos_df = _cbf()
cf_model, u2e, p2e = _cf()

with st.sidebar:
    st.header("Pengaturan")
    user_id = st.number_input("User ID", min_value=int(ratings["User_Id"].min()),
                              max_value=int(ratings["User_Id"].max()),
                              value=int(ratings["User_Id"].min()), step=1)
    topn = st.slider("Top-N rekomendasi", 5, 30, 10)
    alpha = st.slider("Bobot CBF (alpha)", 0.0, 1.0, 0.2, 0.05)
    st.caption("Semakin besar alpha → semakin menekankan CBF. CF butuh file artifacts/cf_model.h5.")

tab1, tab2, tab3 = st.tabs(["Hybrid", "CBF saja", "CF saja (opsional)"])

with tab1:
    if st.button("Hitung rekomendasi (Hybrid)"):
        out = get_hybrid_recs(user_id, places, ratings, cos_df, cf_model, u2e, p2e, topn, alpha)
        st.dataframe(out, use_container_width=True)

with tab2:
    if st.button("Hitung rekomendasi (CBF)"):
        out = get_cbf_recs(user_id, places, ratings, cos_df, topn)
        st.dataframe(out, use_container_width=True)

with tab3:
    st.caption("Butuh file artifacts/cf_model.h5. Kalau tidak ada, tab ini kosong.")
    if st.button("Hitung rekomendasi (CF)"):
        out = get_cf_recs(user_id, places, ratings, cf_model, u2e, p2e, topn)
        if out is not None:
            st.dataframe(out, use_container_width=True)
        else:
            st.info("CF model tidak tersedia atau User ID belum dipetakan. Jalankan tab Hybrid/CBF.")