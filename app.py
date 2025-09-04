import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from reco_utils import (
    load_data, build_cbf_index, try_load_cf, build_cf_maps,
    get_cbf_recs, get_cf_recs, get_hybrid_recs
)

st.set_page_config(page_title='Hybrid Tourism Recommender', page_icon='🧭', layout='wide')
st.title('🧭 Hybrid Tourism Recommender (CBF + optional CF)')

# ===================== Load data =====================
try:
    places, ratings = load_data()
except Exception as e:
    st.error(f'⚠️ {e}')
    st.markdown('''
**Cara pakai:**
1) Taruh 2 file di root repo:
   - `tourism_with_id.csv`: kolom wajib `Place_Id, Place_Name, Category, City, Description, Price`
   - `tourism_rating.csv`: kolom wajib `User_Id, Place_Id, Place_Ratings`
2) Commit & deploy ulang.
''')
    st.stop()

@st.cache_resource
def _cbf():
    # returns: places_cbf, vectorizer, X_tfidf, cosine_df
    return build_cbf_index(places)

@st.cache_resource
def _cf():
    model = try_load_cf('artifacts/cf_model.h5')  # opsional
    u2e, p2e, _, _ = build_cf_maps(ratings)
    return model, u2e, p2e

places_cbf, vectorizer, X_tfidf, cos_df = _cbf()
cf_model, u2e, p2e = _cf()

# ===================== Helpers =====================
def _price_to_float(s: pd.Series) -> pd.Series:
    """Parse harga string -> float (buang Rp, titik, koma)."""
    num = (
        s.astype(str)
         .str.replace(r'[^0-9.,]', '', regex=True)
         .str.replace('.', '', regex=False)
         .str.replace(',', '.', regex=False)
    )
    return pd.to_numeric(num, errors='coerce')

def _apply_filters(df: pd.DataFrame, cities, cats, pmin, pmax):
    """Filter city/category + rentang harga. Mask diselaraskan ke urutan baris df."""
    out = df.copy()
    if cities:
        out = out[out.get('City').isin(cities)]
    if cats:
        out = out[out.get('Category').isin(cats)]

    can_price = (pmin is not None) and (pmax is not None) and ('Place_Id' in out.columns)
    if can_price and not out.empty:
        price_map = _price_to_float(places.set_index('Place_Id')['Price'])
        price_for_rows = price_map.reindex(out['Place_Id'])
        mask = (price_for_rows >= pmin) & (price_for_rows <= pmax)
        mask = mask.fillna(False).to_numpy()
        out = out.loc[mask]
    return out

def _add_maps_link(df: pd.DataFrame) -> pd.DataFrame:
    def maps_url(row):
        q = f"{row['Place_Name']} {row['City']}".replace(' ', '+')
        return f"https://www.google.com/maps/search/?api=1&query={q}"
    df = df.copy()
    df['Maps'] = df.apply(maps_url, axis=1)
    return df

# ====== Preference-based recommender (tanpa riwayat user) ======
def get_preference_recs(
    places_df: pd.DataFrame,
    vectorizer, X_tfidf,
    liked_cats=None, liked_cities=None,
    budget_min=None, budget_max=None,
    keywords:str="", top_n:int=10,
    w_kw:float=0.5, w_cat:float=0.25, w_city:float=0.15, w_price:float=0.10
):
    liked_cats = liked_cats or []
    liked_cities = liked_cities or []

    # --- 1) Skor kata kunci (TF-IDF) ---
    query_tokens = []
    if keywords:
        query_tokens.append(keywords)
    if liked_cats:
        query_tokens.append(" ".join(liked_cats) * 2)  # boost kategori
    if liked_cities:
        query_tokens.append(" ".join(liked_cities))
    if not query_tokens:
        # fallback pakai nama+kategori agar tetap ada sinyal
        query_tokens.append("tourism travel wisata " + " ".join(places_df['Category'].head(50).astype(str).tolist()))
    query_text = " ".join(query_tokens)
    q_vec = vectorizer.transform([query_text])
    sim_kw = cosine_similarity(X_tfidf, q_vec).ravel()
    # normalisasi 0..1 aman
    if sim_kw.max() > sim_kw.min():
        sim_kw = (sim_kw - sim_kw.min()) / (sim_kw.max() - sim_kw.min())
    else:
        sim_kw = np.zeros_like(sim_kw)

    # --- 2) Skor kategori (match 1/0) ---
    cat_series = places_df['Category'].astype(str).str.lower()
    if liked_cats:
        liked_set = set([c.lower() for c in liked_cats])
        cat_score = cat_series.apply(lambda x: 1.0 if any(c in x for c in liked_set) else 0.0).to_numpy()
    else:
        cat_score = np.zeros(len(places_df))

    # --- 3) Skor kota (match 1/0) ---
    city_series = places_df['City'].astype(str).str.lower()
    if liked_cities:
        city_set = set([c.lower() for c in liked_cities])
        city_score = city_series.apply(lambda x: 1.0 if x in city_set else 0.0).to_numpy()
    else:
        city_score = np.zeros(len(places_df))

    # --- 4) Skor harga (dalam rentang = 1, di luar = 0) ---
    price_num = _price_to_float(places_df['Price'])
    if budget_min is not None and budget_max is not None:
        price_score = ((price_num >= budget_min) & (price_num <= budget_max)).astype(float).fillna(0.0).to_numpy()
    else:
        price_score = np.zeros(len(places_df))

    # --- Gabung skor (dibobot) ---
    # Normalisasi bobot kalau total > 0
    w = np.array([w_kw, w_cat, w_city, w_price], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.array([1, 0, 0, 0], dtype=float)
    total = w[0]*sim_kw + w[1]*cat_score + w[2]*city_score + w[3]*price_score

    df = places_df[['Place_Id','Place_Name','Category','City']].copy()
    df['Pref_Score'] = total
    df = df.sort_values('Pref_Score', ascending=False).head(top_n)
    return df

# ===================== Session state =====================
if 'shortlist' not in st.session_state:
    st.session_state['shortlist'] = pd.DataFrame(columns=['Place_Id','Place_Name','Category','City'])

# ===================== Sidebar: Filter global =====================
with st.sidebar:
    st.header('Filter preferensi (opsional)')
    cities_all = sorted(places['City'].dropna().unique().tolist())
    cats_all   = sorted(places['Category'].dropna().unique().tolist())

    global_cities = st.multiselect('City', cities_all, default=[])
    global_cats   = st.multiselect('Category', cats_all, default=[])

    price_num_all = _price_to_float(places['Price'])
    if price_num_all.notna().any():
        lo = float(price_num_all.dropna().quantile(0.05))
        hi = float(price_num_all.dropna().quantile(0.95))
        g_pmin, g_pmax = st.slider('Rentang harga (perkiraan)', min_value=0.0, max_value=max(hi, 1.0),
                                   value=(lo, hi), step=1000.0)
    else:
        g_pmin = g_pmax = None
        st.caption('Harga tidak dapat diparse → filter harga dinonaktifkan.')

# ===================== Tabs =====================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    'Preferensi langsung',      # <== baru
    'Hybrid (riwayat user)',
    'CBF saja',
    'CF saja (opsional)',
    'Mirip tempat ini'
])

# ---------- Tab 0: Preferensi langsung ----------
with tab0:
    st.subheader('🎯 Rekomendasi dari preferensi yang kamu isi (tanpa riwayat rating)')

    c1, c2 = st.columns(2)
    with c1:
        pref_cats = st.multiselect('Saya suka kategori', cats_all, default=[])
        pref_cities = st.multiselect('Saya ingin ke kota', cities_all, default=[])
        kw = st.text_area('Kata kunci (opsional)', placeholder='Contoh: alam hijau air terjun instagramable...')
    with c2:
        st.caption('Bobot komponen skor (total akan dinormalisasi):')
        w_kw   = st.slider('Bobot Kata Kunci', 0.0, 1.0, 0.5, 0.05)
        w_cat  = st.slider('Bobot Kategori',    0.0, 1.0, 0.25, 0.05)
        w_city = st.slider('Bobot Kota',        0.0, 1.0, 0.15, 0.05)
        w_prc  = st.slider('Bobot Harga',       0.0, 1.0, 0.10, 0.05)

    topn_pref = st.slider('Top-N rekomendasi', 5, 30, 10, key='topn_pref')

    if st.button('Hitung rekomendasi (Preferensi)'):
        out = get_preference_recs(
            places, vectorizer, X_tfidf,
            liked_cats=pref_cats, liked_cities=pref_cities,
            budget_min=g_pmin, budget_max=g_pmax,
            keywords=kw, top_n=topn_pref,
            w_kw=w_kw, w_cat=w_cat, w_city=w_city, w_price=w_prc
        )
        out = _apply_filters(out, global_cities, global_cats, g_pmin, g_pmax)
        out = _add_maps_link(out)

        if out.empty:
            st.info('Tidak ada hasil dengan preferensi & filter saat ini. Coba longgarkan pilihan kota/kategori/budget.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','Pref_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp, use_container_width=True, hide_index=True,
                column_config={
                    'Pilih': st.column_config.CheckboxColumn('Pilih'),
                    'Maps': st.column_config.LinkColumn('Maps')
                }
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (Preferensi)', disabled=chosen.empty):
                keep = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep]], ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab 1: Hybrid (riwayat user) ----------
with tab1:
    st.subheader('Hybrid (CBF + CF opsional) dari riwayat User_ID')
    uid_series = ratings['User_Id']
    user_choices = uid_series.dropna().unique().tolist()
    user_id = st.selectbox('User ID', user_choices, index=0, key='uid_hybrid')
    alpha = st.slider('Bobot CBF (alpha)', 0.0, 1.0, 0.2, 0.05, key='alpha_hybrid')
    topn_h = st.slider('Top-N rekomendasi', 5, 30, 10, key='topn_h')

    if st.button('Hitung rekomendasi (Hybrid)'):
        out = get_hybrid_recs(user_id, places, ratings, cos_df, cf_model, u2e, p2e, topn_h, alpha)
        out = _apply_filters(out, global_cities, global_cats, g_pmin, g_pmax)
        out = _add_maps_link(out)
        if out.empty:
            st.info('Tidak ada hasil dengan filter saat ini.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','Hybrid_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp, use_container_width=True, hide_index=True,
                column_config={'Pilih': st.column_config.CheckboxColumn('Pilih'),
                               'Maps': st.column_config.LinkColumn('Maps')}
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (Hybrid)', disabled=chosen.empty):
                keep = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep]], ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab 2: CBF saja ----------
with tab2:
    st.subheader('CBF saja (riwayat User_ID)')
    user_id_cbf = st.selectbox('User ID', user_choices, index=0, key='uid_cbf')
    topn_cbf = st.slider('Top-N rekomendasi', 5, 30, 10, key='topn_cbf')

    if st.button('Hitung rekomendasi (CBF)'):
        out = get_cbf_recs(user_id_cbf, places, ratings, cos_df, topn_cbf)
        out = _apply_filters(out, global_cities, global_cats, g_pmin, g_pmax)
        out = _add_maps_link(out)
        if out.empty:
            st.info('Tidak ada hasil dengan filter saat ini.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','CBF_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp, use_container_width=True, hide_index=True,
                column_config={'Pilih': st.column_config.CheckboxColumn('Pilih'),
                               'Maps': st.column_config.LinkColumn('Maps')}
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (CBF)', disabled=chosen.empty):
                keep = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep]], ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab 3: CF saja (opsional) ----------
with tab3:
    st.subheader('CF saja (opsional)')
    st.caption('Butuh artifacts/cf_model.h5 + TensorFlow. Kalau tidak ada, tab ini bisa kosong.')
    user_id_cf = st.selectbox('User ID', user_choices, index=0, key='uid_cf')
    topn_cf = st.slider('Top-N rekomendasi', 5, 30, 10, key='topn_cf')

    if st.button('Hitung rekomendasi (CF)'):
        out = get_cf_recs(user_id_cf, places, ratings, cf_model, u2e, p2e, topn_cf)
        if out is None:
            st.info('CF model tidak tersedia atau User ID belum dipetakan.')
        else:
            out = _apply_filters(out, global_cities, global_cats, g_pmin, g_pmax)
            out = _add_maps_link(out)
            out_disp = out[['Place_Id','Place_Name','Category','City','CF_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp, use_container_width=True, hide_index=True,
                column_config={'Pilih': st.column_config.CheckboxColumn('Pilih'),
                               'Maps': st.column_config.LinkColumn('Maps')}
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (CF)', disabled=chosen.empty):
                keep = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep]], ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab 4: Mirip tempat ini ----------
with tab4:
    st.subheader('Mirip berdasarkan 1 tempat')
    base_name = st.selectbox('Pilih tempat referensi', sorted(places['Place_Name'].unique().tolist()))
    k = st.slider('Top-N kemiripan', 5, 30, 10, key='topk_similar')
    if st.button('Cari yang mirip'):
        if base_name not in cos_df.index:
            st.info('Tempat tidak ada di indeks CBF.')
        else:
            s = cos_df[base_name].sort_values(ascending=False).drop(base_name).head(k)
            out = places[places['Place_Name'].isin(s.index)][['Place_Id','Place_Name','Category','City']].copy()
            out['CBF_Similarity'] = out['Place_Name'].map(s)
            out = _apply_filters(out, global_cities, global_cats, g_pmin, g_pmax)
            out = _add_maps_link(out)
            out_disp = out[['Place_Id','Place_Name','Category','City','CBF_Similarity','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp, use_container_width=True, hide_index=True,
                column_config={'Pilih': st.column_config.CheckboxColumn('Pilih'),
                               'Maps': st.column_config.LinkColumn('Maps')}
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (Mirip)', disabled=chosen.empty):
                keep = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep]], ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ===================== Shortlist =====================
st.markdown('---')
st.subheader('📝 Shortlist saya')
sl = st.session_state['shortlist']
if sl.empty:
    st.info('Belum ada item di Shortlist. Buat rekomendasi lalu centang beberapa baris.')
else:
    st.dataframe(sl, use_container_width=True, hide_index=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = sl.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download CSV', csv, 'shortlist.csv', 'text/csv')
    with col2:
        if st.button('🗑️ Bersihkan Shortlist'):
            st.session_state['shortlist'] = pd.DataFrame(columns=sl.columns)
            st.success('Shortlist dibersihkan.')
    with col3:
        st.caption('Klik kolom “Maps” di tabel rekomendasi untuk buka Google Maps.')
