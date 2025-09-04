import streamlit as st
import pandas as pd
from reco_utils import (
    load_data, build_cbf_index, try_load_cf, build_cf_maps,
    get_cbf_recs, get_cf_recs, get_hybrid_recs
)

st.set_page_config(page_title='Hybrid Tourism Recommender', page_icon='🧭', layout='wide')
st.title('🧭 Hybrid Tourism Recommender (CBF + optional CF)')

# ---------- Load data (aman & ada pesan bantuan) ----------
try:
    places, ratings = load_data()
except Exception as e:
    st.error(f'⚠️ {e}')
    st.markdown('''
**Cara pakai:**
1. Pastikan di root repo ada dua file:
   - `tourism_with_id.csv` (kolom wajib: Place_Id, Place_Name, Category, City, Description, Price)
   - `tourism_rating.csv` (kolom wajib: User_Id, Place_Id, Place_Ratings)
2. Commit & push ke GitHub, lalu redeploy app.
    ''')
    st.stop()

@st.cache_resource
def _cbf():
    return build_cbf_index(places)

@st.cache_resource
def _cf():
    model = try_load_cf('artifacts/cf_model.h5')  # opsional; butuh TF jika benar2 dipakai
    u2e, p2e, _, _ = build_cf_maps(ratings)
    return model, u2e, p2e

places_cbf, vec, X, cos_df = _cbf()
cf_model, u2e, p2e = _cf()

# ---------- Helpers ----------
def _price_to_float(s: pd.Series) -> pd.Series:
    """Parse harga: ambil angka dari string (buang Rp, titik, koma, dll)."""
    num = (
        s.astype(str)
         .str.replace(r'[^0-9.,]', '', regex=True)
         .str.replace('.', '', regex=False)      # buang pemisah ribuan
         .str.replace(',', '.', regex=False)     # koma -> titik desimal
    )
    return pd.to_numeric(num, errors='coerce')

def _apply_filters(df: pd.DataFrame, cities, cats, pmin, pmax):
    """Filter city/category + rentang harga (mask diselaraskan ke urutan baris df)."""
    out = df.copy()

    if cities:
        out = out[out.get('City').isin(cities)]
    if cats:
        out = out[out.get('Category').isin(cats)]

    can_price = (pmin is not None) and (pmax is not None) and ('Place_Id' in out.columns)
    if can_price and not out.empty:
        # peta harga per Place_Id → parse -> selaraskan ke urutan baris 'out'
        price_map = _price_to_float(places.set_index('Place_Id')['Price'])
        price_for_rows = price_map.reindex(out['Place_Id'])
        mask = (price_for_rows >= pmin) & (price_for_rows <= pmax)
        mask = mask.fillna(False).to_numpy()   # panjang mask = jumlah baris 'out'
        out = out.loc[mask]

    return out

def _similar_places(base_name: str, top_n=10):
    if base_name not in cos_df.index:
        return pd.DataFrame()
    s = cos_df[base_name].sort_values(ascending=False).drop(base_name).head(top_n)
    df = places[places['Place_Name'].isin(s.index)][['Place_Id', 'Place_Name', 'Category', 'City']].copy()
    df['CBF_Similarity'] = df['Place_Name'].map(s)
    return df.sort_values('CBF_Similarity', ascending=False)

def _add_maps_link(df: pd.DataFrame) -> pd.DataFrame:
    def maps_url(row):
        q = f"{row['Place_Name']} {row['City']}".replace(' ', '+')
        return f"https://www.google.com/maps/search/?api=1&query={q}"
    df = df.copy()
    df['Maps'] = df.apply(maps_url, axis=1)
    return df

# ---------- Session state untuk shortlist ----------
if 'shortlist' not in st.session_state:
    st.session_state['shortlist'] = pd.DataFrame(columns=['Place_Id','Place_Name','Category','City'])

# ---------- Sidebar ----------
with st.sidebar:
    st.header('Pengaturan')

    # User Id sebagai selectbox (aman untuk string/angka)
    uid_series   = ratings['User_Id']
    user_choices = uid_series.dropna().unique().tolist()
    user_id = st.selectbox('User ID', user_choices, index=0)

    topn  = st.slider('Top-N rekomendasi', 5, 30, 10)
    alpha = st.slider('Bobot CBF (alpha)', 0.0, 1.0, 0.2, 0.05)
    st.caption('Semakin besar alpha → semakin menekankan CBF. CF butuh artifacts/cf_model.h5 + TensorFlow.')

    # ----- Filter preferensi -----
    st.markdown('---')
    st.subheader('Filter preferensi')
    cities_all = sorted(places['City'].dropna().unique().tolist())
    cats_all   = sorted(places['Category'].dropna().unique().tolist())

    sel_cities = st.multiselect('City', cities_all, default=[])
    sel_cats   = st.multiselect('Category', cats_all, default=[])

    # Rentang harga (jika bisa diparse)
    price_num_all = _price_to_float(places['Price'])
    if price_num_all.notna().any():
        lo = float(price_num_all.dropna().quantile(0.05))
        hi = float(price_num_all.dropna().quantile(0.95))
        pmin, pmax = st.slider('Rentang harga (perkiraan)', min_value=0.0, max_value=max(hi, 1.0),
                               value=(lo, hi), step=1000.0)
    else:
        pmin = pmax = None
        st.caption('Harga tidak dapat diparse → filter harga dinonaktifkan.')

tab1, tab2, tab3, tab4 = st.tabs(['Hybrid', 'CBF saja', 'CF saja (opsional)', 'Mirip tempat ini'])

# ---------- Tab: Hybrid ----------
with tab1:
    st.subheader('Hybrid (CBF + CF opsional)')
    if st.button('Hitung rekomendasi (Hybrid)'):
        out = get_hybrid_recs(user_id, places, ratings, cos_df, cf_model, u2e, p2e, topn, alpha)
        out = _apply_filters(out, sel_cities, sel_cats, pmin, pmax)
        out = _add_maps_link(out)
        if out.empty:
            st.info('Tidak ada hasil dengan filter saat ini.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','Hybrid_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Pilih': st.column_config.CheckboxColumn('Pilih'),
                    'Maps': st.column_config.LinkColumn('Maps')
                }
            )
            chosen = edited[edited['Pilih']]
            colA, colB = st.columns(2)
            with colA:
                st.write(f"Terpilih: **{len(chosen)}** item")
            with colB:
                if st.button('➕ Tambah ke Shortlist', disabled=chosen.empty):
                    keep_cols = ['Place_Id','Place_Name','Category','City']
                    st.session_state['shortlist'] = pd.concat(
                        [st.session_state['shortlist'], chosen[keep_cols]],
                        ignore_index=True
                    ).drop_duplicates(subset=['Place_Id'])
                    st.success('Ditambahkan ke Shortlist.')

# ---------- Tab: CBF ----------
with tab2:
    st.subheader('CBF saja')
    if st.button('Hitung rekomendasi (CBF)'):
        out = get_cbf_recs(user_id, places, ratings, cos_df, topn)
        out = _apply_filters(out, sel_cities, sel_cats, pmin, pmax)
        out = _add_maps_link(out)
        if out.empty:
            st.info('Tidak ada hasil dengan filter saat ini.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','CBF_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Pilih': st.column_config.CheckboxColumn('Pilih'),
                    'Maps': st.column_config.LinkColumn('Maps')
                }
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (CBF)', disabled=chosen.empty):
                keep_cols = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep_cols]],
                    ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab: CF (opsional) ----------
with tab3:
    st.subheader('CF saja (opsional)')
    st.caption('Butuh artifacts/cf_model.h5 dan TensorFlow. Kalau tidak ada, tab ini mungkin kosong.')
    if st.button('Hitung rekomendasi (CF)'):
        out = get_cf_recs(user_id, places, ratings, cf_model, u2e, p2e, topn)
        if out is None:
            st.info('CF model tidak tersedia atau User ID belum dipetakan.')
        else:
            out = _apply_filters(out, sel_cities, sel_cats, pmin, pmax)
            out = _add_maps_link(out)
            out_disp = out[['Place_Id','Place_Name','Category','City','CF_Score','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Pilih': st.column_config.CheckboxColumn('Pilih'),
                    'Maps': st.column_config.LinkColumn('Maps')
                }
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (CF)', disabled=chosen.empty):
                keep_cols = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep_cols]],
                    ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Tab: Mirip tempat ini ----------
with tab4:
    st.subheader('Cari yang mirip berdasarkan 1 tempat')
    base_name = st.selectbox('Pilih tempat referensi', sorted(places['Place_Name'].unique().tolist()))
    k = st.slider('Top-N kemiripan', 5, 30, 10, key='topk_similar')
    if st.button('Cari yang mirip'):
        out = _similar_places(base_name, top_n=k)
        out = _apply_filters(out, sel_cities, sel_cats, pmin, pmax)
        out = _add_maps_link(out)
        if out.empty:
            st.info('Tidak ada hasil dengan filter saat ini.')
        else:
            out_disp = out[['Place_Id','Place_Name','Category','City','CBF_Similarity','Maps']].copy()
            out_disp.insert(0, 'Pilih', False)
            edited = st.data_editor(
                out_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Pilih': st.column_config.CheckboxColumn('Pilih'),
                    'Maps': st.column_config.LinkColumn('Maps')
                }
            )
            chosen = edited[edited['Pilih']]
            if st.button('➕ Tambah ke Shortlist (Mirip)', disabled=chosen.empty):
                keep_cols = ['Place_Id','Place_Name','Category','City']
                st.session_state['shortlist'] = pd.concat(
                    [st.session_state['shortlist'], chosen[keep_cols]],
                    ignore_index=True
                ).drop_duplicates(subset=['Place_Id'])
                st.success('Ditambahkan ke Shortlist.')

# ---------- Shortlist section ----------
st.markdown('---')
st.subheader('📝 Shortlist saya')
sl = st.session_state['shortlist']
if sl.empty:
    st.info('Belum ada item di Shortlist. Centang beberapa rekomendasi lalu klik "Tambah ke Shortlist".')
else:
    st.dataframe(sl, use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        csv = sl.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download CSV', csv, 'shortlist.csv', 'text/csv')
    with c2:
        if st.button('🗑️ Bersihkan Shortlist'):
            st.session_state['shortlist'] = pd.DataFrame(columns=sl.columns)
            st.success('Shortlist dibersihkan.')
    with c3:
        st.caption('Klik link "Maps" pada hasil untuk membuka Google Maps.')
