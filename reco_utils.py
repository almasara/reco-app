import os, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- LOAD DATA ----------
def _read_csv_smart(path):
    # UTF-8 dulu; kalau gagal, fallback latin-1. Baris rusak di-skip.
    try:
        return pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')

def load_data():
    places_path = 'tourism_with_id.csv'
    ratings_path = 'tourism_rating.csv'
    if not os.path.exists(places_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError("Tidak menemukan 'tourism_with_id.csv' atau 'tourism_rating.csv' di root proyek.")

    places  = _read_csv_smart(places_path)
    ratings = _read_csv_smart(ratings_path)

    # bersihkan kolom sampah jika ada
    drop_cols = [c for c in ['Unnamed: 0','Unnamed: 11','Unnamed: 12'] if c in places.columns]
    if drop_cols:
        places = places.drop(columns=drop_cols, errors='ignore')

    # validasi kolom wajib
    need_p = ['Place_Id','Place_Name','Category','City','Description','Price']
    miss_p = [c for c in need_p if c not in places.columns]
    if miss_p:
        raise KeyError(f'Kolom wajib hilang di tourism_with_id.csv: {miss_p}')

    need_r = ['User_Id','Place_Id','Place_Ratings']
    miss_r = [c for c in need_r if c not in ratings.columns]
    if miss_r:
        raise KeyError(f'Kolom wajib hilang di tourism_rating.csv: {miss_r}')

    # isi NA & cast ke string untuk fitur teks
    for c in ['Place_Name','Category','City','Description','Price']:
        places[c] = places[c].fillna('').astype(str)

    return places, ratings

# ---------- CBF INDEX ----------
def build_cbf_index(places: pd.DataFrame):
    def map_price_token(x):
        try:
            val = float(x)
        except Exception:
            val = None
        if val is None: return ''
        if val == 0: return 'price_free'
        if val <= 50000: return 'price_low'
        if val <= 150000: return 'price_medium'
        return 'price_high'

    df = places.copy()
    df['price_token'] = df['Price'].apply(map_price_token)
    df['combined_features'] = (
        ((df['Category'] + ' ') * 3) +
        (df['City'] + ' ') +
        (df['Place_Name'] + ' ') +
        (df['Description'] + ' ') +
        (df['price_token'])
    ).str.lower()

    # kalau ada cache TF-IDF simpan di artifacts/, akan dipakai
    vec_path = os.path.join('artifacts','tfidf_vectorizer.pkl')
    X_path   = os.path.join('artifacts','tfidf_matrix.npz')
    vectorizer, X = None, None
    if os.path.exists(vec_path) and os.path.exists(X_path):
        try:
            vectorizer = pickle.load(open(vec_path,'rb'))
            from scipy.sparse import load_npz
            X = load_npz(X_path)
        except Exception:
            vectorizer, X = None, None

    if vectorizer is None or X is None:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
        X = vectorizer.fit_transform(df['combined_features'])

    cos = cosine_similarity(X)
    cos_df = pd.DataFrame(cos, index=df['Place_Name'], columns=df['Place_Name'])
    return df, vectorizer, X, cos_df

# ---------- CBF RECS ----------
def get_cbf_recs(user_id, places, ratings, cos_df, top_n=10):
    pid2name = dict(zip(places['Place_Id'], places['Place_Name']))
    user_hist = ratings[ratings['User_Id']==user_id][['Place_Id','Place_Ratings']]

    if user_hist.empty:
        out = places[['Place_Id','Place_Name','Category','City']].copy()
        out['CBF_Score'] = 0.0
        return out.head(top_n)

    rated_names = [pid2name.get(pid,'') for pid in user_hist['Place_Id']]
    rdict = dict(zip(rated_names, user_hist['Place_Ratings']))

    scores = pd.Series(0.0, index=cos_df.columns)
    tot = 0.0
    for nm, rt in rdict.items():
        if nm in cos_df.columns:
            scores += cos_df[nm] * rt
            tot += rt
    if tot > 0: scores /= tot

    already = set(user_hist['Place_Id'])
    df = places.copy()
    df['CBF_Score'] = df['Place_Name'].map(scores).fillna(0)
    df = df[~df['Place_Id'].isin(already)].sort_values('CBF_Score', ascending=False).head(top_n)
    return df[['Place_Id','Place_Name','Category','City','CBF_Score']]

# ---------- CF (opsional) ----------
def try_load_cf(model_path='artifacts/cf_model.h5'):
    if not os.path.exists(model_path):
        return None
    import tensorflow as tf  # hanya berhasil jika TF ada di requirements & runtime mendukung
    return tf.keras.models.load_model(model_path, compile=False)

def build_cf_maps(ratings):
    user_ids  = ratings['User_Id'].unique().tolist()
    place_ids = ratings['Place_Id'].unique().tolist()
    u2e = {x:i for i,x in enumerate(user_ids)}
    p2e = {x:i for i,x in enumerate(place_ids)}
    e2u = {i:x for x,i in u2e.items()}
    e2p = {i:x for x,i in p2e.items()}
    return u2e, p2e, e2u, e2p

def get_cf_recs(user_id, places, ratings, cf_model, u2e, p2e, top_n=10):
    if cf_model is None or user_id not in u2e:
        return None
    rated = set(ratings.loc[ratings['User_Id']==user_id,'Place_Id'].tolist())
    candidates = [pid for pid in p2e.keys() if pid not in rated]
    if not candidates:
        return None
    import numpy as np
    X = np.array([[u2e[user_id], p2e[pid]] for pid in candidates], dtype=np.int64)
    preds = cf_model.predict(X, verbose=0).flatten()
    preds = np.clip(preds*5.0, 1.0, 5.0)
    df = places[places['Place_Id'].isin(candidates)].copy()
    score_map = {pid:s for pid,s in zip(candidates, preds)}
    df['CF_Score'] = df['Place_Id'].map(score_map)
    return df.sort_values('CF_Score', ascending=False).head(top_n)[['Place_Id','Place_Name','Category','City','CF_Score']]

# ---------- HYBRID ----------
def get_hybrid_recs(user_id, places, ratings, cos_df, cf_model=None, u2e=None, p2e=None, top_n=10, alpha=0.2):
    cbf = get_cbf_recs(user_id, places, ratings, cos_df, top_n=1000)[['Place_Id','CBF_Score']].copy()
    if cbf['CBF_Score'].max() > cbf['CBF_Score'].min():
        cbf['CBF_Score_Norm'] = (cbf['CBF_Score']-cbf['CBF_Score'].min())/(cbf['CBF_Score'].max()-cbf['CBF_Score'].min())
    else:
        cbf['CBF_Score_Norm'] = 0.0

    cf = get_cf_recs(user_id, places, ratings, cf_model, u2e, p2e, top_n=1000)
    if cf is not None:
        cf = cf[['Place_Id','CF_Score']].copy()
        if cf['CF_Score'].max() > cf['CF_Score'].min():
            cf['CF_Score_Norm'] = (cf['CF_Score']-cf['CF_Score'].min())/(cf['CF_Score'].max()-cf['CF_Score'].min())
        else:
            cf['CF_Score_Norm'] = 0.0
    else:
        cf = pd.DataFrame({'Place_Id': cbf['Place_Id'], 'CF_Score_Norm': [0.0]*len(cbf)})

    hy = cbf.merge(cf, on='Place_Id', how='outer').fillna(0)
    hy['Hybrid_Score'] = alpha*hy['CBF_Score_Norm'] + (1-alpha)*hy['CF_Score_Norm']
    hy = hy.merge(places[['Place_Id','Place_Name','Category','City']], on='Place_Id', how='left')
    cols = ['Place_Id','Place_Name','Category','City','CBF_Score_Norm','CF_Score_Norm','Hybrid_Score']
    return hy.sort_values('Hybrid_Score', ascending=False).head(top_n)[cols]
