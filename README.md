# Hybrid Tourism Recommender (Streamlit)

Aplikasi rekomendasi wisata **CBF + (opsional) CF**. Siap deploy di **Streamlit Community Cloud**.

## Struktur
```
reco-app/
├─ app.py
├─ reco_utils.py
├─ requirements.txt
├─ tourism_with_id.csv        # Anda tambahkan sendiri
├─ tourism_rating.csv         # Anda tambahkan sendiri
└─ artifacts/                 # opsional (untuk CF/TF-IDF cache)
   └─ cf_model.h5             # opsional
```

> **Catatan dataset**  
> Wajib ada file `tourism_with_id.csv` & `tourism_rating.csv` di root.  
> Kolom minimum:
> - `tourism_with_id.csv`: `Place_Id, Place_Name, Category, City, Description, Price`
> - `tourism_rating.csv`: `User_Id, Place_Id, Place_Ratings`

## Deploy cepat (Streamlit Community Cloud)
1. Upload folder ini ke GitHub sebagai repo public (mis. `reco-app`).
2. Buka https://share.streamlit.io → **New app** → pilih repo, branch `main`, file `app.py` → **Deploy**.
3. Jika tidak pakai CF, biarkan `requirements.txt` **tanpa** TensorFlow.

## (Opsional) Simpan artefak supaya cepat
Jika Anda sudah punya TF-IDF vectorizer dan matriks, simpan ke folder `artifacts/`:
```python
import pickle, scipy.sparse as sp
pickle.dump(vectorizer, open("artifacts/tfidf_vectorizer.pkl","wb"))
sp.save_npz("artifacts/tfidf_matrix.npz", tfidf_matrix)
```
Kalau punya model CF:
```python
model.save("artifacts/cf_model.h5")
# lalu di requirements.txt tambahkan:
# tensorflow==2.19.1
```

## Cara pakai
- Pilih **User ID**, atur **Top-N** dan **alpha** (bobot CBF vs CF).
- Tab **Hybrid** menggabungkan skor CBF & CF; jika tidak ada CF, bobot CF dianggap 0.
- Tab **CBF saja** dan **CF saja** disediakan untuk eksplorasi.

Selamat mencoba!