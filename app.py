from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================================
# 1. MEMUAT KEDUA FILE DATA
# ==========================================

# A. Memuat data PROSES (Untuk dihitung oleh Mesin / TF-IDF)
with open('processed_berita_tempo.pkl', 'rb') as f:
    data_proses = pickle.load(f)
df_proses = pd.DataFrame(data_proses)
# --- Memaksa nama kolom menjadi string
df_proses.columns = df_proses.columns.astype(str)

# B. Memuat data MENTAH ASLI (Untuk ditampilkan di Website)
# Tetap menggunakan file .pkl sesuai permintaan Anda
with open('berita_tempo.pkl', 'rb') as f:
    data_mentah = pickle.load(f)
df_mentah = pd.DataFrame(data_mentah)
# --- Memaksa nama kolom menjadi string
df_mentah.columns = df_mentah.columns.astype(str)


# ==========================================
# 2. SISTEM OTOMATIS & TF-IDF
# ==========================================

# Mengambil kolom terakhir dari data proses sebagai teks bersih (stemming)
kolom_bersih = df_proses.columns[-1]

# 3. Membuat matriks pembobotan TF-IDF dari dokumen berita (menggunakan data proses)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_proses[kolom_bersih].fillna(''))


# ==========================================
# 3. RUTE HALAMAN WEB (FLASK)
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    kata_kunci = request.form['query']
    
    # 4. Menghitung Cosine Similarity antara query dari web dengan dokumen
    query_vec = vectorizer.transform([kata_kunci.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 5. Mengurutkan hasil pencarian dari skor tertinggi ke terendah
    top_indices = similarities.argsort()[::-1]
    
    results = []
    for idx in top_indices:
        skor = similarities[idx]
        
        # Hanya masukkan berita jika memiliki skor kemiripan lebih dari 0
        if skor > 0.0:  
            # MENGAMBIL TEKS DARI DATA MENTAH (df_mentah)
            # Berdasarkan struktur data: Kolom 0=no, 1=url, 2=judul, 3=isi
            results.append({
                "url": str(df_mentah.iloc[idx, 1]),   # <--- INI TAMBAHANNYA UNTUK LINK
                "judul": str(df_mentah.iloc[idx, 2]),
                "isi": str(df_mentah.iloc[idx, 3])[:250] + "...", 
                "skor": round(float(skor), 4)
            })
            
        if len(results) >= 10:
            break

    return render_template('result.html', query=kata_kunci, results=results)

if __name__ == '__main__':
    app.run(debug=True)