from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Memuat DataFrame berita dari file pkl hasil preprocessing Anda
with open('processed_berita_tempo.pkl', 'rb') as f:
    df_berita = pickle.load(f)

# 2. Sistem Otomatis: Mencari nama kolom secara mandiri agar tidak terjadi error
kolom_judul = next((c for c in df_berita.columns if 'judul' in c.lower() or 'title' in c.lower()), df_berita.columns[0])
kolom_isi = next((c for c in df_berita.columns if 'isi' in c.lower() or 'content' in c.lower()), df_berita.columns[1] if len(df_berita.columns) > 1 else df_berita.columns[0])
kolom_bersih = next((c for c in df_berita.columns if 'bersih' in c.lower() or 'stem' in c.lower() or 'clean' in c.lower() or 'proses' in c.lower()), df_berita.columns[-1])

# 3. Membuat matriks pembobotan TF-IDF dari dokumen berita
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_berita[kolom_bersih].fillna(''))

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
            results.append({
                "judul": df_berita.iloc[idx][kolom_judul],
                "isi": str(df_berita.iloc[idx][kolom_isi])[:250] + "...", # Potong isi berita agar tidak kepanjangan di web
                "skor": round(float(skor), 4)
            })
            
        # Batasi hasil yang muncul di halaman web maksimal 10 berita teratas
        if len(results) >= 10:
            break

    return render_template('result.html', query=kata_kunci, results=results)

if __name__ == '__main__':
    app.run(debug=True)