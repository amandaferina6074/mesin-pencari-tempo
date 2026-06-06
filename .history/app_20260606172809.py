from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Route untuk menampilkan halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk memproses pencarian saat tombol diklik
@app.route('/search', methods=['POST'])
def search():
    kata_kunci = request.form['query']
    
    # KITA AKAN MASUKKAN LOGIKA DARI .IPYNB ANDA DI SINI NANTI
    # Untuk sementara, ini data dummy (pura-pura) untuk mengecek tampilan HTML
    data_dummy = [
        {"judul": "Berita Percobaan 1", "isi": "Ini adalah contoh isi berita pertama terkait " + kata_kunci, "skor": 0.95},
        {"judul": "Berita Percobaan 2", "isi": "Ini adalah contoh isi berita kedua terkait " + kata_kunci, "skor": 0.82}
    ]
    
    return render_template('result.html', query=kata_kunci, results=data_dummy)

if __name__ == '__main__':
    app.run(debug=True)