# LSA Blog Topic Clustering – Streamlit App

Proyek ini merupakan aplikasi **Streamlit** yang digunakan untuk **mengelompokkan artikel blog berdasarkan kesamaan topik** menggunakan pendekatan **Latent Semantic Analysis (LSA)**. Aplikasi ini menyediakan antarmuka interaktif untuk melakukan preprocessing teks, menerapkan TF-IDF, memproyeksikan ke ruang semantik LSA, dan menampilkan hasil clustering.

---

## 🚀 Fitur Utama

* **Upload Dataset** (CSV)
* **Sampling Data** (opsional, misal hanya ambil 200 data dari 80.000)
* **Preprocessing Teks**

  * Lowercase
  * Remove punctuation
  * Remove numbers
  * Tokenization
  * Stopword removal (Indonesian)
  * Stemming (Sastrawi)
* **Vectorization menggunakan TF-IDF**
* **Dimensionality Reduction menggunakan LSA (SVD)**
* **Clustering menggunakan K-Means**
* **Visualisasi 2D Scatter PCA**
* **Menampilkan Top Terms per Cluster**
* **Preview masing-masing dokumen berdasarkan label cluster**

---

## 📁 Struktur Folder

```
projek/
│── app.py
│── preprocessing.py
│── requirements.txt
│── README.md 
│── .gitignore

```

---

## ⚙️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/ilhamfajarandika/clustering-indonesian-news
cd clustering-indonesian-news
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di browser lokal Anda.

---

## 🧼 Preprocessing Teks

Preprocessing dilakukan melalui file `preprocessing.py` yang berisi fungsi:

* `clean_text()` → membersihkan teks mentah
* `preprocess_dataframe()` → preprocessing kolom tertentu dalam DataFrame
* `load_stopwords()` → memuat daftar stopword Indonesia

Setelah preprocessing, DataFrame akan memiliki kolom baru bernama **clean**.

---

## 📊 Proses Analisis

### 1. TF-IDF Vectorization

Mengubah koleksi dokumen menjadi matriks numerik berbasis bobot TF-IDF.

### 2. LSA (Latent Semantic Analysis)

Menggunakan **TruncatedSVD** untuk mengurangi dimensi dan mengungkap struktur laten topik.

### 3. Clustering (K-Means)

Pengelompokan dokumen berdasarkan representasi semantik dari LSA.

### 4. Evaluasi

Aplikasi dapat menampilkan metrik seperti:

* **Silhouette Score**
* **Davies-Bouldin Index**

---

## 📥 Sampling Dataset Besar

Anda bisa menggunakan hanya sebagian data (misal 200 dari 80.000):

```python
df = df.sample(n=200, random_state=42)
```

Aplikasi juga menyediakan opsi sampling di dalam UI.

---

## 📝 Contoh Dataset

Pastikan dataset Anda memiliki kolom teks, misalnya:

* `title`
* `content`
* `article`
* `news`

Kolom yang dipakai dapat dipilih di UI Streamlit.

---

## 🛠️ Requirements

Pastikan file `requirements.txt` berisi dependency berikut:

```
streamlit
pandas
numpy
scikit-learn
Sastrawi
matplotlib
seaborn
```
