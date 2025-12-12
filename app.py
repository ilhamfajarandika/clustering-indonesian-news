# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score

from preprocessing import preprocess_text, preprocess_dataframe
from helpers import (
    build_tfidf, apply_lsa, run_kmeans, top_terms_per_component,
    top_terms_per_cluster, reduce_2d, validate_dataframe
)

st.set_page_config(page_title="LSA Clustering - IR", layout="wide")

st.title("Pengelompokan Artikel Blog — LSA + Clustering")
st.markdown("Demo pipeline: TF-IDF → LSA (Truncated SVD) → KMeans Clustering.")

# --------------------------
# Sidebar: Upload & Parameters
# --------------------------
st.sidebar.header("Input Data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV (kolom: title, content) atau TXT per artikel",
    type=["csv","txt"], accept_multiple_files=False
)
use_sample = st.sidebar.checkbox("Gunakan sample dataset (contoh)", value=False)

st.sidebar.header("Subset Data (opsional)")
use_subset = st.sidebar.checkbox("Gunakan subset data", value=True)
subset_size = st.sidebar.number_input("Jumlah data subset", min_value=10, max_value=1000, value=200, step=10)

st.sidebar.header("Preprocessing")
lang = st.sidebar.selectbox("Bahasa", ["indonesian","english"])
remove_sw = st.sidebar.checkbox("Hapus stopwords", value=True)
do_stem = st.sidebar.checkbox("Stemming (Sastrawi - hanya ID)", value=True if lang=="indonesian" else False)
extra_stop = st.sidebar.text_input("Stopwords tambahan (pisahkan koma)", "")

st.sidebar.header("Vector & LSA")
max_features = st.sidebar.number_input("Max features TF-IDF", min_value=500, max_value=50000, value=5000, step=500)
ngram_min = st.sidebar.number_input("N-gram min", min_value=1, max_value=2, value=1)
ngram_max = st.sidebar.number_input("N-gram max", min_value=1, max_value=2, value=1)
n_components = st.sidebar.number_input("LSA (n components)", min_value=2, max_value=300, value=100, step=1)

st.sidebar.header("Clustering")
n_clusters = st.sidebar.number_input("Jumlah cluster (KMeans)", min_value=2, max_value=50, value=5, step=1)
run_button = st.sidebar.button("Jalankan LSA & Clustering")

# --------------------------
# Load dataset
# --------------------------
df = None
if use_sample:
    sample = {
        "title": [
            "Belajar Python untuk Data Science",
            "Tips Merawat Tanaman Hias",
            "Tutorial React JS Dasar",
            "Cara Memasak Nasi Goreng Spesial",
            "Machine Learning: Panduan Pemula",
            "Panduan Menanam Cabe di Pekarangan",
            "React vs Vue: Perbandingan",
            "Resep Kue Coklat Lembut",
            "Deep Learning dan Neural Network",
            "Teknik Pencahayaan Fotografi"
        ],
        "content":[
            "Python adalah bahasa pemrograman yang populer untuk data science dan machine learning.",
            "Merawat tanaman hias membutuhkan penyiraman, pencahayaan, dan pemupukan yang tepat.",
            "React JS adalah library JavaScript untuk membuat UI interaktif di web.",
            "Nasi goreng spesial dengan bawang putih, kecap, dan bumbu rahasia keluarga.",
            "Machine learning memungkinkan komputer belajar dari data tanpa diprogram secara eksplisit.",
            "Menanam cabe di pekarangan membutuhkan tanah subur, penyiraman rutin, dan sinar matahari.",
            "Perbandingan React dan Vue dalam hal performa, kemudahan, dan ekosistem.",
            "Resep kue coklat lembut memakai bahan berkualitas dan teknik pemanggangang yang benar.",
            "Deep learning adalah cabang machine learning yang memakai jaringan neural bertingkat.",
            "Pencahayaan dalam fotografi sangat menentukan mood dan kualitas foto."
        ]
    }
    df = pd.DataFrame(sample)
elif uploaded:
    if uploaded.name.endswith(".csv"):
        st.info("Membaca CSV secara bertahap...")
        chunksize = 50000
        chunks = []
        for i, chunk in enumerate(pd.read_csv(uploaded, chunksize=chunksize)):
            st.write(f"Chunk {i+1} dimuat — shape: {chunk.shape}")
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        st.success(f"Dataset penuh dimuat — shape: {df.shape}")
    else:
        text = uploaded.read().decode("utf-8")
        df = pd.DataFrame({"title":[uploaded.name], "content":[text]})
else:
    st.info("Upload CSV (kolom title, content) atau centang 'Gunakan sample dataset'.")
    df = None

# --------------------------
# Gunakan subset jika dipilih
# --------------------------
if df is not None and use_subset:
    if subset_size < len(df):
        df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
        st.info(f"Subset dataset digunakan — {subset_size} data dipilih")

# --------------------------
# Preview dan preprocessing
# --------------------------
if df is not None:
    df = validate_dataframe(df)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Preprocessing")
    if st.checkbox("Tampilkan hasil preprocessing contoh (5 baris)"):
        df['clean'] = df['content'].apply(
            lambda x: preprocess_text(
                x,
                language=lang,
                remove_stopwords=remove_sw,
                do_stem=do_stem,
                extra_stopwords=[w.strip() for w in extra_stop.split(',') if w.strip()!='']
            )
        )
        st.write(df[['title','clean']].head())

# --------------------------
# Run pipeline
# --------------------------
if df is not None and run_button:
    with st.spinner("Menjalankan preprocessing penuh..."):
        df = preprocess_dataframe(
            df,
            text_column='content',
            language=lang,
            remove_stopwords=remove_sw,
            do_stem=do_stem,
            extra_stopwords=[w.strip() for w in extra_stop.split(',') if w.strip()!='']
        )
        st.success("Preprocessing selesai ✅")

    # Pastikan kolom 'clean' ada
    if 'clean' not in df.columns:
        st.warning("Kolom 'clean' belum ada, menjalankan preprocessing sekarang...")
        df['clean'] = df['content'].astype(str).apply(
            lambda x: preprocess_text(
                x,
                language=lang,
                remove_stopwords=remove_sw,
                do_stem=do_stem,
                extra_stopwords=[w.strip() for w in extra_stop.split(',') if w.strip()!='']
            )
        )
        st.success("Kolom 'clean' berhasil dibuat.")

    with st.spinner("Membangun TF-IDF..."):
        docs_clean = df['clean'].astype(str).tolist()
        vectorizer, X_tfidf = build_tfidf(
            docs_clean,
            max_features=int(max_features),
            ngram_range=(int(ngram_min), int(ngram_max)),
            stop_words=None
        )
        st.success(f"TF-IDF matrix built — shape: {X_tfidf.shape}")

    with st.spinner("Menjalankan LSA (Truncated SVD)..."):
        max_components_allowed = X_tfidf.shape[1]
        if n_components > max_components_allowed:
            st.warning(f"n_components terlalu besar. Diset otomatis menjadi {max_components_allowed}.")
            n_components = max_components_allowed

        svd_model, X_lsa = apply_lsa(X_tfidf, n_components=int(n_components))
        st.success(f"LSA selesai — reduced shape: {X_lsa.shape}")

    with st.spinner("Menjalankan KMeans..."):
        kmeans_model, labels = run_kmeans(X_lsa, n_clusters=int(n_clusters))
        df['cluster'] = labels
        st.success("KMeans selesai")

    # Silhouette
    try:
        sil = silhouette_score(X_lsa, labels)
        st.metric("Silhouette Score (LSA space)", f"{sil:.4f}")
    except:
        pass

    # --------------------------
    # Hasil Clustering
    # --------------------------
    st.subheader("Hasil Clustering")
    st.dataframe(df[['title','cluster']].sort_values('cluster').reset_index(drop=True))

    st.subheader("Top terms per LSA component (topic)")
    top_topics = top_terms_per_component(svd_model, vectorizer, top_n=10)
    for i, terms in enumerate(top_topics[:min(10, len(top_topics))]):
        st.markdown(f"**Topik {i+1}:** " + ", ".join(terms))

    st.subheader("Top terms per cluster (berdasarkan rata-rata TF-IDF)")
    cluster_terms = top_terms_per_cluster(kmeans_model, vectorizer, X_tfidf, top_n=10)
    for k, terms in cluster_terms.items():
        st.markdown(f"**Cluster {k}:** " + ", ".join(terms))

    st.subheader("Visualisasi 2D (LSA / PCA fallback)")
    reduced = reduce_2d(X_tfidf.toarray(), method='pca', svd_model=svd_model)
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(reduced[:,0], reduced[:,1], c=labels)
    ax.set_title("2D Projection of Articles")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    st.pyplot(fig)

    st.subheader("Wordcloud per cluster")
    selected_cluster = st.selectbox("Pilih cluster untuk wordcloud", options=sorted(df['cluster'].unique()))
    texts = " ".join(df[df['cluster']==selected_cluster]['clean'].fillna(''))
    if texts.strip()=="":
        st.write("Tidak ada teks bersih untuk cluster ini.")
    else:
        wc = WordCloud(width=800, height=400).generate(texts)
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)

    st.subheader("Download hasil")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV hasil (judul, isi, cluster)", data=csv, file_name='lsa_clusters.csv', mime='text/csv')

    st.success("Pipeline selesai ✅")
