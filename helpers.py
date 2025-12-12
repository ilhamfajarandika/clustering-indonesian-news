# helpers.py
import re
import string
import numpy as np
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import stopwords

# Pastikan stopwords NLTK sudah di-download di lingkungan dev/user:
# nltk.download('stopwords')

# Initialize Indonesian stemmer (Sastrawi)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Minimal default Indonesian stopwords (gabungkan NLTK + custom)
def get_stopwords(language='indonesian'):
    try:
        if language.lower() in ['indonesian', 'id', 'indo']:
            # use NLTK stopwords if available (it has 'indonesian' since some versions),
            # otherwise fallback to a small custom list
            try:
                sw = set(stopwords.words('indonesian'))
            except Exception:
                sw = set([
                    'yang','dan','di','ke','dari','pada','ini','itu','dengan','untuk','sebuah',
                    'adalah','atau','sebagai','oleh','saat','juga','bukan','karena','pada'
                ])
            # add common punct/words
            return sw
        else:
            try:
                return set(stopwords.words('english'))
            except Exception:
                return set()
    except Exception:
        return set()

def clean_text(text):
    """
    Basic cleaning: remove html tags, non-alphanumeric (except whitespace), lowercasing.
    """
    if not isinstance(text, str):
        text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace newlines/tabs with space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove punctuation (keep spaces)
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # lowercase
    text = text.lower()
    return text

def preprocess_text(text, remove_stopwords=True, language='indonesian', do_stem=True, extra_stopwords=None):
    text = clean_text(text)
    tokens = text.split()
    if remove_stopwords:
        sw = get_stopwords(language)
        if extra_stopwords:
            sw = sw.union(set([w.strip().lower() for w in extra_stopwords]))
        tokens = [t for t in tokens if t not in sw]
    if do_stem and language.lower() in ['indonesian', 'id', 'indo']:
        # apply Sastrawi stemmer word-by-word
        tokens = [stemmer.stem(t) for t in tokens]
    # Rejoin
    return ' '.join(tokens)

def build_tfidf(docs, max_features=5000, ngram_range=(1,1), stop_words=None):
    """
    Return vectorizer and tfidf matrix (sparse).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X

def apply_lsa(X_tfidf, n_components=100, random_state=42):
    """
    TruncatedSVD to get reduced latent semantic space.
    Returns: svd_model, X_reduced (dense)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X_tfidf)
    return svd, X_reduced

def run_kmeans(X, n_clusters=5, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def top_terms_per_component(svd_model, vectorizer, top_n=10):
    """
    For each component (topic) in SVD, get top terms by absolute value in components_.
    """
    terms = np.array(vectorizer.get_feature_names_out())
    comps = svd_model.components_
    topics = []
    for comp in comps:
        idx = np.argsort(np.abs(comp))[::-1][:top_n]
        topics.append(list(terms[idx]))
    return topics

def top_terms_per_cluster(kmeans_model, vectorizer, tfidf_matrix, top_n=10):
    """
    Compute average tf-idf per cluster and show top terms for each cluster.
    """
    labels = kmeans_model.labels_
    terms = np.array(vectorizer.get_feature_names_out())
    tfidf_arr = tfidf_matrix.toarray()
    clusters_terms = {}
    for k in np.unique(labels):
        rows = tfidf_arr[labels == k]
        if rows.size == 0:
            clusters_terms[k] = []
            continue
        mean_tfidf = np.mean(rows, axis=0)
        top_idx = np.argsort(mean_tfidf)[::-1][:top_n]
        clusters_terms[k] = list(terms[top_idx])
    return clusters_terms

def reduce_2d(X, method='lsa_pca', svd_model=None):
    """
    Reduce X to 2D for plotting.
    - method='lsa_pca': if svd_model provided and X is tfidf -> use TruncatedSVD to 2 dims
    - method='pca': use PCA on X
    Expects dense matrix
    """
    if method == 'lsa_pca' and svd_model is not None:
        # If SVD model was trained with n_components>=2, we can use it directly.
        if svd_model.n_components >= 2:
            return svd_model.transform(X)[:,:2]
    # fallback PCA
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)

def validate_dataframe(df):
    """
    Ensure df has 'title' and 'content' columns. If only 'text' present, map to content.
    """
    cols = df.columns.str.lower().tolist()
    if 'content' in cols and 'title' in cols:
        # normalize column names
        df = df.rename(columns={df.columns[cols.index('title')]: 'title', df.columns[cols.index('content')]: 'content'})
        return df
    if 'text' in cols:
        df = df.rename(columns={df.columns[cols.index('text')]: 'content'})
        if 'title' not in cols:
            df['title'] = df['content'].apply(lambda x: (x[:60] + '...') if isinstance(x,str) and len(x)>60 else str(x))
        return df
    # try first two columns
    if len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: 'title', df.columns[1]: 'content'})
        return df
    # else assume single column content
    df = df.rename(columns={df.columns[0]: 'content'})
    if 'title' not in df.columns:
        df['title'] = df['content'].apply(lambda x: (x[:60] + '...') if isinstance(x,str) and len(x)>60 else str(x))
    return df
