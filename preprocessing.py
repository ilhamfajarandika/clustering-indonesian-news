# preprocessing.py

import re
import string
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pastikan stopwords sudah di-download:
# nltk.download('stopwords')

# ==========================
# Inisialisasi Stemmer
# ==========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ==========================
# Stopwords
# ==========================
def get_stopwords(language='indonesian', extra_stopwords=None):
    """
    Return set of stopwords.
    language: 'indonesian' or 'english'
    extra_stopwords: list tambahan stopwords
    """
    sw = set()
    if language.lower() in ['indonesian','id','indo']:
        try:
            sw = set(stopwords.words('indonesian'))
        except:
            # fallback manual
            sw = set([
                'yang','dan','di','ke','dari','pada','ini','itu','dengan','untuk',
                'sebuah','adalah','atau','sebagai','oleh','saat','juga','bukan','karena'
            ])
    else:
        try:
            sw = set(stopwords.words('english'))
        except:
            sw = set()
    
    if extra_stopwords:
        sw = sw.union(set([w.strip().lower() for w in extra_stopwords if w.strip()!='']))
    return sw

# ==========================
# Cleaning Text
# ==========================
def clean_text(text):
    """
    Bersihkan teks:
    - Lowercase
    - Remove HTML tags
    - Remove URL
    - Remove angka dan tanda baca
    - Replace multiple spaces
    """
    if not isinstance(text,str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)         # remove HTML
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove URL
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # remove angka & tanda baca
    text = re.sub(r'\s+', ' ', text).strip()   # remove multiple spaces
    return text

# ==========================
# Preprocessing Lengkap
# ==========================
def preprocess_text(text, language='indonesian', remove_stopwords=True, do_stem=True, extra_stopwords=None):
    """
    Lakukan:
    - Cleaning
    - Stopwords removal
    - Stemming (Hanya untuk bahasa Indonesia)
    """
    text = clean_text(text)
    tokens = text.split()
    
    if remove_stopwords:
        sw = get_stopwords(language=language, extra_stopwords=extra_stopwords)
        tokens = [t for t in tokens if t not in sw]
    
    if do_stem and language.lower() in ['indonesian','id','indo']:
        tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)

# ==========================
# Preprocess dataframe
# ==========================
def preprocess_dataframe(df, text_column='content', language='indonesian', remove_stopwords=True, do_stem=True, extra_stopwords=None):
    """
    Preprocess seluruh kolom teks di dataframe.
    Menambahkan kolom baru: clean_content
    """
    df = df.copy()
    df['clean_content'] = df[text_column].astype(str).apply(
        lambda x: preprocess_text(
            x,
            language=language,
            remove_stopwords=remove_stopwords,
            do_stem=do_stem,
            extra_stopwords=extra_stopwords
        )
    )
    return df
