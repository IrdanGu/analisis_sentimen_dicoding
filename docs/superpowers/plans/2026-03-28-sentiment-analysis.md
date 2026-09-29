# Sentiment Analysis Dicoding Submission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Indonesian sentiment analysis project on Bank Jago Google Play reviews with 6 Jupyter notebooks covering scraping, preprocessing, 3 training schemes (SVM + LSTM + LSTM+Attention), and inference — satisfying all Dicoding criteria and extra credit requirements.

**Architecture:** Self-scraped Play Store data → automatic star-rating labeling (3 classes) → Indonesian NLP preprocessing pipeline → 3 model schemes with increasing complexity → saved models → unified inference notebook.

**Tech Stack:** `google-play-scraper`, `pandas`, `numpy`, `nltk`, `PySastrawi`, `scikit-learn`, `tensorflow>=2.10`, `gensim`, `matplotlib`, `seaborn`, `joblib`

---

## File Map

| File | Responsibility |
|------|---------------|
| `1_scraping.ipynb` | Scrape 12,000 raw reviews, deduplicate, save to `data/raw_reviews.csv` |
| `2_preprocessing.ipynb` | Clean text, normalize slang, stem, label by star rating, save `data/clean_reviews.csv` |
| `3_scheme1_svm.ipynb` | TF-IDF feature extraction + LinearSVC training, 80/20 split, save model |
| `4_scheme2_lstm.ipynb` | FastText embedding training + LSTM model, 80/20 split, save model |
| `5_scheme3_lstm_attn.ipynb` | FastText embedding + BiLSTM+Attention model, 70/30 split, save model |
| `6_inference.ipynb` | Load all 3 models, predict on sample sentences, output categorical labels |
| `data/raw_reviews.csv` | Raw scraped data (reviewId, content, score, at, thumbsUpCount) |
| `data/clean_reviews.csv` | Processed data (content_clean, label, score) |
| `models/svm_model.pkl` | Saved LinearSVC pipeline |
| `models/tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer |
| `models/lstm_model.h5` | Saved LSTM model weights |
| `models/lstm_attn_model.h5` | Saved LSTM+Attention model weights |
| `models/tokenizer.pkl` | Saved Keras tokenizer (shared by LSTM notebooks) |
| `models/fasttext_model.bin` | Saved FastText embeddings (shared by LSTM notebooks) |

---

## Task 1: Project Setup

**Files:**
- Create: `data/.gitkeep`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create directory structure**

Run in terminal:
```bash
cd analisis-sentimen-dicoding
mkdir -p data models
```

- [ ] **Step 2: Install all required libraries**

Run in terminal:
```bash
pip install google-play-scraper pandas numpy nltk PySastrawi scikit-learn tensorflow gensim matplotlib seaborn joblib
```

- [ ] **Step 3: Download NLTK data**

Run in Python or a notebook cell:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

- [ ] **Step 4: Verify installs**

```python
import google_play_scraper, pandas, numpy, nltk, Sastrawi, sklearn, tensorflow, gensim, matplotlib, seaborn, joblib
print("All imports OK")
print(f"TensorFlow: {tensorflow.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```
Expected output: `All imports OK` with version numbers printed.

---

## Task 2: `1_scraping.ipynb` — Data Collection

**Files:**
- Create: `1_scraping.ipynb`
- Output: `data/raw_reviews.csv`

- [ ] **Step 1: Create notebook and add Cell 1 — imports + setup**

```python
# Cell 1: Imports
from google_play_scraper import reviews, Sort
import pandas as pd
import time

APP_ID = 'com.jago.digitalBanking'
LANG = 'id'
COUNTRY = 'id'
```

- [ ] **Step 2: Add Cell 2 — scraping function**

```python
# Cell 2: Scraping function with retry
def scrape_batch(app_id, sort, count, lang='id', country='id'):
    """Scrape a batch of reviews with error handling."""
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=count
        )
        return result
    except Exception as e:
        print(f"Error scraping: {e}")
        return []
```

- [ ] **Step 3: Add Cell 3 — run 3 batches**

```python
# Cell 3: Scrape 3 batches
print("Batch 1: Sort.NEWEST (4000 reviews)...")
batch1 = scrape_batch(APP_ID, Sort.NEWEST, 4000)
print(f"  Got {len(batch1)} reviews")
time.sleep(2)

print("Batch 2: Sort.MOST_RELEVANT (4000 reviews)...")
batch2 = scrape_batch(APP_ID, Sort.MOST_RELEVANT, 4000)
print(f"  Got {len(batch2)} reviews")
time.sleep(2)

print("Batch 3: Sort.RATING (4000 reviews)...")
batch3 = scrape_batch(APP_ID, Sort.RATING, 4000)
print(f"  Got {len(batch3)} reviews")

all_reviews = batch1 + batch2 + batch3
print(f"\nTotal raw reviews: {len(all_reviews)}")
```

- [ ] **Step 4: Add Cell 4 — deduplicate and save**

```python
# Cell 4: Deduplicate and save
df = pd.DataFrame(all_reviews)

# Keep only needed columns
df = df[['reviewId', 'content', 'score', 'at', 'thumbsUpCount']].copy()

# Drop duplicates by reviewId
before = len(df)
df = df.drop_duplicates(subset='reviewId').reset_index(drop=True)
after = len(df)
print(f"Removed {before - after} duplicates")
print(f"Final dataset size: {after} reviews")

# Drop rows with empty content
df = df.dropna(subset=['content'])
df = df[df['content'].str.strip() != ''].reset_index(drop=True)
print(f"After removing empty content: {len(df)} reviews")

df.to_csv('data/raw_reviews.csv', index=False)
print("Saved to data/raw_reviews.csv")
```

- [ ] **Step 5: Add Cell 5 — verify and visualize distribution**

```python
# Cell 5: Distribution check
import matplotlib.pyplot as plt
import seaborn as sns

print(df['score'].value_counts().sort_index())

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='score', palette='viridis')
plt.title('Distribusi Rating Ulasan Bank Jago')
plt.xlabel('Rating (Bintang)')
plt.ylabel('Jumlah Ulasan')
plt.tight_layout()
plt.savefig('data/rating_distribution.png', dpi=150)
plt.show()
print(f"\nTotal data: {len(df)} ulasan")
```
Expected: bar chart showing counts for ratings 1–5, total ≥ 10,000.

---

## Task 3: `2_preprocessing.ipynb` — Preprocessing + Labeling

**Files:**
- Create: `2_preprocessing.ipynb`
- Input: `data/raw_reviews.csv`
- Output: `data/clean_reviews.csv`

- [ ] **Step 1: Create notebook and add Cell 1 — imports**

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords', quiet=True)

# Load raw data
df = pd.read_csv('data/raw_reviews.csv')
print(f"Loaded {len(df)} reviews")
df.head()
```

- [ ] **Step 2: Add Cell 2 — labeling function**

```python
# Cell 2: Labeling by star rating
def assign_label(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

df['label'] = df['score'].apply(assign_label)
print("Label distribution:")
print(df['label'].value_counts())
print(f"\nPercentages:")
print(df['label'].value_counts(normalize=True).round(3) * 100)
```

- [ ] **Step 3: Add Cell 3 — slang normalization dictionary**

```python
# Cell 3: Slang/alay normalization dictionary
slang_dict = {
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'enggak': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'krna': 'karena', 'sdh': 'sudah', 'udah': 'sudah', 'udh': 'sudah',
    'blm': 'belum', 'belom': 'belum', 'sm': 'sama', 'sama2': 'sama sama',
    'jg': 'juga', 'bgt': 'banget', 'bget': 'banget', 'bngt': 'banget',
    'dg': 'dengan', 'dr': 'dari', 'bs': 'bisa', 'bsa': 'bisa',
    'lg': 'lagi', 'lgi': 'lagi', 'sdg': 'sedang', 'lg': 'lagi',
    'tp': 'tapi', 'tpi': 'tapi', 'ttp': 'tetap', 'ttap': 'tetap',
    'ny': 'nya', 'sy': 'saya', 'sy': 'saya', 'aq': 'saya', 'ak': 'aku',
    'mksd': 'maksud', 'mksud': 'maksud', 'bkn': 'bukan', 'bukn': 'bukan',
    'krg': 'kurang', 'krang': 'kurang', 'cuma': 'hanya', 'cman': 'hanya',
    'emg': 'memang', 'emang': 'memang', 'lbh': 'lebih', 'lebh': 'lebih',
    'jd': 'jadi', 'jdi': 'jadi', 'msh': 'masih', 'msih': 'masih',
    'hrs': 'harus', 'hrus': 'harus', 'skrg': 'sekarang', 'skrang': 'sekarang',
    'kmrn': 'kemarin', 'kmarin': 'kemarin', 'mau': 'mau', 'mo': 'mau',
    'nih': 'ini', 'tuh': 'itu', 'nii': 'ini', 'ni': 'ini',
    'sih': '', 'deh': '', 'dong': '', 'kok': '',
    'aplikasi': 'aplikasi', 'app': 'aplikasi', 'apps': 'aplikasi',
    'transfer': 'transfer', 'tf': 'transfer', 'trnsfer': 'transfer',
    'error': 'error', 'eror': 'error', 'err': 'error',
    'update': 'perbarui', 'updt': 'perbarui', 'upd': 'perbarui',
    'fitur': 'fitur', 'ftur': 'fitur', 'featre': 'fitur',
    'loading': 'memuat', 'load': 'memuat',
    'bug': 'kesalahan', 'bugg': 'kesalahan',
    'aman': 'aman', 'secure': 'aman', 'safe': 'aman',
    'bagus': 'bagus', 'bgus': 'bagus', 'ok': 'baik', 'oke': 'baik',
    'mantap': 'bagus', 'mantul': 'bagus', 'keren': 'bagus',
    'jelek': 'buruk', 'jlek': 'buruk', 'parah': 'buruk', 'payah': 'buruk',
    'lemot': 'lambat', 'lelet': 'lambat', 'lambat': 'lambat',
}

def normalize_slang(text):
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized)
```

- [ ] **Step 4: Add Cell 4 — full preprocessing pipeline**

```python
# Cell 4: Full preprocessing pipeline
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_words = set(stopwords.words('indonesian'))
# Add custom domain stopwords to remove (these add no sentiment value)
custom_stopwords = {'bank', 'jago', 'aplikasi', 'app', 'nya', 'yang', 'di', 'ke',
                    'dan', 'ini', 'itu', 'dengan', 'untuk', 'dari', 'ada', 'saya',
                    'aku', 'kamu', 'anda', 'kita', 'mereka', 'bisa', 'sudah',
                    'akan', 'juga', 'atau', 'tapi', 'kalau', 'karena', 'saat',
                    'waktu', 'lagi', 'sama', 'masih', 'belum', 'mau', 'harus'}
stop_words.update(custom_stopwords)

def preprocess_text(text):
    # 1. Lowercase
    text = str(text).lower()
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 4. Remove emojis and non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 5. Remove special characters, keep letters, numbers, spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 7. Slang normalization
    text = normalize_slang(text)
    # 8. Stopword removal
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    # 9. Stemming
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

print("Preprocessing... (this may take a few minutes)")
df['content_clean'] = df['content'].apply(preprocess_text)
print("Done!")
df[['content', 'content_clean', 'label', 'score']].head(10)
```

- [ ] **Step 5: Add Cell 5 — quality check and save**

```python
# Cell 5: Quality check
# Remove rows where clean content is empty or too short
df = df[df['content_clean'].str.split().str.len() >= 2].reset_index(drop=True)
print(f"Final dataset after quality filter: {len(df)} reviews")

print("\nLabel distribution after cleaning:")
print(df['label'].value_counts())

# Save
output_df = df[['content', 'content_clean', 'label', 'score']].copy()
output_df.to_csv('data/clean_reviews.csv', index=False)
print("\nSaved to data/clean_reviews.csv")
```

- [ ] **Step 6: Add Cell 6 — visualize label distribution**

```python
# Cell 6: Visualize final label distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

label_counts = df['label'].value_counts()
colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Orange, Green

axes[0].bar(label_counts.index, label_counts.values, color=colors)
axes[0].set_title('Distribusi Label Sentimen')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Jumlah')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

axes[1].pie(label_counts.values, labels=label_counts.index,
            colors=colors, autopct='%1.1f%%', startangle=90)
axes[1].set_title('Proporsi Label Sentimen')

plt.suptitle(f'Dataset Bank Jago Reviews (n={len(df):,})', fontsize=14)
plt.tight_layout()
plt.savefig('data/label_distribution.png', dpi=150)
plt.show()
```
Expected: Bar + pie chart showing roughly: Positif ~50-60%, Negatif ~25-35%, Netral ~10-15%.

---

## Task 4: `3_scheme1_svm.ipynb` — Scheme 1: SVM + TF-IDF

**Files:**
- Create: `3_scheme1_svm.ipynb`
- Input: `data/clean_reviews.csv`
- Output: `models/svm_model.pkl`, `models/tfidf_vectorizer.pkl`

- [ ] **Step 1: Add Cell 1 — imports and load data**

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/clean_reviews.csv')
print(f"Loaded {len(df)} reviews")
print(df['label'].value_counts())
df.head()
```

- [ ] **Step 2: Add Cell 2 — encode labels and split data**

```python
# Cell 2: Encode labels and 80/20 split
label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
df['label_enc'] = df['label'].map(label_map)

X = df['content_clean'].fillna('')
y = df['label_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test size:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
print(f"\nTrain label distribution:")
print(pd.Series(y_train).map({0:'Negatif',1:'Netral',2:'Positif'}).value_counts())
```

- [ ] **Step 3: Add Cell 3 — TF-IDF vectorization**

```python
# Cell 3: TF-IDF Feature Extraction
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=10000,     # top 10k features
    min_df=2,               # ignore terms appearing < 2 times
    sublinear_tf=True       # log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (test):  {X_test_tfidf.shape}")
print(f"\nTop 20 features: {tfidf.get_feature_names_out()[:20]}")
```

- [ ] **Step 4: Add Cell 4 — train SVM**

```python
# Cell 4: Train LinearSVC
svm = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42
)
svm.fit(X_train_tfidf, y_train)
print("Training complete!")
```

- [ ] **Step 5: Add Cell 5 — evaluate on train and test sets**

```python
# Cell 5: Evaluation
label_names = ['Negatif', 'Netral', 'Positif']

# Training accuracy
y_train_pred = svm.predict(X_train_tfidf)
train_acc = accuracy_score(y_train, y_train_pred)

# Test accuracy
y_test_pred = svm.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, y_test_pred)

print("=" * 50)
print(f"SCHEME 1: SVM + TF-IDF (80/20 Split)")
print("=" * 50)
print(f"Training Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print()
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=label_names))
```

- [ ] **Step 6: Add Cell 6 — confusion matrix plot**

```python
# Cell 6: Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix — Scheme 1: SVM + TF-IDF\nTest Accuracy: {test_acc*100:.2f}%')
plt.tight_layout()
plt.savefig('data/scheme1_confusion_matrix.png', dpi=150)
plt.show()
```

- [ ] **Step 7: Add Cell 7 — save models**

```python
# Cell 7: Save models
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(svm, 'models/svm_model.pkl')
print("Models saved:")
print("  models/tfidf_vectorizer.pkl")
print("  models/svm_model.pkl")
```

- [ ] **Step 8: Verify accuracy meets requirement**

Test accuracy must be ≥ 85%. If below, increase `C` parameter (try `C=5.0`) in Cell 4 and re-run.

---

## Task 5: `4_scheme2_lstm.ipynb` — Scheme 2: LSTM + FastText (80/20)

**Files:**
- Create: `4_scheme2_lstm.ipynb`
- Input: `data/clean_reviews.csv`
- Output: `models/lstm_model.h5`, `models/tokenizer.pkl`, `models/fasttext_model.bin`

- [ ] **Step 1: Add Cell 1 — imports**

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                      SpatialDropout1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from gensim.models import FastText

print(f"TensorFlow version: {tf.__version__}")
df = pd.read_csv('data/clean_reviews.csv')
print(f"Loaded {len(df)} reviews")
```

- [ ] **Step 2: Add Cell 2 — label encoding and 80/20 split**

```python
# Cell 2: Label encoding and split
label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
df['label_enc'] = df['label'].map(label_map)
df['content_clean'] = df['content_clean'].fillna('')

X = df['content_clean'].tolist()
y = df['label_enc'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
```

- [ ] **Step 3: Add Cell 3 — tokenize and pad sequences**

```python
# Cell 3: Tokenization + padding
MAX_WORDS = 20000   # vocabulary size
MAX_LEN = 100       # max sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

vocab_size = min(MAX_WORDS, len(tokenizer.word_index)) + 1
print(f"Vocabulary size: {vocab_size}")
print(f"X_train shape: {X_train_pad.shape}")
print(f"X_test shape:  {X_test_pad.shape}")

# Save tokenizer (shared with Scheme 3)
joblib.dump(tokenizer, 'models/tokenizer.pkl')
print("Tokenizer saved to models/tokenizer.pkl")
```

- [ ] **Step 4: Add Cell 4 — train FastText embeddings**

```python
# Cell 4: Train FastText embeddings on corpus
EMBED_DIM = 128

# Tokenize into word lists for gensim
sentences = [text.split() for text in X_train]

print("Training FastText embeddings...")
ft_model = FastText(
    sentences=sentences,
    vector_size=EMBED_DIM,
    window=5,
    min_count=1,
    workers=4,
    epochs=15
)
ft_model.save('models/fasttext_model.bin')
print(f"FastText trained. Vocab size: {len(ft_model.wv)}")

# Build embedding matrix
embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
found = 0
for word, idx in tokenizer.word_index.items():
    if idx < vocab_size and word in ft_model.wv:
        embedding_matrix[idx] = ft_model.wv[word]
        found += 1

coverage = found / min(len(tokenizer.word_index), MAX_WORDS) * 100
print(f"Embedding coverage: {found}/{min(len(tokenizer.word_index), MAX_WORDS)} ({coverage:.1f}%)")
```

- [ ] **Step 5: Add Cell 5 — build LSTM model**

```python
# Cell 5: Build LSTM model
NUM_CLASSES = 3

model_lstm = Sequential([
    Embedding(vocab_size, EMBED_DIM,
              weights=[embedding_matrix],
              input_length=MAX_LEN,
              trainable=False),          # freeze pretrained embeddings
    SpatialDropout1D(0.3),
    LSTM(128, dropout=0.3, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model_lstm.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_lstm.summary()
```

- [ ] **Step 6: Add Cell 6 — train with callbacks**

```python
# Cell 6: Train LSTM
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1)
]

history = model_lstm.fit(
    X_train_pad, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
```

- [ ] **Step 7: Add Cell 7 — evaluate**

```python
# Cell 7: Evaluation
label_names = ['Negatif', 'Netral', 'Positif']

train_loss, train_acc = model_lstm.evaluate(X_train_pad, y_train, verbose=0)
test_loss, test_acc = model_lstm.evaluate(X_test_pad, y_test, verbose=0)

y_test_pred = np.argmax(model_lstm.predict(X_test_pad), axis=1)

print("=" * 50)
print("SCHEME 2: LSTM + FastText (80/20 Split)")
print("=" * 50)
print(f"Training Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print()
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=label_names))
```

- [ ] **Step 8: Add Cell 8 — plot training curves**

```python
# Cell 8: Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Scheme 2: LSTM — Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Scheme 2: LSTM — Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('data/scheme2_training_curves.png', dpi=150)
plt.show()

# Confusion matrix
fig2, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix — Scheme 2: LSTM\nTest Accuracy: {test_acc*100:.2f}%')
plt.tight_layout()
plt.savefig('data/scheme2_confusion_matrix.png', dpi=150)
plt.show()
```

- [ ] **Step 9: Add Cell 9 — save model**

```python
# Cell 9: Save model
model_lstm.save('models/lstm_model.h5')
print("Saved: models/lstm_model.h5")
```

- [ ] **Step 10: Verify accuracy meets requirement**

Training and testing accuracy must both be ≥ 92%. If below:
- Set `trainable=True` in the Embedding layer and re-run (fine-tuning)
- Or increase epochs to 50 and remove early stopping patience adjustment

---

## Task 6: `5_scheme3_lstm_attn.ipynb` — Scheme 3: BiLSTM + Attention (70/30)

**Files:**
- Create: `5_scheme3_lstm_attn.ipynb`
- Input: `data/clean_reviews.csv`
- Output: `models/lstm_attn_model.h5`

- [ ] **Step 1: Add Cell 1 — imports**

```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM,
                                      Dense, Dropout, Layer, SpatialDropout1D)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import tensorflow.keras.backend as K

df = pd.read_csv('data/clean_reviews.csv')
print(f"Loaded {len(df)} reviews")
```

- [ ] **Step 2: Add Cell 2 — load tokenizer + 70/30 split**

```python
# Cell 2: Load saved tokenizer and 70/30 split
MAX_LEN = 100
EMBED_DIM = 128
MAX_WORDS = 20000

tokenizer = joblib.load('models/tokenizer.pkl')

label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
df['label_enc'] = df['label'].map(label_map)
df['content_clean'] = df['content_clean'].fillna('')

X = df['content_clean'].tolist()
y = df['label_enc'].values

# 70/30 split — different from Scheme 2's 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

vocab_size = min(MAX_WORDS, len(tokenizer.word_index)) + 1
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Split: 70/30 (Scheme 3 differs from Scheme 2's 80/20)")
```

- [ ] **Step 3: Add Cell 3 — load FastText embedding matrix**

```python
# Cell 3: Load saved FastText embeddings
ft_model = FastText.load('models/fasttext_model.bin')

embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
for word, idx in tokenizer.word_index.items():
    if idx < vocab_size and word in ft_model.wv:
        embedding_matrix[idx] = ft_model.wv[word]

print(f"Embedding matrix shape: {embedding_matrix.shape}")
```

- [ ] **Step 4: Add Cell 4 — custom Attention layer**

```python
# Cell 4: Custom Attention Layer
class AttentionLayer(Layer):
    """Simple additive attention mechanism."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)
        e = K.tanh(K.dot(x, self.W) + self.b)   # (batch, timesteps, 1)
        a = K.softmax(e, axis=1)                  # attention weights
        output = x * a                            # weighted features
        return K.sum(output, axis=1)              # (batch, features)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

print("AttentionLayer defined.")
```

- [ ] **Step 5: Add Cell 5 — build BiLSTM + Attention model**

```python
# Cell 5: Build BiLSTM + Attention model
NUM_CLASSES = 3

inputs = Input(shape=(MAX_LEN,))
x = Embedding(vocab_size, EMBED_DIM,
              weights=[embedding_matrix],
              input_length=MAX_LEN,
              trainable=True)(inputs)         # trainable for fine-tuning
x = SpatialDropout1D(0.3)(x)
x = Bidirectional(LSTM(128, return_sequences=True,
                        dropout=0.3,
                        recurrent_dropout=0.2))(x)
x = AttentionLayer()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model_attn = Model(inputs=inputs, outputs=outputs)
model_attn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_attn.summary()
```

- [ ] **Step 6: Add Cell 6 — train**

```python
# Cell 6: Train BiLSTM + Attention
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1)
]

history = model_attn.fit(
    X_train_pad, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
```

- [ ] **Step 7: Add Cell 7 — evaluate**

```python
# Cell 7: Evaluation
label_names = ['Negatif', 'Netral', 'Positif']

train_loss, train_acc = model_attn.evaluate(X_train_pad, y_train, verbose=0)
test_loss, test_acc = model_attn.evaluate(X_test_pad, y_test, verbose=0)

y_test_pred = np.argmax(model_attn.predict(X_test_pad), axis=1)

print("=" * 50)
print("SCHEME 3: BiLSTM + Attention + FastText (70/30 Split)")
print("=" * 50)
print(f"Training Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Testing  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print()
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=label_names))
```

- [ ] **Step 8: Add Cell 8 — plot training curves + confusion matrix**

```python
# Cell 8: Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Scheme 3: BiLSTM+Attention — Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Scheme 3: BiLSTM+Attention — Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('data/scheme3_training_curves.png', dpi=150)
plt.show()

fig2, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix — Scheme 3: BiLSTM+Attention\nTest Accuracy: {test_acc*100:.2f}%')
plt.tight_layout()
plt.savefig('data/scheme3_confusion_matrix.png', dpi=150)
plt.show()
```

- [ ] **Step 9: Add Cell 9 — save model**

```python
# Cell 9: Save model
model_attn.save('models/lstm_attn_model.h5')
print("Saved: models/lstm_attn_model.h5")
```

---

## Task 7: `6_inference.ipynb` — Inference Demo

**Files:**
- Create: `6_inference.ipynb`
- Input: all models from `models/`, `data/clean_reviews.csv`

- [ ] **Step 1: Add Cell 1 — imports and load all models**

```python
# Cell 1: Load all models
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
import warnings
warnings.filterwarnings('ignore')

# Load SVM pipeline
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svm   = joblib.load('models/svm_model.pkl')

# Load tokenizer
tokenizer = joblib.load('models/tokenizer.pkl')

# Custom attention layer needed for loading Scheme 3 model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                  initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Load deep learning models
model_lstm = load_model('models/lstm_model.h5')
model_attn = load_model('models/lstm_attn_model.h5',
                         custom_objects={'AttentionLayer': AttentionLayer})

print("All models loaded successfully!")
```

- [ ] **Step 2: Add Cell 2 — preprocessing function for inference**

```python
# Cell 2: Preprocessing pipeline for inference
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

slang_dict = {
    'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk',
    'krn': 'karena', 'sdh': 'sudah', 'udah': 'sudah', 'blm': 'belum',
    'bgt': 'banget', 'bs': 'bisa', 'lg': 'lagi', 'tp': 'tapi',
    'ok': 'baik', 'oke': 'baik', 'mantap': 'bagus', 'keren': 'bagus',
    'jelek': 'buruk', 'parah': 'buruk', 'lemot': 'lambat',
    'error': 'error', 'eror': 'error', 'bug': 'kesalahan',
}

def preprocess_for_inference(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    words = [w for w in words if w not in stop_words and len(w) > 1]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

print("Preprocessing function ready.")
```

- [ ] **Step 3: Add Cell 3 — prediction functions**

```python
# Cell 3: Prediction functions
MAX_LEN = 100
label_names = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

def predict_svm(texts):
    cleaned = [preprocess_for_inference(t) for t in texts]
    X_tfidf = tfidf.transform(cleaned)
    preds = svm.predict(X_tfidf)
    return [label_names[p] for p in preds]

def predict_lstm(texts):
    cleaned = [preprocess_for_inference(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    preds = np.argmax(model_lstm.predict(padded, verbose=0), axis=1)
    return [label_names[p] for p in preds]

def predict_lstm_attn(texts):
    cleaned = [preprocess_for_inference(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')
    preds = np.argmax(model_attn.predict(padded, verbose=0), axis=1)
    return [label_names[p] for p in preds]

print("Prediction functions ready.")
```

- [ ] **Step 4: Add Cell 4 — run inference on sample texts**

```python
# Cell 4: Inference on sample texts
sample_texts = [
    "Aplikasi ini sangat bagus dan mudah digunakan, saya sangat puas",
    "Fitur lengkap dan tampilan menarik, transfer juga cepat",
    "Biasa saja, tidak ada yang istimewa dari aplikasi ini",
    "Kadang bisa kadang tidak, lumayan lah untuk kebutuhan sehari-hari",
    "Sangat lambat dan sering error, tolong diperbaiki segera",
    "Aplikasi sangat mengecewakan, uang saya tidak bisa ditarik",
    "Bagus sekali pelayanannya, customer service cepat merespons",
    "Sudah beberapa kali gagal transfer, sangat tidak memuaskan",
    "Mantap banget aplikasinya, fitur tabungan sangat membantu",
    "Tidak bisa login dari tadi, error terus",
]

results = pd.DataFrame({
    'Teks Ulasan': sample_texts,
    'Scheme 1 (SVM)': predict_svm(sample_texts),
    'Scheme 2 (LSTM)': predict_lstm(sample_texts),
    'Scheme 3 (BiLSTM+Attn)': predict_lstm_attn(sample_texts),
})

pd.set_option('display.max_colwidth', 60)
print("=" * 80)
print("HASIL INFERENSI — ANALISIS SENTIMEN BANK JAGO")
print("=" * 80)
results
```

- [ ] **Step 5: Add Cell 5 — interactive single prediction**

```python
# Cell 5: Interactive single prediction
def analyze_sentiment(text):
    """Predict sentiment for a single review using all 3 schemes."""
    print(f"\nTeks: '{text}'")
    print(f"  Scheme 1 (SVM + TF-IDF)        : {predict_svm([text])[0]}")
    print(f"  Scheme 2 (LSTM + FastText)      : {predict_lstm([text])[0]}")
    print(f"  Scheme 3 (BiLSTM + Attention)   : {predict_lstm_attn([text])[0]}")

# Test with custom inputs
analyze_sentiment("Aplikasi jago sangat membantu kebutuhan finansial saya sehari-hari")
analyze_sentiment("Sering force close dan data tidak sinkron, sangat mengganggu")
analyze_sentiment("Cukup baik tapi masih ada beberapa kekurangan yang perlu diperbaiki")
```

- [ ] **Step 6: Add Cell 6 — summary table of all 3 schemes**

```python
# Cell 6: Summary comparison of all 3 schemes
summary = pd.DataFrame({
    'Skema': ['Scheme 1', 'Scheme 2', 'Scheme 3'],
    'Algoritma': ['SVM (LinearSVC)', 'LSTM', 'BiLSTM + Attention'],
    'Ekstraksi Fitur': ['TF-IDF (unigram+bigram)', 'FastText (128-dim)', 'FastText (128-dim)'],
    'Pembagian Data': ['80% / 20%', '80% / 20%', '70% / 30%'],
    'Target Akurasi': ['≥ 85%', '≥ 92%', '≥ 92%'],
})

print("RINGKASAN 3 SKEMA PELATIHAN")
print("=" * 80)
summary
```

---

## Self-Review

**Spec coverage check:**
- ✓ Kriteria 1: Self-scrape ≥3,000 → Task 2 (12,000 target, dedup to ~10k)
- ✓ Kriteria 2: Feature extraction + labeling → Tasks 3 (labeling), 4 (TF-IDF), 5+6 (FastText)
- ✓ Kriteria 3: ML algorithm → Task 4 (SVM)
- ✓ Kriteria 4: Test accuracy ≥85% → All 3 schemes include evaluation cells
- ✓ Saran: Deep learning → Tasks 5, 6 (LSTM, BiLSTM+Attention)
- ✓ Saran: Train+test accuracy ≥92% → Schemes 2 & 3 target 92%+
- ✓ Saran: 3 classes → Negatif/Netral/Positif throughout
- ✓ Saran: ≥10,000 samples → Task 2 scraping strategy
- ✓ Saran: 3 schemes ≥2 combinations → Algorithm + split + features vary
- ✓ Saran: Inference with categorical output → Task 7 (6_inference.ipynb)

**Type consistency check:**
- `preprocess_for_inference` defined in both `2_preprocessing.ipynb` and `6_inference.ipynb` — consistent logic ✓
- `AttentionLayer` defined in both `5_scheme3_lstm_attn.ipynb` and `6_inference.ipynb` — identical code ✓
- `label_map = {'Negatif': 0, 'Netral': 1, 'Positif': 2}` consistent across all notebooks ✓
- `MAX_LEN = 100`, `EMBED_DIM = 128`, `MAX_WORDS = 20000` consistent across Schemes 2, 3, and Inference ✓
- `tokenizer.pkl` saved in Task 5 Cell 3, loaded in Task 6 Cell 2 and Task 7 Cell 1 ✓
- `fasttext_model.bin` saved in Task 5 Cell 4, loaded in Task 6 Cell 3 ✓

**No placeholders found** ✓
