# Sentiment Analysis — Dicoding Submission Design Spec
**Date:** 2026-03-28
**Status:** Approved

---

## Overview

Indonesian-language sentiment analysis on Google Play Store reviews for Bank Jago (`com.jago.digitalBanking`). The project fulfills all Dicoding submission criteria plus all extra credit (saran) requirements: deep learning, 3-class sentiment, 10,000+ samples, 92%+ accuracy, and 3 distinct training schemes.

---

## 1. Data Collection

**Source:** Google Play Store via `google-play-scraper` Python library
**App:** Bank Jago — `com.jago.digitalBanking`
**Language:** Indonesian (`lang='id'`, `country='id'`)

**Scraping strategy — 3 batches to maximize unique reviews:**
- Batch 1: 4,000 reviews (`Sort.NEWEST`)
- Batch 2: 4,000 reviews (`Sort.MOST_RELEVANT`)
- Batch 3: 4,000 reviews (`Sort.RATING`)
- Deduplicate by `reviewId` field
- Expected yield after dedup: **10,000–11,000 unique reviews**

**Output:** `data/raw_reviews.csv`

---

## 2. Labeling

Automatic labeling from star rating — no manual annotation needed:

| Star Rating | Sentiment Label |
|-------------|----------------|
| 1–2 stars   | `Negatif`      |
| 3 stars     | `Netral`       |
| 4–5 stars   | `Positif`      |

This satisfies the **3-class minimum** requirement.

---

## 3. Preprocessing Pipeline

Applied in sequence to the `content` (review text) column:

1. **Lowercase** — normalize casing
2. **Remove noise** — strip URLs, emails, HTML tags, emojis, special characters
3. **Slang normalization** — map common Indonesian internet slang to formal forms (e.g., `gk → tidak`, `yg → yang`, `dgn → dengan`) using a kamus alay dictionary
4. **Stopword removal** — NLTK Indonesian stopwords + custom domain stopwords (e.g., `bank`, `jago`, `aplikasi`)
5. **Stemming** — Sastrawi Indonesian stemmer
6. **Tokenization** — split into word tokens

**Output:** `data/clean_reviews.csv` with columns: `content_clean`, `label`, `score`

---

## 4. Three Training Schemes

### Scheme 1 — SVM + TF-IDF (Baseline ML)
- **Algorithm:** LinearSVC (Support Vector Machine)
- **Feature Extraction:** TF-IDF with unigram + bigram (`ngram_range=(1,2)`), `max_features=10000`
- **Data Split:** 80% train / 20% test
- **Target Accuracy:** ≥ 85%
- **Notebook:** `3_scheme1_svm.ipynb`

### Scheme 2 — LSTM + Word2Vec/FastText (Deep Learning)
- **Algorithm:** LSTM (Long Short-Term Memory) neural network
- **Feature Extraction:** Pre-trained FastText Indonesian word embeddings (128-dim)
- **Architecture:** Embedding → LSTM(128) → Dropout(0.3) → Dense(64) → Softmax(3)
- **Data Split:** 80% train / 20% test
- **Target Accuracy:** ≥ 92% (train and test)
- **Notebook:** `4_scheme2_lstm.ipynb`

### Scheme 3 — LSTM + Attention (Deep Learning, different split)
- **Algorithm:** LSTM with Bahdanau Attention mechanism
- **Feature Extraction:** Same FastText Indonesian embeddings as Scheme 2
- **Architecture:** Embedding → Bidirectional LSTM(128) → Attention → Dense(64) → Softmax(3)
- **Data Split:** 70% train / 30% test
- **Target Accuracy:** ≥ 92% (train and test)
- **Notebook:** `5_scheme3_lstm_attn.ipynb`

**Combination differences (min 2 required ✓):**
- Algorithm varies: SVM vs LSTM vs LSTM+Attention
- Data split varies: 80/20 (Scheme 1 & 2) vs 70/30 (Scheme 3)
- Feature extraction varies: TF-IDF (Scheme 1) vs FastText embeddings (Schemes 2 & 3)

---

## 5. Inference

A dedicated `6_inference.ipynb` loads all 3 trained models and runs prediction on sample texts, outputting categorical class labels (`Negatif`, `Netral`, `Positif`).

```python
# Example inference output
texts = [
    "Aplikasi ini sangat bagus dan mudah digunakan",   # → Positif
    "Biasa saja, tidak ada yang istimewa",             # → Netral
    "Sangat lambat dan sering error, sangat mengecewakan"  # → Negatif
]
```

Results are displayed as a DataFrame with columns: `text`, `scheme1_pred`, `scheme2_pred`, `scheme3_pred`.

---

## 6. Project File Structure

```
analisis-sentimen-dicoding/
├── 1_scraping.ipynb              # Criteria 1: Data collection
├── 2_preprocessing.ipynb         # Criteria 2: Feature extraction + labeling
├── 3_scheme1_svm.ipynb           # Criteria 3 + 4: SVM + TF-IDF
├── 4_scheme2_lstm.ipynb          # Extra: LSTM + FastText 80/20
├── 5_scheme3_lstm_attn.ipynb     # Extra: LSTM+Attention + FastText 70/30
├── 6_inference.ipynb             # Criteria: Inference with categorical output
├── data/
│   ├── raw_reviews.csv           # Raw scraped data
│   └── clean_reviews.csv         # Preprocessed + labeled data
├── models/
│   ├── svm_model.pkl             # Saved SVM model
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
│   ├── lstm_model.h5             # Saved LSTM model
│   ├── lstm_attn_model.h5        # Saved LSTM+Attention model
│   └── tokenizer.pkl             # Saved Keras tokenizer
└── docs/
    └── superpowers/specs/
        └── 2026-03-28-sentiment-analysis-design.md
```

---

## 7. Criteria Checklist

| Kriteria | Requirement | Approach |
|----------|-------------|----------|
| 1 | Self-scraped ≥3,000 samples | google-play-scraper, target 10,000+ |
| 2 | Feature extraction + labeling | TF-IDF / FastText + star-rating mapping |
| 3 | ML algorithm | SVM (Scheme 1), LSTM (Schemes 2 & 3) |
| 4 | Test accuracy ≥85% | All 3 schemes target ≥85%; Schemes 2&3 target ≥92% |
| Saran | Deep learning | LSTM + LSTM+Attention |
| Saran | Train+test accuracy ≥92% | Schemes 2 & 3 |
| Saran | 3 sentiment classes | Negatif / Netral / Positif |
| Saran | ≥10,000 samples | ~10,000–11,000 after dedup |
| Saran | 3 schemes with ≥2 combinations | ✓ Algorithm + split + feature vary |
| Saran | Inference with categorical output | `6_inference.ipynb` |

---

## 8. Key Libraries

```
google-play-scraper    # scraping
pandas, numpy          # data manipulation
nltk                   # stopwords
PySastrawi             # Indonesian stemmer
scikit-learn           # SVM, TF-IDF, metrics
tensorflow / keras     # LSTM, Attention
gensim                 # FastText embeddings
matplotlib, seaborn    # visualization
```
