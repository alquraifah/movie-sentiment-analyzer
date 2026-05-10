"""
train_model.py
--------------
IMDB Sentiment Analysis - Full ML Training Pipeline

Steps:
  1. Load  data/IMDB Dataset.csv
  2. Remove duplicates / null values
  3. Preprocess text  (HTML > symbols > lowercase > stopwords > stemming)
  4. Encode labels    (positive=1 / negative=0)
  5. Train/test split (80/20, random_state=42)
  6. TF-IDF vectorization   (max_features=5000)
  7. Train five models and record metrics
  8. Save the primary model  ->  models/logistic_regression_model.pkl
     Save the vectorizer     ->  models/tfidf_vectorizer.pkl
     Save all metrics        ->  models/metrics.json
     Save feature shapes     ->  models/shapes.json

Run:
    python train_model.py
"""

import os
import re
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

# Force UTF-8 output so the console never chokes on any character
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_PATH = os.path.join("data", "IMDB Dataset.csv")
MODELS_DIR   = "models"
SEP          = "=" * 60
THIN         = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing  (identical logic used in app.py at prediction time)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    1. Remove HTML tags          (BeautifulSoup)
    2. Remove symbols/punctuation  ([^a-zA-Z])
    3. Lowercase
    4. Remove English stop-words  (NLTK)
    5. Porter stemming
    """
    text   = BeautifulSoup(text, "lxml").get_text()
    text   = re.sub(r"[^a-zA-Z]", " ", text)
    text   = text.lower()
    tokens = text.split()
    stops  = set(stopwords.words("english"))
    ps     = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens if t not in stops]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":         round(float(accuracy_score (y_true, y_pred)), 4),
        "precision":        round(float(precision_score(y_true, y_pred)), 4),
        "recall":           round(float(recall_score   (y_true, y_pred)), 4),
        "f1":               round(float(f1_score       (y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_pkl(obj, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  Saved -> {path}")


def save_json(obj, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save():

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [1/7]  Loading dataset ...")
    print(SEP)

    if not os.path.exists(DATASET_PATH):
        print(f"\n  ERROR: '{DATASET_PATH}' not found.")
        print("  Place 'IMDB Dataset.csv' inside the data/ folder and re-run.\n")
        return

    df = pd.read_csv(DATASET_PATH)
    print(f"  Raw shape     : {df.shape}")

    # ── 2. Clean ──────────────────────────────────────────────────────────────
    print(f"\n  [2/7]  Removing duplicates and null values ...")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(f"  Clean shape   : {df.shape}")
    print(f"  Positive      : {(df['sentiment'] == 'positive').sum()}")
    print(f"  Negative      : {(df['sentiment'] == 'negative').sum()}")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    print(f"\n  [3/7]  Preprocessing reviews ...")
    print("         (HTML strip -> punctuation -> lowercase -> stopwords -> Porter stem)")
    df["cleaned"] = df["review"].apply(preprocess_text)
    print("  Done.")

    # ── 4. Encode labels ──────────────────────────────────────────────────────
    print(f"\n  [4/7]  Encoding labels  (positive=1 / negative=0) ...")
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # ── 5. Split ──────────────────────────────────────────────────────────────
    print(f"\n  [5/7]  Train/test split  (80% / 20%, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.20, random_state=42
    )
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")

    # ── 6. Vectorize ──────────────────────────────────────────────────────────
    print(f"\n  [6/7]  Vectorizing (TF-IDF + BoW, max_features=5000) ...")

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    print(f"  TF-IDF train  : {X_train_tfidf.shape}")
    print(f"  TF-IDF test   : {X_test_tfidf.shape}")

    bow = CountVectorizer(max_features=5000)
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow  = bow.transform(X_test)
    print(f"  BoW    train  : {X_train_bow.shape}")
    print(f"  BoW    test   : {X_test_bow.shape}")

    # Record real shapes for the Features page
    shapes = {
        "tfidf_train": str(X_train_tfidf.shape),
        "tfidf_test":  str(X_test_tfidf.shape),
        "bow_train":   str(X_train_bow.shape),
        "bow_test":    str(X_test_bow.shape),
    }

    # ── 7. Train models ───────────────────────────────────────────────────────
    print(f"\n  [7/7]  Training models ...")
    all_metrics = {}

    print("         . Naive Bayes (TF-IDF) ...")
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    all_metrics["Naive Bayes (TF-IDF)"] = compute_metrics(
        y_test, nb_tfidf.predict(X_test_tfidf)
    )
    print(f"           Accuracy: {all_metrics['Naive Bayes (TF-IDF)']['accuracy']:.4f}")

    print("         . Naive Bayes (BoW) ...")
    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, y_train)
    all_metrics["Naive Bayes (BoW)"] = compute_metrics(
        y_test, nb_bow.predict(X_test_bow)
    )
    print(f"           Accuracy: {all_metrics['Naive Bayes (BoW)']['accuracy']:.4f}")

    print("         . Logistic Regression (TF-IDF) [PRIMARY] ...")
    lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
    lr_tfidf.fit(X_train_tfidf, y_train)
    all_metrics["Logistic Regression (TF-IDF)"] = compute_metrics(
        y_test, lr_tfidf.predict(X_test_tfidf)
    )
    print(f"           Accuracy: {all_metrics['Logistic Regression (TF-IDF)']['accuracy']:.4f}")

    print("         . Logistic Regression (BoW) ...")
    lr_bow = LogisticRegression(max_iter=1000, random_state=42)
    lr_bow.fit(X_train_bow, y_train)
    all_metrics["Logistic Regression (BoW)"] = compute_metrics(
        y_test, lr_bow.predict(X_test_bow)
    )
    print(f"           Accuracy: {all_metrics['Logistic Regression (BoW)']['accuracy']:.4f}")

    print("         . Neural Network - MLPClassifier (TF-IDF) ...")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train_tfidf, y_train)
    all_metrics["Neural Network (TF-IDF)"] = compute_metrics(
        y_test, mlp.predict(X_test_tfidf)
    )
    print(f"           Accuracy: {all_metrics['Neural Network (TF-IDF)']['accuracy']:.4f}")

    # ── 8. Save artefacts ─────────────────────────────────────────────────────
    print(f"\n  Saving artefacts to '{MODELS_DIR}/' ...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Primary prediction pipeline
    save_pkl(lr_tfidf,  "logistic_regression_model.pkl")
    save_pkl(tfidf,     "tfidf_vectorizer.pkl")

    # Auxiliary models (used by comparison + confusion-matrix pages)
    save_pkl(nb_tfidf,  "nb_tfidf_model.pkl")
    save_pkl(nb_bow,    "nb_bow_model.pkl")
    save_pkl(lr_bow,    "lr_bow_model.pkl")
    save_pkl(mlp,       "mlp_tfidf_model.pkl")
    save_pkl(bow,       "bow_vectorizer.pkl")

    # JSON data files
    save_json(all_metrics, "metrics.json")
    save_json(shapes,      "shapes.json")

    # ── Results summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESULTS SUMMARY")
    print(SEP)
    print(f"  {'Model':<36} {'Acc':>6}  {'Pre':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {THIN}")
    for name, m in all_metrics.items():
        tag = " [PRIMARY]" if name == "Logistic Regression (TF-IDF)" else ""
        print(
            f"  {name:<36} "
            f"{m['accuracy']:.4f}  {m['precision']:.4f}  "
            f"{m['recall']:.4f}  {m['f1']:.4f}{tag}"
        )
    print(SEP)
    print("\n  Training complete.  Start the web app with:\n")
    print("      python app.py\n")


if __name__ == "__main__":
    train_and_save()
