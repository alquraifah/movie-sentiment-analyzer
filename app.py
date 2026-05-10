"""
app.py
------
IMDB Movie Sentiment Analyzer - Flask Backend

Compatible with local development and Vercel serverless deployment.
Run  python train_model.py  once to generate the model files, then deploy.
"""

import os
import re
import sys
import json
import pickle

import nltk
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────────────────────────────────────
# NLTK setup
# Vercel's filesystem is read-only — NLTK data must be written to /tmp.
# On Windows (local dev) the default user directory is used instead.
# ─────────────────────────────────────────────────────────────────────────────
if not sys.platform.startswith("win"):
    _NLTK_DIR = "/tmp/nltk_data"
    os.makedirs(_NLTK_DIR, exist_ok=True)
    if _NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DIR)
    nltk.download("stopwords", download_dir=_NLTK_DIR, quiet=True)
else:
    nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# Absolute paths — required for Vercel where CWD != project root
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")


# ─────────────────────────────────────────────────────────────────────────────
# Artefact loader — no fallback, no demo data
# ─────────────────────────────────────────────────────────────────────────────
def _load_pkl(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_json(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_artefacts() -> dict:
    return {
        "tfidf":   _load_pkl("tfidf_vectorizer.pkl"),
        "model":   _load_pkl("logistic_regression_model.pkl"),
        "metrics": _load_json("metrics.json"),
        "shapes":  _load_json("shapes.json"),
    }


A = load_artefacts()
MODELS_READY = A["tfidf"] is not None and A["model"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing — identical pipeline used in train_model.py
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text   = BeautifulSoup(text, "lxml").get_text()
    text   = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    stops  = set(stopwords.words("english"))
    ps     = PorterStemmer()
    return " ".join(ps.stem(t) for t in tokens if t not in stops)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", models_ready=MODELS_READY)


@app.route("/analyzer")
def analyzer():
    return render_template("analyzer.html", models_ready=MODELS_READY)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_READY:
        return jsonify({
            "error": "Model not found. Please run python train_model.py first."
        }), 503

    data   = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Please enter a review."}), 400

    cleaned    = preprocess(review)
    vec        = A["tfidf"].transform([cleaned])
    prediction = int(A["model"].predict(vec)[0])
    proba      = A["model"].predict_proba(vec)[0]
    confidence = round(float(max(proba)) * 100, 2)
    sentiment  = "positive" if prediction == 1 else "negative"

    return jsonify({"sentiment": sentiment, "confidence": confidence})


@app.route("/comparison")
def comparison():
    if A["metrics"] is None:
        return render_template("not_trained.html")
    best_name = max(A["metrics"], key=lambda k: A["metrics"][k]["accuracy"])
    return render_template("comparison.html", metrics=A["metrics"], best_name=best_name)


@app.route("/features")
def features():
    if A["shapes"] is None:
        return render_template("not_trained.html")
    return render_template("features.html", shapes=A["shapes"])


@app.route("/confusion-matrix")
def confusion_matrix_page():
    if A["metrics"] is None:
        return render_template("not_trained.html")
    return render_template("confusion_matrix.html", metrics=A["metrics"])


@app.route("/about")
def about():
    return render_template("about.html")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not MODELS_READY:
        print("\n" + "=" * 60)
        print("  WARNING: trained model files not found in models/")
        print("  Run  python train_model.py  before starting the server.")
        print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
