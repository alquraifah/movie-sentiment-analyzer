# Movie Sentiment Analyzer
### IMDB Binary Sentiment Classification — NLP + Machine Learning

A production-ready Flask web application that classifies movie reviews as
**Positive** or **Negative** using a real trained Logistic Regression model
with TF-IDF vectorization.

---

## Run Locally

### Step 1 — Install dependencies
```
pip install -r requirements.txt
```

### Step 2 — Place the dataset
Copy **IMDB Dataset.csv** into the `data/` folder:
```
movie-sentiment-app/
└── data/
    └── IMDB Dataset.csv
```

### Step 3 — Train the model
```
python train_model.py
```

### Step 4 — Start the web app
```
python app.py
```
Open **http://localhost:5000**

---

## Deploy to Vercel

### Step 1 — Install Git and push the project to GitHub

Make sure Git is installed: https://git-scm.com/downloads

Then in the project folder run:
```
git init
git add .
git commit -m "first commit"
```

Create a new repository on https://github.com then push:
```
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

> The `data/` folder and large model files are excluded by `.gitignore`.
> Only the four small files your app needs are committed:
> - `models/logistic_regression_model.pkl`  (40 KB)
> - `models/tfidf_vectorizer.pkl`           (175 KB)
> - `models/metrics.json`                   (1 KB)
> - `models/shapes.json`                    (0.1 KB)

### Step 2 — Connect to Vercel

1. Go to https://vercel.com and sign in with your GitHub account
2. Click **Add New Project**
3. Select your GitHub repository from the list
4. Click **Deploy** — Vercel auto-detects the `vercel.json` configuration

### Step 3 — Done

Vercel will build and deploy the app automatically.
Your live URL will be:
```
https://YOUR_PROJECT_NAME.vercel.app
```

---

## Project Structure

```
movie-sentiment-app/
├── app.py                  ← Flask backend
├── train_model.py          ← ML training pipeline (run locally)
├── vercel.json             ← Vercel deployment config
├── requirements.txt
├── .gitignore
├── README.md
├── data/                   ← place IMDB Dataset.csv here (gitignored)
├── models/
│   ├── logistic_regression_model.pkl   ← committed to git
│   ├── tfidf_vectorizer.pkl            ← committed to git
│   ├── metrics.json                    ← committed to git
│   └── shapes.json                     ← committed to git
├── static/
│   ├── css/style.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── analyzer.html
    ├── comparison.html
    ├── features.html
    ├── confusion_matrix.html
    ├── about.html
    └── not_trained.html
```

---

## ML Pipeline

| Step | Details |
|------|---------|
| Dataset | IMDB — 50,000 reviews (49,582 after dedup) |
| Preprocessing | HTML removal · punctuation strip · lowercase · stopwords · Porter stemming |
| Labels | positive → 1 · negative → 0 |
| Split | 80% train / 20% test · random_state=42 |
| Vectorizer | TF-IDF · max_features=5000 |
| Primary model | Logistic Regression · max_iter=1000 |
| Best accuracy | 88.62% |
