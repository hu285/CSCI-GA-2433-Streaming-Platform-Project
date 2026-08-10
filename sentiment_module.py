"""
Part IV - Data-Driven Sentiment Module
Streaming Platform Database Systems Project
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "IMDB Dataset.csv"
MODEL_FILE = BASE_DIR / "sentiment_model.joblib"
METADATA_FILE = BASE_DIR / "sentiment_model_metadata.json"
RANDOM_STATE = 42


def calculate_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_review_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. Place it beside this script."
        )

    df = pd.read_csv(DATA_FILE)

    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError("Dataset must contain review and sentiment columns.")

    df = df.dropna(subset=["review", "sentiment"])
    df = df.drop_duplicates(subset=["review"]).copy()

    df["sentiment"] = (
        df["sentiment"]
        .str.lower()
        .map({"negative": 0, "positive": 1})
    )

    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    return df


def train_model() -> dict:
    df = load_review_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions), 4),
        "recall": round(recall_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions), 4),
    }

    joblib.dump(model, MODEL_FILE)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": DATA_FILE.name,
        "source_hash": calculate_file_hash(DATA_FILE),
        "rows_used": len(df),
        "model": "TF-IDF + Logistic Regression",
        "metrics": metrics,
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("Model trained.")
    print(metadata)
    return metadata


def source_data_changed() -> bool:
    if not MODEL_FILE.exists() or not METADATA_FILE.exists():
        return True

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return metadata.get("source_hash") != calculate_file_hash(DATA_FILE)


def retrain_if_needed() -> bool:
    if source_data_changed():
        print("Source data changed or no model exists. Re-training...")
        train_model()
        return True

    print("Source data unchanged. Existing model is current.")
    return False


def get_model():
    retrain_if_needed()
    return joblib.load(MODEL_FILE)


def predict_sentiment(review_text: str) -> dict:
    if not review_text or not review_text.strip():
        raise ValueError("Review text cannot be empty.")

    model = get_model()
    prediction = int(model.predict([review_text])[0])
    probabilities = model.predict_proba([review_text])[0]
    confidence = float(probabilities[prediction])

    return {
        "sentiment": "POSITIVE" if prediction == 1 else "NEGATIVE",
        "confidence": round(confidence, 4),
    }


if __name__ == "__main__":
    retrain_if_needed()

    sample_review = (
        "I really enjoyed this movie. "
        "The story was interesting and the acting was excellent."
    )

    result = predict_sentiment(sample_review)

    print("\nSample prediction")
    print("Review:", sample_review)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
