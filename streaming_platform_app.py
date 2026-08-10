"""
Database Systems Project - Part IV
End-to-End Streaming Platform MVP

Technology:
- Python
- Tkinter
- SQLAlchemy ORM
- SQLite
- sentiment_module.py (TF-IDF + Logistic Regression)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from sqlalchemy import create_engine, Column, Integer, Text, Float, ForeignKey, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from sentiment_module import predict_sentiment

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "streaming_platform.db"

if not DB_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {DB_FILE.name}. Place it in the same folder as this script."
    )

engine = create_engine(f"sqlite:///{DB_FILE}", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class UserProfile(Base):
    __tablename__ = "user_profile"
    profile_id = Column(Integer, primary_key=True)
    account_id = Column(Integer, nullable=False)
    profile_name = Column(Text, nullable=False)
    birth_year = Column(Integer)
    maturity_level = Column(Text, nullable=False)
    language_preference = Column(Text, nullable=False)


class Content(Base):
    __tablename__ = "content"
    content_id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    content_type = Column(Text, nullable=False)
    release_date = Column(Text)
    runtime_minutes = Column(Integer)
    maturity_rating = Column(Text)
    original_language = Column(Text, nullable=False)
    production_studio_name = Column(Text)
    distributor_name = Column(Text)


class WatchlistItem(Base):
    __tablename__ = "watchlist_item"
    profile_id = Column(Integer, ForeignKey("user_profile.profile_id"), primary_key=True)
    content_id = Column(Integer, ForeignKey("content.content_id"), primary_key=True)
    date_added = Column(Text, nullable=False)


class RatingReview(Base):
    __tablename__ = "rating_review"
    review_id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("user_profile.profile_id"))
    content_id = Column(Integer, ForeignKey("content.content_id"), nullable=False)
    rating_value = Column(Integer, nullable=False)
    review_text = Column(Text)
    review_date = Column(Text, nullable=False)
    source_system = Column(Text, nullable=False, default="MVP")
    sentiment_label = Column(Text)
    sentiment_confidence = Column(Float)


def ensure_part4_columns():
    """Add two Part IV columns to RATING_REVIEW if necessary."""
    columns = {c["name"] for c in inspect(engine).get_columns("rating_review")}
    with engine.begin() as connection:
        if "sentiment_label" not in columns:
            connection.execute(
                text("ALTER TABLE rating_review ADD COLUMN sentiment_label TEXT")
            )
        if "sentiment_confidence" not in columns:
            connection.execute(
                text("ALTER TABLE rating_review ADD COLUMN sentiment_confidence REAL")
            )


def get_profiles():
    with SessionLocal() as session:
        return session.query(UserProfile).order_by(UserProfile.profile_id).all()


def get_content(search_text=""):
    with SessionLocal() as session:
        query = session.query(Content)
        if search_text.strip():
            query = query.filter(Content.title.ilike(f"%{search_text.strip()}%"))
        return query.order_by(Content.title).limit(200).all()


def add_to_watchlist(profile_id, content_id):
    with SessionLocal() as session:
        existing = session.query(WatchlistItem).filter_by(
            profile_id=profile_id,
            content_id=content_id,
        ).first()

        if existing:
            return "This title is already in the selected profile's watchlist."

        session.add(
            WatchlistItem(
                profile_id=profile_id,
                content_id=content_id,
                date_added=datetime.now().isoformat(timespec="seconds"),
            )
        )
        session.commit()

    return "Title added to watchlist."


def submit_review(profile_id, content_id, rating, review_text):
    """Analyze review sentiment and persist both feedback and ML insight."""
    if not review_text.strip():
        raise ValueError("Please enter a review.")

    sentiment = predict_sentiment(review_text)

    with SessionLocal() as session:
        review = session.query(RatingReview).filter_by(
            profile_id=profile_id,
            content_id=content_id,
        ).first()

        if review is None:
            review = RatingReview(
                profile_id=profile_id,
                content_id=content_id,
                rating_value=rating,
                review_text=review_text.strip(),
                review_date=datetime.now().isoformat(timespec="seconds"),
                source_system="PART4_UI",
                sentiment_label=sentiment["sentiment"],
                sentiment_confidence=sentiment["confidence"],
            )
            session.add(review)
        else:
            review.rating_value = rating
            review.review_text = review_text.strip()
            review.review_date = datetime.now().isoformat(timespec="seconds")
            review.source_system = "PART4_UI"
            review.sentiment_label = sentiment["sentiment"]
            review.sentiment_confidence = sentiment["confidence"]

        session.commit()

    return sentiment


class StreamingPlatformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Streaming Platform - Part IV MVP")
        self.geometry("920x650")
        self.profile_lookup = {}
        self.content_lookup = {}

        self._build_ui()
        self._load_profiles()
        self._load_content()

    def _build_ui(self):
        ttk.Label(
            self,
            text="Streaming Platform - End-to-End MVP",
            font=("Arial", 18, "bold"),
        ).pack(pady=(15, 8))

        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        row = ttk.Frame(main)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="User Profile:").pack(side="left")
        self.profile_var = tk.StringVar()
        self.profile_combo = ttk.Combobox(
            row, textvariable=self.profile_var, state="readonly", width=35
        )
        self.profile_combo.pack(side="left", padx=8)

        row = ttk.Frame(main)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="Search Content:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.search_var, width=45).pack(side="left", padx=8)
        ttk.Button(row, text="Search", command=self._load_content).pack(side="left")
        ttk.Button(row, text="Show All", command=self._clear_search).pack(side="left", padx=5)

        ttk.Label(main, text="Content Catalog:").pack(anchor="w", pady=(10, 3))
        self.content_list = tk.Listbox(main, height=10, exportselection=False)
        self.content_list.pack(fill="x")

        ttk.Button(
            main,
            text="Add Selected Content to Watchlist",
            command=self._add_watchlist,
        ).pack(anchor="w", pady=8)

        row = ttk.Frame(main)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="Rating:").pack(side="left")
        self.rating_var = tk.IntVar(value=5)
        ttk.Combobox(
            row,
            textvariable=self.rating_var,
            values=[1, 2, 3, 4, 5],
            state="readonly",
            width=5,
        ).pack(side="left", padx=8)

        ttk.Label(main, text="Review:").pack(anchor="w", pady=(8, 3))
        self.review_text = tk.Text(main, height=7, wrap="word")
        self.review_text.pack(fill="x")

        ttk.Button(
            main,
            text="Submit Review & Analyze Sentiment",
            command=self._submit_review,
        ).pack(anchor="w", pady=10)

        self.result_var = tk.StringVar(value="Sentiment result will appear here.")
        ttk.Label(
            main,
            textvariable=self.result_var,
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", pady=8)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main, textvariable=self.status_var).pack(anchor="w")

    def _load_profiles(self):
        profiles = get_profiles()
        labels = []
        self.profile_lookup.clear()

        for p in profiles:
            label = f"{p.profile_id} - {p.profile_name} (Account {p.account_id})"
            labels.append(label)
            self.profile_lookup[label] = p.profile_id

        self.profile_combo["values"] = labels
        if labels:
            self.profile_combo.current(0)

    def _load_content(self):
        items = get_content(self.search_var.get())
        self.content_list.delete(0, tk.END)
        self.content_lookup.clear()

        for c in items:
            label = f"{c.content_id} | {c.title} | {c.content_type}"
            self.content_list.insert(tk.END, label)
            self.content_lookup[label] = c.content_id

        self.status_var.set(f"{len(items)} content item(s) displayed.")

    def _clear_search(self):
        self.search_var.set("")
        self._load_content()

    def _selected_profile_id(self):
        label = self.profile_var.get()
        if not label:
            raise ValueError("Please select a user profile.")
        return self.profile_lookup[label]

    def _selected_content_id(self):
        selection = self.content_list.curselection()
        if not selection:
            raise ValueError("Please select a content item.")
        label = self.content_list.get(selection[0])
        return self.content_lookup[label]

    def _add_watchlist(self):
        try:
            self.status_var.set(
                add_to_watchlist(
                    self._selected_profile_id(),
                    self._selected_content_id(),
                )
            )
        except Exception as exc:
            messagebox.showerror("Watchlist Error", str(exc))

    def _submit_review(self):
        try:
            result = submit_review(
                self._selected_profile_id(),
                self._selected_content_id(),
                int(self.rating_var.get()),
                self.review_text.get("1.0", tk.END).strip(),
            )

            self.result_var.set(
                f"Sentiment: {result['sentiment']} | "
                f"Confidence: {result['confidence'] * 100:.2f}%"
            )
            self.status_var.set(
                "Review, rating, and ML sentiment saved to SQLite."
            )
        except Exception as exc:
            messagebox.showerror("Review Error", str(exc))


if __name__ == "__main__":
    ensure_part4_columns()
    StreamingPlatformApp().mainloop()
