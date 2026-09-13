"""
retrieval_service.py
---------------------
Lightweight TF-IDF retrieval over the "reference" split of the dataset.

Why TF-IDF instead of a vector DB / dense embeddings?
- The dataset is small (a few hundred rows), so an in-memory sparse matrix
  is both fast and trivially reproducible with no extra infrastructure.
- It keeps the dependency footprint (scikit-learn only) small for a
  time-boxed implementation, per the challenge's own guidance to prioritize
  reliability over sophistication at this scale.
- Trade-off: TF-IDF is lexical, not semantic, so paraphrased queries with
  little vocabulary overlap may retrieve weaker matches than dense
  embeddings would. This is called out explicitly in the README.

Retrieval is restricted to the "reference" split only, so evaluation-split
emails are never leaked into the few-shot context used for generation.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.helpers import normalize_text


class RetrievalService:
    def __init__(self, dataframe: pd.DataFrame, split_col: str = "split"):
        self.reference_df = dataframe[dataframe[split_col] == "reference"].reset_index(drop=True)
        self._corpus = [normalize_text(t) for t in self.reference_df["incoming_email"].tolist()]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        if self._corpus:
            self._matrix = self.vectorizer.fit_transform(self._corpus)
        else:
            self._matrix = None

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top_k reference records most similar to `query`, each with a similarity score."""
        if self._matrix is None or not query.strip():
            return []

        query_vec = self.vectorizer.transform([normalize_text(query)])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = sims.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score <= 0:
                continue
            row = self.reference_df.iloc[idx]
            results.append({
                "id": row["id"],
                "category": row["category"],
                "incoming_email": row["incoming_email"],
                "historical_reply": row["historical_reply"],
                "similarity": round(score, 3),
            })
        return results
