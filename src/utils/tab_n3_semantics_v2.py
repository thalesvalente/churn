"""Semantic + sentiment classification for TAB_N3 entries using sentence embeddings."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CLUSTERS = 12
DEFAULT_RANDOM_STATE = 42
DEFAULT_SENTIMENT_THRESHOLDS = (-0.5, 0.5)
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 32

# Basic Portuguese stopword list (extendable if needed).
PORTUGUESE_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "e",
    "o",
    "os",
    "da",
    "das",
    "do",
    "dos",
    "de",
    "em",
    "um",
    "uma",
    "por",
    "para",
    "no",
    "na",
    "nos",
    "nas",
    "como",
    "que",
    "com",
}


def normalize_text(value) -> str:
    """Utility kept for comparability with the legacy classifier."""
    if isinstance(value, str):
        normalized = (
            unicodedata.normalize("NFKD", value.upper())
            .encode("ASCII", "ignore")
            .decode("ASCII")
        )
        return normalized.strip()
    return "" if value is None else str(value).upper().strip()


class TabN3SemanticSentimentModel:
    """Combines semantic clustering with a sentiment projection."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        n_clusters: int = DEFAULT_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        sentiment_thresholds: Tuple[float, float] = DEFAULT_SENTIMENT_THRESHOLDS,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sentiment_thresholds = sentiment_thresholds
        self.device = device if device is not None else DEFAULT_DEVICE
        self.batch_size = batch_size
        self._encoder: SentenceTransformer | None = None
        self._kmeans: KMeans | None = None
        self._cluster_labels: Dict[int, str] = {}
        self._sentiment_vector: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Helpers
    def _ensure_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
            print(f"Usando dispositivo: {self.device}")
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._ensure_encoder()
        return encoder.encode(
            list(texts),
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )

    @staticmethod
    def _summarize_cluster(texts: Sequence[str], top_k: int = 4) -> str:
        if not texts:
            return "cluster_vazio"
        vectorizer = TfidfVectorizer(stop_words=list(PORTUGUESE_STOPWORDS), ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] == 0:
            return " | ".join(texts[:2])
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        indices = scores.argsort()[::-1][:top_k]
        vocab = vectorizer.get_feature_names_out()
        keywords = [vocab[i] for i in indices if scores[i] > 0]
        if not keywords:
            return " | ".join(texts[:2])
        return ", ".join(keywords)

    @staticmethod
    def _prepare_texts(texts: Iterable[str]) -> List[str]:
        return ["" if text is None else str(text).strip() for text in texts]

    # ------------------------------------------------------------------
    # Public API
    def fit(self, texts: Sequence[str], sentiment_seeds: Dict[str, float] | None = None) -> None:
        clean_texts = self._prepare_texts(texts)
        if not clean_texts:
            raise ValueError("No texts provided to fit the model")

        embeddings = self._encode(clean_texts)
        n_clusters = min(self.n_clusters, max(2, len(clean_texts)))
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        self._kmeans.fit(embeddings)

        labels = self._kmeans.labels_
        df = pd.DataFrame({"text": clean_texts, "cluster": labels})
        cluster_labels: Dict[int, str] = {}
        for cluster_id in sorted(df["cluster"].unique()):
            subset = df[df["cluster"] == cluster_id]["text"].tolist()
            cluster_labels[cluster_id] = self._summarize_cluster(subset)
        self._cluster_labels = cluster_labels

        if sentiment_seeds:
            self._build_sentiment_vector(sentiment_seeds)

    def _build_sentiment_vector(self, sentiment_seeds: Dict[str, float]) -> None:
        seed_texts = list(sentiment_seeds.keys())
        seed_scores = np.array(list(sentiment_seeds.values()), dtype=float)
        seed_embeddings = self._encode(seed_texts)
        vector, *_ = np.linalg.lstsq(seed_embeddings, seed_scores, rcond=None)
        self._sentiment_vector = vector

    def _predict_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("Model must be fitted before predicting")
        return self._kmeans.predict(embeddings)

    def _score_sentiment(self, embeddings: np.ndarray) -> np.ndarray:
        if self._sentiment_vector is None:
            return np.zeros(len(embeddings))
        return embeddings @ self._sentiment_vector

    def _label_sentiment(self, scores: np.ndarray) -> List[str]:
        low, high = self.sentiment_thresholds
        labels: List[str] = []
        for score in scores:
            if score <= low:
                labels.append("insatisfacao")
            elif score >= high:
                labels.append("positivo")
            else:
                labels.append("neutro")
        return labels

    def transform(self, texts: Sequence[str]) -> pd.DataFrame:
        clean_texts = self._prepare_texts(texts)
        embeddings = self._encode(clean_texts)
        clusters = self._predict_clusters(embeddings)
        sentiment_scores = self._score_sentiment(embeddings)
        sentiment_labels = self._label_sentiment(sentiment_scores)
        semantic_labels = [self._cluster_labels.get(cid, f"cluster_{cid}") for cid in clusters]
        return pd.DataFrame(
            {
                "original_text": clean_texts,
                "semantic_group": semantic_labels,
                "sentiment_score": sentiment_scores,
                "sentiment_label": sentiment_labels,
            }
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    def save(self, path: str | Path) -> None:
        if self._kmeans is None:
            raise RuntimeError("Model must be fitted before saving")
        payload = {
            "model_name": self.model_name,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "sentiment_thresholds": self.sentiment_thresholds,
            "device": self.device,
            "batch_size": self.batch_size,
            "kmeans": self._kmeans,
            "cluster_labels": self._cluster_labels,
            "sentiment_vector": self._sentiment_vector,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "TabN3SemanticSentimentModel":
        payload = joblib.load(path)
        instance = cls(
            model_name=payload["model_name"],
            n_clusters=payload["n_clusters"],
            random_state=payload["random_state"],
            sentiment_thresholds=tuple(payload["sentiment_thresholds"]),
            device=payload.get("device"),
            batch_size=payload.get("batch_size", DEFAULT_BATCH_SIZE),
        )
        instance._kmeans = payload["kmeans"]
        instance._cluster_labels = payload["cluster_labels"]
        instance._sentiment_vector = payload["sentiment_vector"]
        return instance


def export_cluster_dictionary(model: TabN3SemanticSentimentModel, path: str | Path) -> None:
    """Utility to persist cluster label summaries for documentation."""
    if not model._cluster_labels:
        raise RuntimeError("Model has no cluster labels to export")
    data = {str(k): v for k, v in model._cluster_labels.items()}
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "TabN3SemanticSentimentModel",
    "normalize_text",
    "export_cluster_dictionary",
]
