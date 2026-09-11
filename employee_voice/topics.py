"""
Topic discovery. Tries BERTopic + sentence-transformers first; falls back to
TF-IDF + KMeans. Output column shape is identical.

BERTopic generates stable integer topic IDs (with -1 for outliers).
"""
from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import EMBEDDING_MODEL_ID

log = logging.getLogger(__name__)

# Avoid sklearn "ConvergenceWarning" spam for KMeans on tiny corpora.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _try_bertopic(texts: List[str]) -> pd.DataFrame | None:
    try:
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP
    except Exception as exc:
        log.info("BERTopic unavailable (%s) — using TF-IDF/KMeans fallback", exc)
        return None

    n = len(texts)
    if n < 5:
        return None  # too few docs for BERTopic to be meaningful

    log.info("BERTopic on %d documents with %s", n, EMBEDDING_MODEL_ID)
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    umap = UMAP(n_neighbors=min(15, n - 1), n_components=min(5, n - 2),
                metric="cosine", random_state=42)
    topic_model = BERTopic(umap_model=umap, calculate_probabilities=False,
                           verbose=False, min_topic_size=max(2, n // 20))
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)

    info = topic_model.get_topic_info()
    # info: Topic, Count, Name, Representation
    name_map = {int(row.Topic): row.Name for row in info.itertuples()}
    kw_map = {int(row.Topic): row.Representation for row in info.itertuples()}

    return pd.DataFrame({
        "Topic": [int(t) for t in topics],
        "TopicName": [name_map.get(int(t), "outlier") for t in topics],
        "TopicKeywords": [",".join(kw_map.get(int(t), [])[:8]) for t in topics],
    })


def _tfidf_kmeans(texts: List[str], k: int = 5) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                          stop_words="english")
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return pd.DataFrame({"Topic": [0] * len(texts),
                             "TopicName": ["general"] * len(texts),
                             "TopicKeywords": [""] * len(texts)})

    n_clusters = max(1, min(k, len(texts)))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(X)

    terms = vec.get_feature_names_out()
    centroids = km.cluster_centers_
    top_terms = []
    for c in range(n_clusters):
        idx = centroids[c].argsort()[::-1][:8]
        top_terms.append([terms[i] for i in idx])
    # outliers: smallest cluster -> -1
    counts = np.bincount(labels, minlength=n_clusters)
    smallest = int(np.argmin(counts))
    labels = [-1 if lab == smallest else int(lab) for lab in labels]
    name_map = {i: " / ".join(top_terms[i][:3]) for i in range(n_clusters)}
    name_map[-1] = "outlier"
    kw_map = {i: ",".join(top_terms[i]) for i in range(n_clusters)}
    kw_map[-1] = ""

    return pd.DataFrame({
        "Topic": labels,
        "TopicName": [name_map[t] for t in labels],
        "TopicKeywords": [kw_map[t] for t in labels],
    })


def discover_topics(texts: List[str]) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (per-doc topic df, topic_metadata).

    topic_metadata maps topic_id -> {"Name": str, "Keywords": list[str], "Count": int}
    suitable for emitting dim_topic.csv.
    """
    bt = _try_bertopic(texts)
    if bt is not None:
        meta: dict = {}
        for t in sorted(set(bt["Topic"])):
            sub = bt[bt["Topic"] == t]
            meta[int(t)] = {
                "Name": sub["TopicName"].iloc[0],
                "Keywords": sub["TopicKeywords"].iloc[0].split(",") if sub["TopicKeywords"].iloc[0] else [],
                "Count": int(len(sub)),
            }
        return bt, meta

    log.info("Falling back to TF-IDF + KMeans topic discovery")
    tfidf = _tfidf_kmeans(texts)
    meta = {}
    for t in sorted(set(tfidf["Topic"])):
        sub = tfidf[tfidf["Topic"] == t]
        meta[int(t)] = {
            "Name": sub["TopicName"].iloc[0],
            "Keywords": sub["TopicKeywords"].iloc[0].split(",") if sub["TopicKeywords"].iloc[0] else [],
            "Count": int(len(sub)),
        }
    return tfidf, meta
