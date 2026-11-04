"""Inspectable lexical retrieval, reciprocal rank fusion and exact evidence citations."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class Document:
    id: str
    station: str
    valid_from: str
    valid_to: str | None
    text: str


def rrf(rankings, k=60, window=20):
    if k <= 0 or window < 1:
        raise ValueError("positive rank constant and window required")
    scores = Counter()
    for ranking in rankings:
        seen = set()
        for rank, key in enumerate(ranking[:window], 1):
            if key in seen:
                raise ValueError("duplicate document in ranking")
            seen.add(key)
            scores[key] += 1 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))


class Index:
    def __init__(self, documents):
        self.documents = list(documents)
        if not self.documents or len({d.id for d in self.documents}) != len(self.documents):
            raise ValueError("nonempty documents with unique IDs required")
        for d in self.documents:
            start = date.fromisoformat(d.valid_from)
            if d.valid_to is not None and date.fromisoformat(d.valid_to) <= start:
                raise ValueError("valid_to is exclusive and must follow valid_from")
        self.terms = [Counter(tokens(d.text)) for d in self.documents]
        self.length = np.array([sum(t.values()) for t in self.terms])
        self.average_length = float(self.length.mean())
        self.df = Counter(t for terms in self.terms for t in terms)
        self.word = TfidfVectorizer(ngram_range=(1, 2))
        self.word_matrix = self.word.fit_transform(d.text for d in self.documents)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_matrix = self.char.fit_transform(d.text for d in self.documents)

    def search(self, query, station, as_of="2026-11-04", mode="hybrid", limit=5):
        if mode not in {"bm25", "word", "character", "hybrid"} or limit < 1 or not query.strip():
            raise ValueError("invalid search mode, limit or empty query")
        when = date.fromisoformat(as_of)
        eligible = [
            i
            for i, d in enumerate(self.documents)
            if d.station == station
            and date.fromisoformat(d.valid_from) <= when
            and (d.valid_to is None or when < date.fromisoformat(d.valid_to))
        ]
        if not eligible:
            return []
        bm = np.zeros(len(self.documents))
        for term in set(tokens(query)):
            idf = np.log(1 + (len(self.documents) - self.df[term] + 0.5) / (self.df[term] + 0.5))
            tf = np.array([d[term] for d in self.terms], float)
            bm += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.length / self.average_length))
        word = (self.word_matrix @ self.word.transform([query]).T).toarray().ravel()
        char = (self.char_matrix @ self.char.transform([query]).T).toarray().ravel()
        scores = {"bm25": bm, "word": word, "character": char}
        rankings = {
            key: sorted(eligible, key=lambda i: (-values[i], self.documents[i].id))
            for key, values in scores.items()
        }
        if mode == "hybrid":
            order = rrf([rankings["bm25"], rankings["character"]], window=20)
        else:
            order = rankings[mode]
        return [
            {
                "document": self.documents[i],
                "bm25": float(bm[i]),
                "word": float(word[i]),
                "character": float(char[i]),
            }
            for i in order[:limit]
        ]


def cite(document):
    # A citation is an exact byte-for-byte quote plus source identity; not an entailment score.
    return {
        "document_id": document.id,
        "sha256": hashlib.sha256(document.text.encode()).hexdigest(),
        "start": 0,
        "end": len(document.text),
        "quote": document.text,
    }


def verify_citation(citation, document):
    return (
        citation.get("document_id") == document.id
        and citation.get("sha256") == hashlib.sha256(document.text.encode()).hexdigest()
        and type(citation.get("start")) is int
        and type(citation.get("end")) is int
        and 0 <= citation["start"] < citation["end"] <= len(document.text)
        and document.text[citation["start"] : citation["end"]] == citation.get("quote")
    )


def answer(index, query, station, cutoff, mode="hybrid"):
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be a finite cosine threshold")
    hits = index.search(query, station, mode=mode)
    if not hits or hits[0]["word"] < cutoff:
        return {
            "abstained": True,
            "citation": None,
            "evidence_score": 0.0 if not hits else hits[0]["word"],
        }
    return {
        "abstained": False,
        "citation": cite(hits[0]["document"]),
        "evidence_score": hits[0]["word"],
    }


def ranking_metrics(ranking, relevant, k=5):
    ranking = ranking[:k]
    relevant = set(relevant)
    if not relevant:
        raise ValueError("retrieval metrics require at least one relevant document")
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking must contain unique IDs")
    matches = np.array([doc in relevant for doc in ranking])
    recall = float(matches.sum() / len(relevant))
    rr = next((1 / (i + 1) for i, hit in enumerate(matches) if hit), 0.0)
    dcg = sum(1 / np.log2(i + 2) for i, hit in enumerate(matches) if hit)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"recall_at_5": recall, "mrr_at_5": rr, "ndcg_at_5": float(dcg / ideal)}
