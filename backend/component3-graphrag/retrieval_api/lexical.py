"""Lexical text matching for retrieval: TF-IDF weighted term coverage.

For a query and a chunk, `coverage` is the share of the query's informative words (weighted by
how rare they are in the corpus, IDF) that appear in the chunk: 1.0 = every informative word is
there. Question words ("what", "my") are removed first; a word the Acts never use ("VAT")
counts as a rare word that is missing, so out-of-corpus questions score low.
Simple, explainable and fast; dense embeddings (ChromaDB) are the next step.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# question words and dates add nothing to finding the rule (the date is in as_of_date)
STOPWORDS = set(
    """a an and are as at be by can do does for from how i in is it its me my of on or the
    this that to under was what when which who will with year years month months
    much many get got need should would could there their about any all some please tell
    know want i'm im our we you your his her she he they them than also into on""".split()
    + """
    january february march april may june july august september october november december
    applied apply applies current currently now today rule rules law""".split()
)

# words people use that the Acts do not ("tax slabs") -> the Acts' own words
SYNONYMS = {
    "slab": ["rate", "taxable", "exceeding"],
    "band": ["rate", "taxable", "exceeding"],
    "bracket": ["rate", "taxable", "exceeding"],
    "allowance": ["relief"],
    "deductible": ["deduction"],
    "salary": ["employment"],
    "wage": ["employment"],
}


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def query_terms(text: str) -> list[str]:
    terms = tokens(text)
    for t in list(terms):
        terms += SYNONYMS.get(t, [])
    return list(dict.fromkeys(terms))


class LexicalIndex:
    def __init__(self, docs: dict[str, str]):
        self.terms = {doc_id: set(tokens(text)) for doc_id, text in docs.items()}
        df = Counter(t for ts in self.terms.values() for t in ts)
        n = max(len(docs), 1)
        self.idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}

    def coverage(self, query: str, doc_id: str) -> float:
        # a word the Acts never use ("VAT") is as informative as the rarest word: it cannot be
        # found, so a question about something outside the corpus scores low
        rarest = max(self.idf.values(), default=0.0)
        weights = {t: self.idf.get(t, rarest) for t in query_terms(query)}
        total = sum(weights.values())
        if not total:
            return 0.0
        found = sum(w for t, w in weights.items() if t in self.terms.get(doc_id, ()))
        return found / total


def bm25_tokens(text: str) -> list[str]:
    """Words and numbers (a flat search may match "2021" in a question to "No. 10 of 2021")."""
    return [_stem(w) for w in re.findall(r"[a-z]+|\d+", text.lower()) if w not in STOPWORDS]


class BM25:
    """Okapi BM25 (k1 = 1.5, b = 0.75), the standard sparse retriever used by flat RAG."""

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.tf = [Counter(bm25_tokens(docs[i])) for i in self.ids]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for tf in self.tf for t in tf)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        terms = [t for t in bm25_tokens(query) if t in self.idf]
        scores = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            if s > 0:
                scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: -x[1])[:k]
