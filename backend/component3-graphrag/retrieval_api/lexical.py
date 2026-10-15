"""Lexical text matching for retrieval: TF-IDF weighted term coverage.

For a query and a chunk, `coverage` is the share of the query's informative words (weighted by
how rare they are in the corpus, IDF) that appear in the chunk: 1.0 = every informative word is
there. Words that never occur in the Acts ("what", "my", typos) carry no weight, so they do not
lower the score. Simple, explainable and fast; dense embeddings (ChromaDB) are the next step.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# question words and dates add nothing to finding the rule (the date is in as_of_date)
STOPWORDS = set(
    """a an and are as at be by can do does for from how i in is it its me my of on or the
    this that to under was what when which who will with year years month months
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
        weights = {t: self.idf[t] for t in query_terms(query) if t in self.idf}
        total = sum(weights.values())
        if not total:
            return 0.0
        found = sum(w for t, w in weights.items() if t in self.terms.get(doc_id, ()))
        return found / total
