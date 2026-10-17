"""Scrape a web page for PDF links, to find new or updated regulatory documents.

Usage:

    python -m data_collection.discover https://www.ird.gov.lk/en/publications/SitePages/... \
        --match amendment --match "inland revenue"

Prints every PDF link on the page whose link text or URL contains all the --match words, and
marks which ones are already in the corpus registry. Use it to spot new amendment Acts; add a
document to corpus_sources.json to start collecting it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

from data_collection.sources import load_registry


@dataclass(frozen=True)
class PdfLink:
    text: str
    url: str
    known: bool  # already in the corpus registry


def _normalise(url: str) -> str:
    return unquote(url).strip().lower()


def find_pdf_links(html: str, base_url: str, match: list[str] | None = None) -> list[PdfLink]:
    """All PDF links on an HTML page, as absolute URLs, filtered by `match` words."""
    from bs4 import BeautifulSoup

    known = {_normalise(d.url) for d in load_registry().documents}
    words = [m.lower() for m in (match or [])]
    seen: set[str] = set()
    links: list[PdfLink] = []
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        url = urljoin(base_url, a["href"].strip())
        if not urlparse(url).path.lower().endswith(".pdf") or url in seen:
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        haystack = f"{text} {unquote(url)}".lower()
        if all(w in haystack for w in words):
            seen.add(url)
            links.append(PdfLink(text=text, url=url, known=_normalise(url) in known))
    return links


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List PDF links on a page.")
    parser.add_argument("url")
    parser.add_argument("--match", action="append", default=[], help="keyword filter (repeatable)")
    args = parser.parse_args(argv)

    from data_collection.download import TIMEOUT_S, make_session

    response = make_session().get(args.url, timeout=TIMEOUT_S)
    response.raise_for_status()
    links = find_pdf_links(response.text, response.url, args.match)
    for link in links:
        mark = "in corpus" if link.known else "NEW"
        print(f"[{mark:9}] {link.text or '(no text)'}\n            {link.url}")
    print(f"\n{len(links)} PDF link(s); {sum(not lk.known for lk in links)} not in the corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
