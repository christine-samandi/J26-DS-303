"""Show how the parser sees a PDF's layout, line by line. Used to tune `layout.py`.

    python -m data_collection.inspect_pdf raw_regulatory_docs/IRA_AMEND_02_2025_E.pdf --pages 1-3

Each line prints: page, kind (body/header/footer/margin), x0, y0, width, font size, text.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_collection.layout import classify, read_pages


def _page_range(spec: str | None, total: int) -> range:
    if not spec:
        return range(1, min(total, 3) + 1)
    first, _, last = spec.partition("-")
    return range(int(first), int(last or first) + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the layout classification of a PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--pages", help="e.g. 1-3 (default: first 3 pages)")
    args = parser.parse_args(argv)

    pages = read_pages(args.pdf)
    sizes = classify(pages)
    print(f"{args.pdf.name}: {len(pages)} pages")
    for number in _page_range(args.pages, len(pages)):
        if not 1 <= number <= len(pages):
            continue
        page = pages[number - 1]
        main = f"main text {sizes[number]}pt" if number in sizes else "no body text"
        print(f"\n--- page {number} ({page.width:.0f} x {page.height:.0f}; {main}) ---")
        for ln in page.lines:
            print(
                f"p{ln.page:<3} {ln.kind:6} x={ln.x0:5.0f} y={ln.y0:5.0f} "
                f"w={ln.width:5.0f} s={ln.size:4.1f} | {ln.text[:90]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
