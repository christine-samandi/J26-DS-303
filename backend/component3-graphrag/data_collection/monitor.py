"""Scheduled law monitor: find new income tax Acts on the IRD site and add them by themselves.

    python -m data_collection.monitor                 # check once now (and add what is safe)
    python -m data_collection.monitor --every         # keep running: check once a month (30d)
    python -m data_collection.monitor --no-auto       # only report what is new, add nothing
    python -m data_collection.monitor --list          # found, added, and held with the reason
    python -m data_collection.monitor --retry URL     # try a held document again
    python -m data_collection.monitor --dismiss URL   # not relevant: stop listing it

What a check does:
1. Read each page in C3_MONITOR_PAGES (default: the IRD Acts page) and collect its PDF links
   about income tax (link text or URL with one of C3_MONITOR_KEYWORDS).
2. A link that is not in the corpus registry, not already seen and not dismissed is NEW.
3. Each new document goes through knowledge_graph.auto_update: seven safety checks (is it an
   income tax Act, does it read correctly, does the verified history stay the same, ...). If
   all pass it is added to the graph and the change is recorded. If one fails, nothing
   changes and it is HELD with the reason (maintainer notification); `--retry` tries again,
   e.g. after a fix or when Neo4j was down.

Why once a month by default: new Acts are rare (6 amendment Acts in 2017-2026) and an Act is
usually certified weeks after the change is announced, so a monthly check (one small request
per page) is enough. The cost is that a new Act can wait up to a month before the admin hears
of it; --every 7d or 1d checks more often (e.g. around the budget and April). On Windows the
check can also be scheduled without a terminal left open (Task Scheduler), e.g.:
    schtasks /create /tn "C3 law monitor" /sc monthly /tr "<path>\\.venv\\Scripts\\python.exe -m data_collection.monitor"
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from c3_common.config import COMPONENT_DIR
from data_collection.discover import PdfLink, find_pdf_links

DEFAULT_PAGES = ["https://www.ird.gov.lk/en/publications/sitepages/acts.aspx"]
DEFAULT_KEYWORDS = ["inland revenue", "ir_act", "ir act", "income tax"]
DEFAULT_EVERY = "30d"  # once a month
STATE_PATH = COMPONENT_DIR / "data" / "monitor" / "state.json"


@dataclass
class Finding:
    url: str
    text: str
    page: str
    first_seen: str
    status: str = "new"  # new | added | held | skipped
    reason: str = ""
    document_id: str | None = None


@dataclass
class State:
    last_checked: str | None = None
    pending: list[Finding] = field(default_factory=list)
    dismissed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # pages the last check could not read

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> State:
        if not path.is_file():
            return cls()
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d.get("last_checked"), [Finding(**f) for f in d.get("pending", [])],
                   list(d.get("dismissed", [])), list(d.get("errors", [])))  # fmt: skip

    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def settings() -> tuple[list[str], list[str]]:
    from c3_common.config import get_settings

    get_settings()  # loads .env
    pages = os.getenv("C3_MONITOR_PAGES")
    words = os.getenv("C3_MONITOR_KEYWORDS")
    return ([p.strip() for p in pages.split(",") if p.strip()] if pages else DEFAULT_PAGES,
            [w.strip().lower() for w in words.split(",") if w.strip()] if words
            else DEFAULT_KEYWORDS)  # fmt: skip


def about_income_tax(link: PdfLink, keywords: list[str]) -> bool:
    haystack = f"{link.text} {link.url}".lower().replace("%20", " ")
    return any(k in haystack for k in keywords)


def check(fetch, pages: list[str], keywords: list[str], state: State,
          now: datetime | None = None) -> list[Finding]:  # fmt: skip
    """One check. `fetch(url) -> (html, final_url)`. Returns the NEW findings (also added to
    state.pending). A page that cannot be read is reported and skipped, not fatal."""
    now = now or datetime.now(UTC)
    seen = {f.url for f in state.pending} | set(state.dismissed)
    new, state.errors = [], []
    for page in pages:
        try:
            html, final_url = fetch(page)
        except Exception as e:  # network down, site changed: report, try again next time
            print(f"  could not read {page}: {type(e).__name__}: {e}")
            state.errors.append(page)
            continue
        links = [lk for lk in find_pdf_links(html, final_url) if about_income_tax(lk, keywords)]
        print(f"  {page}: {len(links)} income tax PDF(s), "
              f"{sum(not lk.known for lk in links)} not in the corpus")  # fmt: skip
        for lk in links:
            if lk.known or lk.url in seen:
                continue
            f = Finding(lk.url, lk.text, page, now.isoformat(timespec="seconds"))
            state.pending.append(f)
            seen.add(lk.url)
            new.append(f)
    if len(state.errors) < len(pages):  # at least one page was read
        state.last_checked = now.isoformat(timespec="seconds")
    return new


def notify_admin(findings: list[Finding]):
    """One maintainer notification for what this check found and did."""
    from c3_common import notifications

    if not findings:
        return None
    lines = []
    for f in findings:
        name = f.text or Path(f.url).name
        lines.append({"added": f"added automatically: {name} ({f.reason})",
                      "held": f"HELD, not added: {name}. Reason: {f.reason}",
                      "skipped": f"skipped: {name} ({f.reason})"}.get(
                          f.status, f"found, not added (--no-auto): {name}"))  # fmt: skip
    held = sum(f.status == "held" for f in findings)
    n = notifications.make(
        "new_document_found", "admin",
        f"Law monitor: {len(findings)} new document(s) on the IRD site"
        + (f", {held} held" if held else ""),
        "\n".join(lines)
        + ("\nA held document changed nothing; fix the cause, then "
           "python -m data_collection.monitor --retry <URL>." if held else ""),
        url=findings[0].url,
    )  # fmt: skip
    return notifications.add(n)


def process(findings: list[Finding], ingest=None) -> None:
    """Run the automatic update on each finding and record what happened."""
    if ingest is None:
        from knowledge_graph.auto_update import download, reload_api
        from knowledge_graph.auto_update import ingest as auto_ingest

        def ingest(url):
            return auto_ingest(url, download, reload=reload_api)

    for f in findings:
        print(f"  checking {f.text or f.url}")
        out = ingest(f.url)
        for c in out.checks:
            print(f"    ok  {c}")
        f.status, f.reason, f.document_id = out.status, out.reason, out.document_id
        print(f"    {out.status.upper()}: {out.reason}")


def _fetch(url: str) -> tuple[str, str]:
    from data_collection.download import TIMEOUT_S, make_session

    response = make_session().get(url, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.text, response.url


def parse_every(text: str) -> float:
    """'30d' / '7d' / '12h' / '30m' -> seconds."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([dhm])", text.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError("use e.g. 30d, 7d, 12h or 30m")
    return float(m[1]) * {"d": 86400, "h": 3600, "m": 60}[m[2]]


def run_once(fetch=_fetch, state_path: Path = STATE_PATH, auto: bool = True,
             ingest=None) -> list[Finding]:  # fmt: skip
    pages, keywords = settings()
    state = State.load(state_path)
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] checking {len(pages)} page(s) for new tax Acts")
    new = check(fetch, pages, keywords, state)
    for f in new:
        print(f"  NEW  {f.text or '(no text)'}\n       {f.url}")
    if auto:  # the new ones, and any found earlier with --no-auto and not checked yet
        todo = new + [f for f in state.pending if f.status == "new" and f not in new]
        if todo:
            process(todo, ingest)
    state.save(state_path)
    note = notify_admin(new)
    if note:
        print(f"  maintainer notified: {note.id}")
    elif state.errors:
        print(f"  {len(state.errors)} page(s) could not be checked; trying again next time")
    else:
        print("  nothing new")
    if held := [f for f in state.pending if f.status in {"held", "new"}]:
        print(f"  {len(held)} document(s) not added: python -m data_collection.monitor --list")
    return new


def retry(url: str, state_path: Path = STATE_PATH, ingest=None) -> Finding | None:
    state = State.load(state_path)
    f = next((f for f in state.pending if f.url == url), None)
    if f is None:
        print(f"{url} is not in the monitor's list")
        return None
    process([f], ingest)
    state.save(state_path)
    notify_admin([f])
    return f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the IRD site for new tax Acts.")
    parser.add_argument(
        "--every",
        type=parse_every,
        nargs="?",
        const=parse_every(DEFAULT_EVERY),
        help="keep checking; alone = once a month (30d), or e.g. 7d, 1d",
    )
    parser.add_argument("--no-auto", action="store_true", help="report only, add nothing")
    parser.add_argument("--list", action="store_true", help="what was found and what happened")
    parser.add_argument("--retry", metavar="URL", help="run the safety checks again on a document")
    parser.add_argument("--dismiss", metavar="URL", help="stop listing a document")
    args = parser.parse_args(argv)
    if args.retry:
        f = retry(args.retry)
        return 0 if f and f.status != "held" else 1

    if args.list or args.dismiss:
        state = State.load()
        if args.dismiss:
            state.pending = [f for f in state.pending if f.url != args.dismiss]
            state.dismissed.append(args.dismiss)
            state.save()
            print(f"dismissed {args.dismiss}")
        print(f"last checked: {state.last_checked or 'never'}")
        for f in state.pending:
            reason = f"\n           {f.reason}" if f.reason else ""
            print(f"  {f.status:8} since {f.first_seen[:10]}: {f.text or '(no text)'}\n"
                  f"           {f.url}{reason}")  # fmt: skip
        if not state.pending:
            print("  nothing found yet")
        return 0
    if not args.every:
        run_once(auto=not args.no_auto)
        return 0
    print(f"Checking every {args.every / 3600:g} h (Ctrl+C to stop).")
    try:
        while True:
            run_once(auto=not args.no_auto)
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
