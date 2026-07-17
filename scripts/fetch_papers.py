"""Fetch open-access papers from arXiv into data/.

    python -m scripts.fetch_papers 1706.03762            # by arXiv id
    python -m scripts.fetch_papers attention is all you need   # by search

arXiv is free and open-access, so downloading PDFs here is fine. Build an index
on whichever paper you want to query:  python -m scripts.build_index data\<id>.pdf paper
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from config import DATA_DIR

ARXIV_API = "http://export.arxiv.org/api/query"
ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ATOM = "{http://www.w3.org/2005/Atom}"


def resolve(query):
    """Return (arxiv_id, title). Accepts a bare id or a free-text search."""
    if ID_RE.match(query):
        return query, query
    r = requests.get(
        ARXIV_API,
        params={"search_query": f"all:{query}", "start": 0, "max_results": 1},
        timeout=30,
    )
    r.raise_for_status()
    entry = ET.fromstring(r.text).find(f"{ATOM}entry")
    if entry is None:
        raise SystemExit(f"No arXiv results for: {query}")
    aid = entry.findtext(f"{ATOM}id").rsplit("/abs/", 1)[-1]
    title = " ".join(entry.findtext(f"{ATOM}title").split())
    return aid, title


def download(query, out_dir=DATA_DIR):
    aid, title = resolve(query)
    out = Path(out_dir) / f"{aid.replace('/', '_')}.pdf"
    print(f"Downloading {aid}  ({title[:60]}) ...")
    r = requests.get(f"https://arxiv.org/pdf/{aid}.pdf", timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"  saved {out}  ({len(r.content) // 1024} KB)")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: python -m scripts.fetch_papers <arxiv_id ...| search query>")
    # several ids -> fetch each; anything else is a single free-text search
    if all(ID_RE.match(a) for a in args):
        for a in args:
            download(a)
    else:
        download(" ".join(args))
