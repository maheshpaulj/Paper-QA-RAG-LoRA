"""Turn a PDF into text chunks (page + section metadata) and figure chunks.

Phase 3 note: we do NOT extract embedded image XObjects. Many paper figures are
vector plots with no raster image at all (in the ImageNet paper, pages 3 and 6
have zero embedded images but do have figures), and pages that *do* have them are
full of fragments and logos. Rendering the page region above each caption gets
the real figure either way.
"""
import re
import fitz  # PyMuPDF

from config import CHUNK_SIZE, CHUNK_OVERLAP, ROOT
from src.sections import match_header

# A real caption is "Figure 3: ...", "Figure 3. ..." or "Figure 1.1: ..." -- papers
# disagree on all of it. ImageNet uses a colon, ResNet a period, GPT-3 numbers
# figures per section ("Figure 1.1"). Each variant silently yielded ZERO figures
# until a paper using it got indexed.
# The discriminator against in-text references ("Figure 1.2 illustrates ...") is
# the period/colon *after* the number -- a reference has only a space there.
FIG_CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?|table)\s*\.?\s*\d+(?:\.\d+)*\s*[.:]\s", re.IGNORECASE
)
MAX_FIG_HEIGHT = 400  # points to look above a caption
MIN_FIG_SIDE = 40     # ignore slivers
# Only prose paragraphs and other captions bound a figure. Figures are full of
# their own short text (axis labels, "(a) (b)", legends) -- clipping at those
# collapses the crop to nothing and silently drops the figure.
PROSE_LEN = 200

# Section headers come from src/sections.py so tagging and routing can't drift.


def extract_pages(pdf_path):
    """Return [{page, text}, ...] in reading order."""
    doc = fitz.open(pdf_path)
    pages = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(doc)]
    doc.close()
    return pages


def extract_meta(pdf_path):
    """Document-level facts that live in no single chunk.

    Authors are deliberately NOT guessed here. The embedded author field is
    routinely absent, truncated or wrong (GPT-3's PDF names 2 of its ~31 authors,
    and not the first), and an earlier line-scanning heuristic was no better.
    The pipeline reads authors off the title page instead -- see
    Retriever.front_matter. Only facts we can state exactly belong in here.
    """
    doc = fitz.open(pdf_path)
    md = doc.metadata or {}
    text = "\n".join(page.get_text("text") for page in doc)
    
    pub_date = None
    pub_year = None
    
    creation_date = md.get("creationDate")
    if creation_date:
        date_match = re.search(r"D:(\d{4})(\d{2})(\d{2})", creation_date)
        if date_match:
            pub_year = date_match.group(1)
            pub_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

    if not pub_date:
        # Look for arXiv date in first 1000 chars
        arxiv_match = re.search(r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[.*?\]\s+(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})", text[:1000], re.IGNORECASE)
        if arxiv_match:
            pub_date = arxiv_match.group(1)
            pub_year = pub_date.split()[-1]
        else:
            # Look for typical month year pattern
            date_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", text[:1000], re.IGNORECASE)
            if date_match:
                pub_date = f"{date_match.group(1)} {date_match.group(2)}"
                pub_year = date_match.group(2)

    meta = {
        "page_count": doc.page_count,
        "title": (md.get("title") or "").strip() or "Untitled",
        "reference_count": count_references(text),  # None when not countable
        "publication_date": pub_date,
        "publication_year": pub_year,
    }
    doc.close()
    return meta


REF_HEADER_RE = re.compile(r"^\s*(?:\d+\.?\s+)?(?:references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
REF_NUM_RE = re.compile(r"^\s*\[(\d{1,3})\]", re.MULTILINE)


def count_references(text):
    """Number of entries in the bibliography, or None if it can't be counted.

    Only numeric-bracketed bibliographies ("[1] ... [40] ...") are counted, and
    only when the numbers run contiguously from 1 -- then the highest number IS
    the count, exactly.

    Author-year lists (BERT, VGG) and alphanumeric keys (GPT-3's "[ADG+16]") are
    deliberately left uncounted. Heuristics for them were tried and were wrong by
    a wide margin -- author-year undercounted 43 vs ~60, alpha keys overcounted
    144 vs ~88 by sweeping up appendix citations. A confidently wrong number is
    worse than admitting we can't count, so those return None and the pipeline
    says so.
    """
    headers = list(REF_HEADER_RE.finditer(text))
    if not headers:
        return None
    tail = text[headers[-1].end():]
    nums = [int(n) for n in REF_NUM_RE.findall(tail)]
    if not nums:
        return None
    highest = max(nums)
    found = len(set(nums) & set(range(1, highest + 1)))
    # contiguous enough that the highest label is the entry count
    if highest >= 5 and found >= 0.8 * highest:
        return highest
    return None


def _guess_section(line, current):
    # canonical, not the literal header text -- "Results"/"Result" are one section
    return match_header(line) or current


def chunk_pages(pages):
    """Greedy chunker: accumulate paragraphs up to CHUNK_SIZE, with char overlap.

    Each chunk records the page it started on and the last section header seen.
    """
    chunks, buf, buf_page, section, cid = [], "", 1, "Body", 0

    def flush():
        nonlocal buf, cid
        if buf.strip():
            chunks.append({
                "id": f"C{cid}",
                "text": buf.strip(),
                "page": buf_page,
                "section": section,
            })
            cid += 1

    for p in pages:
        for para in p["text"].split("\n"):
            section = _guess_section(para, section)
            if not para.strip():
                continue
            if len(buf) + len(para) + 1 > CHUNK_SIZE and buf:
                flush()
                buf = buf[-CHUNK_OVERLAP:]  # keep tail for context overlap
                buf_page = p["page"]
            if not buf:
                buf_page = p["page"]
            buf += " " + para.strip()
    flush()
    return chunks


def extract_figures(pdf_path, out_dir):
    """Render the figure above each caption. Returns figure chunks whose `text`
    is the caption -- captions are gold for retrieval, and the rendered image is
    what we hand to the LLM at answer time.

    `out_dir` must be per-paper: figure ids restart at F0 for every document, so
    a shared directory silently overwrites one paper's figures with another's.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    figures = []

    for pno, page in enumerate(doc, start=1):
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], " ".join(b[4].split())
            if not FIG_CAPTION_RE.match(text):
                continue

            # The figure sits directly above the caption, in the caption's column.
            # Stop at the nearest prose block or preceding caption -- see PROSE_LEN.
            top = max(0.0, y0 - MAX_FIG_HEIGHT)
            for ob in blocks:
                if ob is b or ob[3] > y0:
                    continue
                if not (ob[2] > x0 and ob[0] < x1):  # must share the caption's column
                    continue
                ob_text = " ".join(ob[4].split())
                if FIG_CAPTION_RE.match(ob_text) or len(ob_text) > PROSE_LEN:
                    top = max(top, ob[3])
            rect = fitz.Rect(x0 - 4, top, x1 + 4, y0 - 2)
            if rect.height < MIN_FIG_SIDE or rect.width < MIN_FIG_SIDE:
                continue

            fid = len(figures)
            img_path = out_dir / f"F{fid}_p{pno}.png"
            page.get_pixmap(clip=rect, dpi=130).save(str(img_path))
            figures.append({
                "id": f"F{fid}",
                "type": "figure",
                "text": text,
                "page": pno,
                "section": "Figure",
                "image_path": img_path.relative_to(ROOT).as_posix(),
            })

    doc.close()
    return figures


def ingest_pdf(pdf_path):
    return chunk_pages(extract_pages(pdf_path))