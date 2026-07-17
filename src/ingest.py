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

# A real caption is "Figure 3: ..." or "Figure 3. ..." -- papers use both styles
# (ImageNet uses the colon, ResNet the period). Requiring a period/colon *after*
# the number is what keeps out in-text references like "Fig. 9 illustrates ...",
# which have nothing but a space there.
FIG_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?|table)\s*\.?\s*\d+\s*[.:]\s", re.IGNORECASE)
MAX_FIG_HEIGHT = 400  # points to look above a caption
MIN_FIG_SIDE = 40     # ignore slivers
# Only prose paragraphs and other captions bound a figure. Figures are full of
# their own short text (axis labels, "(a) (b)", legends) -- clipping at those
# collapses the crop to nothing and silently drops the figure.
PROSE_LEN = 200

# Heuristic: lines that look like a section header in a research paper.
SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    r"abstract|introduction|related work|background|method(?:s|ology)?|approach|"
    r"experiment(?:s|al)?|result(?:s)?|evaluation|discussion|conclusion(?:s)?|"
    r"references|appendix"
    r")\b",
    re.IGNORECASE,
)


def extract_pages(pdf_path):
    """Return [{page, text}, ...] in reading order."""
    doc = fitz.open(pdf_path)
    pages = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(doc)]
    doc.close()
    return pages


def extract_meta(pdf_path):
    """Document-level facts that aren't in any single chunk (page count, title)."""
    doc = fitz.open(pdf_path)
    md = doc.metadata or {}
    meta = {
        "page_count": doc.page_count,
        "title": (md.get("title") or "").strip(),
        "author": (md.get("author") or "").strip(),
    }
    doc.close()
    return meta


def _guess_section(line, current):
    m = SECTION_RE.match(line.strip())
    return m.group(1).title() if m else current


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