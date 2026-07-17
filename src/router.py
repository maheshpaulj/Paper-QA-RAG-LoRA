"""Query router: decide *how* to answer before we retrieve.

Semantic search matches on topical content, so it fumbles two kinds of query:
  * structural  -- "how many pages", "who wrote this" (metadata, not content)
  * section     -- "what is the abstract" (a request for a named section, whose
                   words don't overlap the section's actual text)

The router spots these and sends them down a direct path instead of top-k search.

    route(query) -> (kind, arg)   kind in {metadata, section, summary, qa}
"""
import re

# Canonical section labels match how ingest.py Title-cases them.
SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "intro": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Approach",
    "experiment": "Experiment",
    "experiments": "Experiment",
    "result": "Result",
    "results": "Result",
    "evaluation": "Evaluation",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
}

METADATA_RE = re.compile(
    r"\b(how many pages|number of pages|page count|how long is (?:this|the)|"
    r"who (?:wrote|are the authors?|is the author)|author'?s? names?|"
    r"what is the title|when was (?:it|this) published)\b",
    re.IGNORECASE,
)

SUMMARY_RE = re.compile(
    r"\b(summar(?:y|ize|ise|isation|ization)|tl;?dr|overview|"
    r"what(?:'s| is) this paper about|"
    r"key (?:points|findings|contributions|takeaways)|"
    r"main (?:points|idea|contribution|findings))\b",
    re.IGNORECASE,
)

# Strip common request wrappers; if what's left is exactly a section name, it's
# a section request. Strict residue-matching keeps normal questions out.
_WRAP_RE = re.compile(
    r"^(?:what(?:'s| is| are| does| do)?|show me|give me|tell me|display|read|"
    r"summar(?:ize|ise)|please|can you)\s+",
    re.IGNORECASE,
)
_OF_PAPER_RE = re.compile(
    r"\b(?:of|in|from)\s+(?:this|the)\s+(?:paper|document|pdf|study)\b", re.IGNORECASE
)
_FILLER_RE = re.compile(
    r"\b(?:the|this|a|section|say|says|state|states|cover|covers|discuss|"
    r"discusses|about|contain|contains|content|part)\b",
    re.IGNORECASE,
)


def _named_section(query):
    q = query.lower().strip().rstrip("?.!")
    q = _WRAP_RE.sub("", q)
    q = _OF_PAPER_RE.sub("", q)
    q = _FILLER_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    return SECTION_ALIASES.get(q)


def route(query):
    if METADATA_RE.search(query):
        return "metadata", None
    section = _named_section(query)
    if section:
        return "section", section
    if SUMMARY_RE.search(query):
        return "summary", None
    return "qa", None
