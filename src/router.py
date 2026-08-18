"""Query router: decide *how* to answer before we retrieve.

Semantic search matches on topical content, so it fumbles two kinds of query:
  * structural  -- "how many pages", "who wrote this" (metadata, not content)
  * section     -- "what is the abstract" (a request for a named section, whose
                   words don't overlap the section's actual text)

The router spots these and sends them down a direct path instead of top-k search.

    route(query) -> (kind, arg)   kind in {metadata, section, summary, qa}
"""
import re

# Shared with ingest so a query's section name always matches the tag on disk.
from src.sections import canonical

METADATA_RE = re.compile(
    r"\b(how many pages|number of pages|page count|how long is (?:this|the)|"
    r"who (?:wrote|are the authors?|is the author)|author'?s? names?|"
    r"what is the title|what(?:'s| is) (?:the |this )?paper(?:'s)? title|"
    r"when was (?:it|this|this paper|the paper) published|publication (?:date|year)|"
    r"what year was (?:it|this|this paper|the paper) published|"
    r"how many (?:references|citations|papers are cited)|"
    r"number of (?:references|citations))\b",
    re.IGNORECASE,
)

# "what does figure 3 show", "explain table 2.1". Semantic search matches on
# topic, so a figure *number* doesn't pull its caption -- the number is a tiny
# part of the caption's meaning. Route these straight to the caption instead.
FIGURE_RE = re.compile(
    r"\b(figure|fig\.?|table)\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE
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
    return canonical(q)


def route(query):
    if METADATA_RE.search(query):
        return "metadata", None
    section = _named_section(query)
    if section:
        return "section", section
    fig = FIGURE_RE.search(query)
    if fig:
        # ("figure", "1.1") -- kind and number, so the retriever can match the caption
        return "figure", (fig.group(1).rstrip(".").lower(), fig.group(2))
    if SUMMARY_RE.search(query):
        return "summary", None
    return "qa", None
