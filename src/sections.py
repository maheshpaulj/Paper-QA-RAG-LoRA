"""Canonical section names -- the single source of truth for tagging AND routing.

This exists because the two used to keep separate lists and drifted apart:

  * ingest tagged chunks with the *literal* header text, so a paper writing
    "Results" and another writing "Result" produced two different sections. The
    router looked up "Result" and silently missed every chunk tagged "Results".
  * "Limitations" was in neither list, so a paper with a Limitations section had
    those chunks absorbed into whatever section came before, and the question
    "what are the limitations" fell through to plain semantic search.

Add a section here once and both the tagger and the router pick it up.
"""
import re

# variant (lowercase) -> canonical name
SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "intro": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "prior work": "Related Work",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "experiment": "Experiments",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "evaluation": "Evaluation",
    "result": "Results",
    "results": "Results",
    "discussion": "Discussion",
    "limitation": "Limitations",
    "limitations": "Limitations",
    "future work": "Future Work",
    "broader impact": "Broader Impacts",
    "broader impacts": "Broader Impacts",
    "ethical considerations": "Ethics",
    "ethics statement": "Ethics",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "acknowledgment": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "acknowledgement": "Acknowledgements",
    "acknowledgements": "Acknowledgements",
    "references": "References",
    "bibliography": "References",
    "appendix": "Appendix",
}

# Longest-first so "related work" wins over "work", "limitations" over "limitation".
_ALTERNATION = "|".join(
    re.escape(k) for k in sorted(SECTION_ALIASES, key=len, reverse=True)
)

# A header line: optional numbering then a known section name. Numbering is
# either arabic ("5.", "5.2") or roman ("VIII."), because IEEE-style papers
# number sections "VIII. LIMITATIONS" -- unmatched, every chunk in such a paper
# inherited the previous section and section routing silently did nothing.
#
# The roman part is deliberately case-SENSITIVE via (?-i:...): matched
# case-insensitively, [IVXLC]+ also matches ordinary words, so "Civil results
# were mixed" would parse as numeral "Civil" + section "results" and mistag the
# chunk. Real headers are uppercase, so requiring that costs nothing.
SECTION_RE = re.compile(
    rf"^\s*(?:(?:\d+(?:\.\d+)*|(?-i:[IVXLC]{{1,7}}))\.?\s+)?({_ALTERNATION})\b",
    re.IGNORECASE,
)


def canonical(name):
    """Map any known variant to its canonical name, else None."""
    return SECTION_ALIASES.get((name or "").strip().lower())


# Inline headers: "Abstract—For people who ..." / "Abstract: ..." (IEEE style).
_INLINE_SEP = ("—", "–", ":", "-")
MAX_HEADER_LEN = 60


def _looks_like_header(remainder, whole):
    """True if what follows the section name is header-ish, not prose.

    Matching the name alone is not enough: an ordinary sentence beginning with
    a section word passes that test. In an IEEE paper the line
    "conclusions are drawn." sat right under "VIII. LIMITATIONS" and was read as
    a Conclusion header, overwriting Limitations -- so the section existed, was
    detected, and was then immediately clobbered. Every chunk in it ended up
    tagged Conclusion and "what are the limitations" retrieved the wrong text.
    """
    rest = remainder.strip()
    if not rest:
        return True                                  # "VIII. LIMITATIONS"
    if rest[0] in _INLINE_SEP:
        return True                                  # "Abstract—For people ..."
    if len(whole) > MAX_HEADER_LEN:
        return False                                 # too long to be a heading
    # A heading's trailing words are capitalised ("CONCLUSION AND FUTURE WORK");
    # prose has lowercase ones ("conclusions are drawn").
    words = [w for w in re.split(r"\W+", rest) if w]
    return all(w[0].isupper() for w in words)


def match_header(line):
    """Canonical section name if `line` is that section's header, else None."""
    line = line.strip()
    m = SECTION_RE.match(line)
    if not m:
        return None
    if not _looks_like_header(line[m.end():], line):
        return None
    return canonical(m.group(1))
