import numpy as np
import pandas as pd
from dataclass import dataclass, field  # native to Python
from enum import Enum                   # native to Python
from datetime import datetime
from citation_accuracy_classes import Citation, CitationScore, SourceType, ExistenceResult, Claim, FidelityResult, PrecisionResult, RecallResult
import re
import difflib
import requests
import json

# Matches common citation styles: [1], [12], (Smith 2020), URLs, DOIs
CITATION_PATTERNS = {
    "bracket_numeric": re.compile(r"\[(\d+)\]"),
    "url": re.compile(r"https?://[^\s\)\]]+"),
    "doi": re.compile(r"\b10\.\d{4,9}/[^\s\)\]]+"),
}

def extract_citations(response_text: str) -> list[Citation]:
    """
    Find all citation markers in raw response text and build Citation objects.
    Numeric bracket citations like [1] are left unresolved to an identifier
    here — pair them with a reference list separately via resolve_bracket_citations.
    """

    # sub-function of claim_and_citation_extraction()

    citations = []

    for match in CITATION_PATTERNS["url"].finditer(response_text):
        citations.append(Citation(
            raw_text=match.group(0),
            source_type=SourceType.URL,
            identifier=match.group(0),
            span_start=match.start(),
            span_end=match.end(),
        ))

    for match in CITATION_PATTERNS["doi"].finditer(response_text):
        citations.append(Citation(
            raw_text=match.group(0),
            source_type=SourceType.DOI,
            identifier=match.group(0),
            span_start=match.start(),
            span_end=match.end(),
        ))

    for match in CITATION_PATTERNS["bracket_numeric"].finditer(response_text):
        citations.append(Citation(
            raw_text=match.group(0),
            source_type=SourceType.UNKNOWN,   # resolved later against a reference list
            identifier=match.group(1),        # just the number for now, e.g. "1"
            span_start=match.start(),
            span_end=match.end(),
        ))

    return sorted(citations, key=lambda c: c.span_start)

def resolve_bracket_citations(citations: list[Citation], reference_list: dict[str, str]) -> list[Citation]:
    """
    Map numeric bracket citations like [1] to actual identifiers using a
    reference list, e.g. {"1": "https://example.com/paper", "2": "doi:10.1234/xyz"}.
    """

    # sub-function of claim_and_citation_extraction()

    resolved = []
    for c in citations:
        if c.source_type == SourceType.UNKNOWN and c.identifier in reference_list:
            target = reference_list[c.identifier]
            source_type = (
                SourceType.DOI if target.startswith("10.") or "doi.org" in target
                else SourceType.URL if target.startswith("http")
                else SourceType.LOCAL_DOC
            )
            resolved.append(Citation(
                raw_text=c.raw_text,
                source_type=source_type,
                identifier=target,
                span_start=c.span_start,
                span_end=c.span_end,
            ))
        else:
            resolved.append(c)
    return resolved

def extract_claims(response_text: str, citations: list[Citation], extractor_fn=None) -> list[Claim]:
    """
    Segment response text into discrete factual claims, judge whether each
    needs a citation, detect direct quotes, and link the nearest citation
    marker within range.
    """

    # sub-function of claim_and_citation_extraction()

    if extractor_fn is None:
        extractor_fn = default_llm_claim_extractor

    raw_claims = extractor_fn(response_text)  # list of dicts, see below

    claims = []
    for rc in raw_claims:
        span_start = rc["span_start"]
        span_end = rc["span_end"]

        linked_citation = find_nearest_citation(citations, span_end, max_distance=20)

        claims.append(Claim(
            text=rc["text"],
            span_start=span_start,
            span_end=span_end,
            requires_citation=rc["requires_citation"],
            has_quote=rc.get("has_quote", False),
            quoted_text=rc.get("quoted_text"),
            citation=linked_citation,
        ))

    return claims

def find_nearest_citation(citations: list[Citation], claim_end: int, max_distance: int = 20) -> Citation | None:
    """
    A citation is considered 'attached' to a claim if it appears shortly
    after the claim ends, e.g. 'X causes Y [3].' — tune max_distance to
    your citation style (bracket markers sit close; footnote-style may not).
    """

    # sub-function of extract_claims()

    candidates = [
        c for c in citations
        if c.span_start is not None and 0 <= c.span_start - claim_end <= max_distance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.span_start - claim_end)

def default_llm_claim_extractor(response_text: str) -> list[dict]:
    """
    Uses an LLM to segment text into atomic claims and judge citation need.
    Returns raw dicts (not Claim objects) so this function stays swappable
    independent of the dataclass shape.
    """

    # sub-function of extract_claims()

    import anthropic

    client = anthropic.Anthropic()
    prompt = f"""Segment the following text into discrete factual claims.
    For each claim, determine:
    - the exact character span (start, end) in the ORIGINAL text below
    - whether it is the kind of claim that should be backed by a citation
    (specific facts, statistics, quotes, attributed statements — NOT
    opinions, transitions, or general knowledge like "water boils at 100C")
    - whether it contains a direct quote, and if so, the exact quoted substring

    Text:
    \"\"\"
    {response_text}
    \"\"\"

    Respond with ONLY a JSON array, no other text, in this exact format:
    [
    {{
        "text": "...",
        "span_start": 0,
        "span_end": 42,
        "requires_citation": true,
        "has_quote": false,
        "quoted_text": null
    }}
    ]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)  # strip markdown fences if present

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []  # fail closed: treat as no extractable claims rather than crashing
    

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise for fair comparison.
        helper method for citation_quote_fidelity()
    """

    # sub-function of citation_quote_fidelity()
    
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”\"']", "", text)
    return text.strip()