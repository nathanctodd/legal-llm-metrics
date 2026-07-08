import numpy as np
import pandas as pd
from dataclass import dataclass, field  # native to Python
from enum import Enum                   # native to Python
from datetime import datetime
from citation_accuracy_classes import Citation, CitationScore, SourceType, ExistenceResult, Claim, FidelityResult, PrecisionResult, RecallResult
from citation_accuracy_helpers import extract_citations, resolve_bracket_citations, extract_claims, normalize_text
import re
import difflib
import requests
import json


######################################################################################
#                        CLAIM, CITATION ORCHESTRATOR
######################################################################################

def claim_and_citation_extraction(response_text: str, reference_list: dict[str, str] | None = None) -> list[Claim]:
    """Full pipeline: extract citations, resolve bracket refs, extract claims, link them."""
    
    # orchestrator of above functions call this first

    citations = extract_citations(response_text)
    
    if reference_list:
        citations = resolve_bracket_citations(citations, reference_list)
    
    return extract_claims(response_text, citations)


######################################################################################
#                                REMOVE BELOW LATER
######################################################################################
def _default_llm_judge(claim_text: str, source_text: str) -> tuple[str, float, str]:
    """
    Example judge using the Anthropic API. Truncates source_text to keep
    prompts manageable; swap for chunk-retrieval if sources are long.
    """
    import anthropic

    client = anthropic.Anthropic()
    prompt = f"""You are verifying whether a source supports a claim.

    Claim: {claim_text}

    Source text (may be truncated):
    {source_text[:4000]}

    Respond with exactly one line in this format:
    SUPPORT_LEVEL|CONFIDENCE|RATIONALE

    Where SUPPORT_LEVEL is one of: full, partial, unsupported, contradicted
    CONFIDENCE is a number 0.0-1.0
    RATIONALE is a one-sentence explanation."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    try:
        level, conf, rationale = text.split("|", 2)
        return level.strip().lower(), float(conf.strip()), rationale.strip()
    except ValueError:
        return "unsupported", 0.0, f"judge output unparseable: {text}"

######################################################################################
#                                REMOVE ABOVE LATER
######################################################################################

######################################################################################
#                          INDIVIDUAL SCORE COMPONENTS
######################################################################################

def citation_existence(citation: Citation, timeout: float = 5.0) -> ExistenceResult:

    # sub-function of score_citation

    try:
        if citation.source_type == SourceType.URL:
            resp = requests.get(citation.identifier, timeout=timeout)
            if resp.status_code == 200:
                return ExistenceResult(
                    resolved=True,
                    source_type=citation.source_type,
                    identifier=citation.identifier,
                    source_text=resp.text,
                )
            return ExistenceResult(
                resolved=False,
                source_type=citation.source_type,
                identifier=citation.identifier,
                fetch_error=f"HTTP {resp.status_code}",
            )

        elif citation.source_type == SourceType.DOI:
            # Resolve via doi.org, which redirects to the actual publisher page
            resp = requests.get(f"https://doi.org/{citation.identifier}", timeout=timeout)
            if resp.status_code == 200:
                return ExistenceResult(
                    resolved=True,
                    source_type=citation.source_type,
                    identifier=citation.identifier,
                    source_text=resp.text,
                )
            return ExistenceResult(
                resolved=False,
                source_type=citation.source_type,
                identifier=citation.identifier,
                fetch_error=f"DOI lookup failed: HTTP {resp.status_code}",
            )

        # the removed 'elif' block (retained below in comment block) would go here

        else:
            return ExistenceResult(
                resolved=False,
                source_type=SourceType.UNKNOWN,
                identifier=citation.identifier,
                fetch_error="unrecognized source type",
            )

    except requests.RequestException as e:
        return ExistenceResult(
            resolved=False,
            source_type=citation.source_type,
            identifier=citation.identifier,
            fetch_error=str(e),
        )
    
        '''
        # REMOVED FROM AI-GENERATED FUNCTION, DOES NOT APPEAR TO BE RELEVANT

        elif citation.source_type == SourceType.LOCAL_DOC:
            # Assumes a lookup function/dict mapping doc_id -> text exists elsewhere
            source_text = LOCAL_DOC_STORE.get(citation.identifier)
            if source_text is not None:
                return ExistenceResult(
                    resolved=True,
                    source_type=citation.source_type,
                    identifier=citation.identifier,
                    source_text=source_text,
                )
            return ExistenceResult(
                resolved=False,
                source_type=citation.source_type,
                identifier=citation.identifier,
                fetch_error="doc_id not found in local store",
            )
        '''

def citation_precision(claim: Claim, source_text: str) -> PrecisionResult:

    # sub-function of score_citation

    if judge_fn is None:
        judge_fn = _default_llm_judge # <<<<<<<<<<<<<<<<<<<<< CHANGE THIS FOR IMPLEMENTATION

    support_level, confidence, rationale = judge_fn(claim.text, source_text)

    return PrecisionResult(
        supported=support_level == "full",
        confidence=confidence,
        judge_rationale=rationale,
        support_level=support_level,
    )

def citation_quote_fidelity(claim: Claim, source_text: str, similarity_threshold: float = 0.95) -> FidelityResult:

    # sub-function of score_citation

    if not claim.has_quote or not claim.quoted_text:
        return FidelityResult(matched=True, similarity_score=1.0)  # nothing to check

    quoted = normalize_text(claim.quoted_text)
    source_norm = normalize_text(source_text)

        # Fast path: exact substring match
    if quoted in source_norm:
        return FidelityResult(
            matched=True,
            similarity_score=1.0,
            quoted_text=claim.quoted_text,
            source_excerpt=claim.quoted_text,
        )

    # Slow path: fuzzy match against sliding windows of source text
    best_ratio = 0.0
    best_excerpt = ""
    window_size = len(quoted)
    step = max(1, window_size // 4)

    for i in range(0, max(1, len(source_norm) - window_size), step):
        window = source_norm[i:i + window_size]
        ratio = difflib.SequenceMatcher(None, quoted, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_excerpt = window

    return FidelityResult(
        matched=best_ratio >= similarity_threshold,
        similarity_score=round(best_ratio, 3),
        quoted_text=claim.quoted_text,
        source_excerpt=best_excerpt,
    )

######################################################################################
#                    ORCHESTRATOR FOR SCORING INDIVIDUAL CITATION
######################################################################################

def score_citation(claim: Claim, citation: Citation | None) -> CitationScore:

    # this is called second

    if citation is None:
        return CitationScore(recall_miss=True)

    existence = citation_existence(citation)
    if not existence.resolved:
        return CitationScore(existence=existence, precision=None, fidelity=None)

    fidelity = citation_quote_fidelity(claim, existence.source_text) if claim.has_quote else None
    precision = citation_precision(claim, existence.source_text)

    return CitationScore(existence=existence, precision=precision, fidelity=fidelity)

######################################################################################
#                                CITATION RECALL
######################################################################################

def citation_recall(claims: list[Claim]) -> RecallResult:

    # this is called after score_citation >> loops through the citations for a given response

    needing_citation = [c for c in claims if c.requires_citation]
    with_citation = [c for c in needing_citation if c.citation is not None]

    return RecallResult(
        total_claims_requiring_citation=len(needing_citation),
        claims_with_citation=len(with_citation),
    )