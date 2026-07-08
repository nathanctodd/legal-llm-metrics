import numpy as np
import pandas as pd
from dataclass import dataclass, field  # native to Python
from enum import Enum                   # native to Python
from datetime import datetime

class SourceType(Enum):
    URL = 'url'
    DOI = 'doi'
    LOCAL_DOC = 'local_doc'
    UNKNOWN = 'unknown'

@dataclass
class Citation():
    raw_text: str                     # citation as it appeared in model output, e.g. "[1]" or a URL
    source_type: SourceType
    identifier: str                   # the URL, DOI, or doc_id to resolve
    span_start: int | None = None     # char offset in response where citation appears
    span_end: int | None = None

@dataclass
class ExistenceResult:
    resolved: bool
    source_type: SourceType
    identifier: str
    source_text: str | None = None    # fetched content, if resolved
    fetch_error: str | None = None    # reason it failed, if not resolved
    resolved_at: datetime = field(default_factory=datetime.timezone.utc)

@dataclass
class Claim:
    text: str                         # the claim/assertion as extracted from model output
    span_start: int
    span_end: int
    requires_citation: bool           # whether this claim type needed a source at all
    has_quote: bool = False           # whether it contains a direct/verbatim quote
    quoted_text: str | None = None    # the quoted substring, if has_quote
    citation: Citation | None = None  # linked citation, if any was given

@dataclass
class FidelityResult:
    matched: bool
    similarity_score: float | None = None   # e.g. edit distance or fuzzy match ratio
    quoted_text: str = ""
    source_excerpt: str = ""

@dataclass
class PrecisionResult:
    supported: bool
    confidence: float                 # 0-1, from NLI model or judge
    judge_rationale: str | None = None
    support_level: str = "unsupported"  # "full" | "partial" | "unsupported" | "contradicted"

@dataclass
class RecallResult:
    total_claims_requiring_citation: int
    claims_with_citation: int

    @property
    def recall(self) -> float:
        if self.total_claims_requiring_citation == 0:
            return 1.0
        return self.claims_with_citation / self.total_claims_requiring_citation

@dataclass
class CitationScore:
    claim: Claim
    existence: ExistenceResult | None = None
    precision: PrecisionResult | None = None
    fidelity: FidelityResult | None = None
    recall_miss: bool = False         # True if no citation was given at all

    @property
    def passed(self) -> bool:
        if self.recall_miss:
            return False
        if self.existence and not self.existence.resolved:
            return False
        if self.precision and not self.precision.supported:
            return False
        if self.fidelity and not self.fidelity.matched:
            return False
        return True