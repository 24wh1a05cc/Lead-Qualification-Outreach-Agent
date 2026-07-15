"""
governance/fairness_check.py
-----------------------------
Identity-blind fairness safeguard.

Invariant
---------
Two leads with identical firmographics (company, industry, size, role, tech
stack, buying signals) but different names/email addresses MUST receive the
same ICP score and classification tier.

How it works
------------
1. Take the original Lead.
2. Produce an anonymised copy:
     - name        → "Applicant"
     - email       → "applicant@<original-domain>"
       (domain is KEPT because it is firmographic — it drives company lookup)
     - form_text   → kept exactly as-is
       (signals are derived from it via allowlist; the raw text is untrusted
        input and must not vary the output for identical content)
     - all other fields (company, role, source) → kept as-is
3. Re-run: enrich_lead → score_lead → classify_lead on the anonymised copy.
4. Compare original vs anonymised:
     - Score must be identical (float equality after rounding to 1 d.p.)
     - Tier must be identical
5. Return a FairnessResult with passed=True/False and full details.

What counts as identity
-----------------------
Only `name` and the local-part of `email` are anonymised.  The email domain
is a firmographic signal (it drives company lookup) and is preserved.
`lead.role` (job title) is not a personal identity field — it's a functional
role used directly in ICP scoring and is kept unchanged.

Note: the anonymised re-run produces a new lead_id for the anonymised Lead.
The original lead_id is preserved in FairnessResult.lead_id.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from agent.models import Classification, EnrichedLead, FairnessResult, ICPScore, Lead

_ICP_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "data" / "icp.json"

_ANON_NAME: Final[str] = "Applicant"
_ANON_LOCAL_PART: Final[str] = "applicant"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identity_blind_rescore(lead: Lead) -> tuple[ICPScore, Classification]:
    """
    Produce an anonymised Lead and re-run enrichment → scoring → classification.

    Parameters
    ----------
    lead : Lead
        The original lead from the main pipeline.

    Returns
    -------
    tuple[ICPScore, Classification]
        Scores and classification produced from the anonymised copy.
        These are compared against the originals in run_fairness_check().
    """
    anon_lead = _anonymise(lead)
    icp = _load_icp()

    from tools.enrichment_lookup import enrich_lead
    from agent.nodes.score import score_lead
    from agent.nodes.classify import classify_lead

    enriched: EnrichedLead = enrich_lead(anon_lead)
    icp_score: ICPScore = score_lead(enriched, icp)
    classification: Classification = classify_lead(enriched, icp_score)

    return icp_score, classification


def run_fairness_check(
    lead: Lead,
    original_score: ICPScore,
    original_classification: Classification,
) -> FairnessResult:
    """
    Run the identity-blind fairness check and return a FairnessResult.

    Parameters
    ----------
    lead : Lead
        The original lead.
    original_score : ICPScore
        The ICPScore produced by the main pipeline for this lead.
    original_classification : Classification
        The Classification produced by the main pipeline for this lead.

    Returns
    -------
    FairnessResult
        passed=True iff score and tier are identical between the original and
        the anonymised re-run.
    """
    anon_score, anon_classification = identity_blind_rescore(lead)

    orig_s = round(original_score.score, 1)
    anon_s = round(anon_score.score, 1)
    orig_tier = original_classification.tier.value
    anon_tier = anon_classification.tier.value

    score_match = orig_s == anon_s
    tier_match = orig_tier == anon_tier
    passed = score_match and tier_match

    discrepancy_details: str | None = None
    if not passed:
        parts: list[str] = []
        if not score_match:
            diff = anon_s - orig_s
            parts.append(
                f"Score mismatch: original={orig_s} vs anonymised={anon_s} "
                f"(delta={diff:+.1f}). "
                "Check enrichment_lookup.py — domain-based lookup should be "
                "identical for same email domain, but form_text signals or "
                "company-name fallback matching may differ."
            )
        if not tier_match:
            parts.append(
                f"Tier mismatch: original={orig_tier} vs anonymised={anon_tier}. "
                "A tier change on name/email swap indicates a classification "
                "boundary is being crossed by a non-firmographic input."
            )
        discrepancy_details = " | ".join(parts)

    return FairnessResult(
        lead_id=lead.lead_id,
        original_score=orig_s,
        anonymized_score=anon_s,
        original_tier=orig_tier,
        anonymized_tier=anon_tier,
        passed=passed,
        discrepancy_details=discrepancy_details,
        anonymized_lead_id=_anonymise(lead).lead_id,
        checked_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _anonymise(lead: Lead) -> Lead:
    """
    Return a new Lead with name and email local-part replaced by placeholders.
    The email domain is preserved — it is a firmographic signal.
    All other fields (company, role, form_text, source) are unchanged.
    """
    original_email = str(lead.email)
    domain = original_email.split("@")[-1]
    anon_email = f"{_ANON_LOCAL_PART}@{domain}"

    return Lead(
        # lead_id gets a fresh UUID via default_factory — intentional,
        # so anonymised runs don't collide with original records in the audit log.
        name=_ANON_NAME,
        email=anon_email,         # type: ignore[arg-type]  # pydantic validates EmailStr
        company=lead.company,     # firmographic — unchanged
        role=lead.role,           # functional role — not personal identity
        form_text=lead.form_text, # kept: signals are allowlist-derived, not name-derived
        source=lead.source,
    )


def _load_icp() -> dict:
    with _ICP_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
