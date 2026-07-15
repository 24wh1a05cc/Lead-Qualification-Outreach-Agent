"""
agent/nodes/classify.py
-----------------------
Deterministic, rule-based classification node.

Decision logic
--------------
There are two independent paths that can produce a result:

  Path A — No-signal override (checked FIRST, before any score comparison):
    If enriched.enrichment_confidence == 0.0 the enrichment tool found NO
    company data at all (e.g. personal email, unknown domain).  Regardless of
    whatever the scoring node computed, this lead MUST be DISQUALIFIED.
    The reason explicitly cites "no company signal" so downstream audits and
    future regression tests can assert on this exact string.

  Path B — Score-threshold classification:
    HOT        score >= HOT_THRESHOLD  (default 70)
    NURTURE    NURTURE_THRESHOLD <= score < HOT_THRESHOLD  (default 40–69)
    DISQUALIFY score < NURTURE_THRESHOLD  (default < 40)

Cited reason format
-------------------
The `reason` field is a human-readable sentence that:
  1. States the tier and the score.
  2. Lists the top matched criteria that drove the decision (for HOT/NURTURE).
  3. Lists the top unmatched/excluded criteria (for DISQUALIFY/NURTURE).

The `cited_signals` field is a machine-readable list of the raw criterion
strings from ICPScore.matched_criteria / ICPScore.unmatched_criteria, so
downstream nodes and evaluators can pattern-match against specific signals
without parsing the free-text reason.

Public API
----------
  classify_lead(enriched, icp_score) -> Classification   ← pure function
  classify_node(state) -> dict                            ← LangGraph wrapper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agent.models import Classification, EnrichedLead, ICPScore, LeadTier

_ICP_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "data" / "icp.json"
)

# How many matched/unmatched criteria to include in the cited reason string.
_MAX_CITED_MATCHED: Final[int] = 4
_MAX_CITED_UNMATCHED: Final[int] = 3

# Sentinel used in the no-signal path — must stay stable across versions
# because test assertions will match on it.
_NO_SIGNAL_REASON_TAG: Final[str] = "no company signal found"


# ---------------------------------------------------------------------------
# Public pure function
# ---------------------------------------------------------------------------

def classify_lead(
    enriched: EnrichedLead,
    icp_score: ICPScore,
) -> Classification:
    """
    Classify a scored lead into a LeadTier with a fully cited reason.

    Parameters
    ----------
    enriched : EnrichedLead
        The enriched lead (used only for enrichment_confidence check).
    icp_score : ICPScore
        The scored lead produced by the scoring node.

    Returns
    -------
    Classification
        Tier + reason + cited_signals + icp_score_used.
    """
    thresholds = _load_thresholds()
    hot_min: float = thresholds["HOT"]["min_score"]
    nurture_min: float = thresholds["NURTURE"]["min_score"]

    # ── Path A: no-signal override ────────────────────────────────────────────
    # enrichment_confidence == 0.0 means the enrichment tool found NO company
    # data.  This check must run BEFORE score comparison so a coincidentally
    # low-but-nonzero score can't masquerade as a real DISQUALIFY.
    if enriched.enrichment_confidence == 0.0:
        return Classification(
            lead_id=icp_score.lead_id,
            tier=LeadTier.DISQUALIFY,
            reason=(
                f"DISQUALIFY (score={icp_score.score:.1f}): {_NO_SIGNAL_REASON_TAG} — "
                "email domain is personal or unknown; cannot verify company, "
                "industry, or buying intent. No outreach resources allocated."
            ),
            cited_signals=[_NO_SIGNAL_REASON_TAG] + icp_score.unmatched_criteria,
            icp_score_used=icp_score.score,
        )

    # ── Path B: score-threshold classification ────────────────────────────────
    score = icp_score.score

    if score >= hot_min:
        tier = LeadTier.HOT
        reason = _build_hot_reason(score, icp_score)
        cited = _top_signals(icp_score.matched_criteria, _MAX_CITED_MATCHED)

    elif score >= nurture_min:
        tier = LeadTier.NURTURE
        reason = _build_nurture_reason(score, hot_min, icp_score)
        cited = (
            _top_signals(icp_score.matched_criteria, _MAX_CITED_MATCHED // 2)
            + _top_signals(icp_score.unmatched_criteria, _MAX_CITED_UNMATCHED)
        )

    else:
        tier = LeadTier.DISQUALIFY
        reason = _build_disqualify_reason(score, nurture_min, icp_score)
        cited = _top_signals(icp_score.unmatched_criteria, _MAX_CITED_UNMATCHED)

    return Classification(
        lead_id=icp_score.lead_id,
        tier=tier,
        reason=reason,
        cited_signals=cited,
        icp_score_used=score,
    )


# ---------------------------------------------------------------------------
# LangGraph node wrapper
# ---------------------------------------------------------------------------

def classify_node(state: dict) -> dict:
    """
    LangGraph node: classify the enriched + scored lead in state.

    Reads ``state['enriched_lead']`` and ``state['icp_score']``.
    Writes ``state['classification']``.
    """
    enriched: EnrichedLead = state["enriched_lead"]
    icp_score: ICPScore = state["icp_score"]
    classification = classify_lead(enriched, icp_score)
    return {**state, "classification": classification}


# ---------------------------------------------------------------------------
# Reason builders — keep logic separate so each path is easy to unit-test
# ---------------------------------------------------------------------------

def _build_hot_reason(score: float, icp_score: ICPScore) -> str:
    top_matched = _top_signals(icp_score.matched_criteria, _MAX_CITED_MATCHED)
    criteria_str = "; ".join(top_matched) if top_matched else "all ICP criteria met"
    return (
        f"HOT (score={score:.1f}/100): strong ICP fit — {criteria_str}."
    )


def _build_nurture_reason(score: float, hot_min: float, icp_score: ICPScore) -> str:
    top_matched = _top_signals(icp_score.matched_criteria, 2)
    top_unmatched = _top_signals(icp_score.unmatched_criteria, _MAX_CITED_UNMATCHED)

    parts: list[str] = []
    if top_matched:
        parts.append("matched: " + "; ".join(top_matched))
    if top_unmatched:
        parts.append("gaps: " + "; ".join(top_unmatched))

    detail = " | ".join(parts) if parts else "partial ICP fit"
    return (
        f"NURTURE (score={score:.1f}/100, below HOT threshold of {hot_min:.0f}): "
        f"moderate fit — {detail}."
    )


def _build_disqualify_reason(score: float, nurture_min: float, icp_score: ICPScore) -> str:
    top_unmatched = _top_signals(icp_score.unmatched_criteria, _MAX_CITED_UNMATCHED)
    gaps_str = "; ".join(top_unmatched) if top_unmatched else "insufficient ICP signal"
    return (
        f"DISQUALIFY (score={score:.1f}/100, below NURTURE threshold of {nurture_min:.0f}): "
        f"poor ICP fit — {gaps_str}."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_thresholds() -> dict[str, Any]:
    """Load scoring thresholds from icp.json."""
    with _ICP_PATH.open(encoding="utf-8") as fh:
        icp = json.load(fh)
    return icp["scoring_thresholds"]


def _top_signals(signals: list[str], n: int) -> list[str]:
    """Return the first n signals, trimmed to avoid overly long reason strings."""
    return signals[:n]
