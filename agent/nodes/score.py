"""
agent/nodes/score.py
--------------------
Deterministic, rule-based ICP scoring node.

Scoring model
-------------
The ICP definition in /data/icp.json assigns a maximum weight to each
criterion category.  This node evaluates every category independently and
awards points on a 0–100 scale.

  Category              Max points   Logic
  ──────────────────────────────────────────────────────────────────────
  company_size          20           Full if preferred band; half if only accepted
  annual_revenue_usd    15           Full if preferred band; half if only accepted
  industry              20           Full (20) for Tier-1; partial (12) for Tier-2;
                                     zero for excluded or unknown
  geography              5           Full if preferred location; half if accepted
  role                  20           Full (20) champion; partial (14) influencer;
                                     partial (8) economic buyer; zero low-value/unknown
  tech_stack            10           Scales with positive hits; capped at max;
                                     negative signals subtract up to half
  buying_signals        10           high_intent ≥ 1 → full; moderate only → half;
                                     low_intent only → zero

  Total possible       100

Confidence penalty
------------------
If enrichment_confidence < 0.5, the raw score is multiplied by
enrichment_confidence so that un-enriched leads can't accidentally score high
on criteria that default to None/empty.

The scoring function is a pure function: score_lead(enriched, icp) -> ICPScore.
The LangGraph node wrapper score_node(state) -> dict handles state I/O.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

from agent.models import EnrichedLead, ICPScore

_ICP_PATH: Final[Path] = Path(__file__).resolve().parent.parent.parent / "data" / "icp.json"

# ---------------------------------------------------------------------------
# Scoring constants — mirror the weights in icp.json
# ---------------------------------------------------------------------------
_W_SIZE: Final[int] = 20
_W_REVENUE: Final[int] = 15
_W_INDUSTRY: Final[int] = 20
_W_GEO: Final[int] = 5
_W_ROLE: Final[int] = 20
_W_TECH: Final[int] = 10
_W_SIGNALS: Final[int] = 10


# ---------------------------------------------------------------------------
# Pure scoring function (easily unit-testable with no side effects)
# ---------------------------------------------------------------------------

def score_lead(enriched: EnrichedLead, icp: dict[str, Any]) -> ICPScore:
    """
    Score an EnrichedLead against the ICP definition.

    Parameters
    ----------
    enriched : EnrichedLead
        The enriched lead record produced by the enrichment node.
    icp : dict
        The parsed contents of /data/icp.json.

    Returns
    -------
    ICPScore
        Numeric score 0–100, with matched/unmatched criteria lists.
    """
    matched: list[str] = []
    unmatched: list[str] = []
    notes: list[str] = []
    raw_score: float = 0.0

    firm = icp.get("firmographic_criteria", {})
    role_cfg = icp.get("role_criteria", {})
    tech_cfg = icp.get("tech_stack_criteria", {})
    signal_cfg = icp.get("buying_signal_keywords", {})

    # ── 1. Company size (max 20) ──────────────────────────────────────────────
    size_cfg = firm.get("company_size", {})
    preferred_sizes = size_cfg.get("preferred_bands", [])
    accepted_sizes = size_cfg.get("accepted_bands", [])
    company_size = enriched.company_size

    if company_size and company_size in preferred_sizes:
        raw_score += _W_SIZE
        matched.append(f"company_size:{company_size} (preferred)")
    elif company_size and company_size in accepted_sizes:
        raw_score += _W_SIZE * 0.5
        matched.append(f"company_size:{company_size} (accepted)")
        notes.append(f"Company size {company_size!r} is accepted but not preferred; half credit.")
    else:
        unmatched.append(
            f"company_size:{company_size or 'unknown'} "
            f"(preferred={preferred_sizes}, accepted={accepted_sizes})"
        )

    # ── 2. Annual revenue (max 15) ────────────────────────────────────────────
    rev_cfg = firm.get("annual_revenue_usd", {})
    preferred_rev = rev_cfg.get("preferred_bands", [])
    accepted_rev = rev_cfg.get("accepted_bands", [])
    revenue = enriched.annual_revenue_usd

    if revenue and revenue in preferred_rev:
        raw_score += _W_REVENUE
        matched.append(f"annual_revenue:{revenue} (preferred)")
    elif revenue and revenue in accepted_rev:
        raw_score += _W_REVENUE * 0.5
        matched.append(f"annual_revenue:{revenue} (accepted)")
        notes.append(f"Revenue {revenue!r} is accepted but not preferred; half credit.")
    else:
        unmatched.append(
            f"annual_revenue:{revenue or 'unknown'} "
            f"(preferred={preferred_rev}, accepted={accepted_rev})"
        )

    # ── 3. Industry (max 20) ──────────────────────────────────────────────────
    ind_cfg = firm.get("industries", {})
    tier1 = ind_cfg.get("tier_1", [])
    tier2 = ind_cfg.get("tier_2", [])
    excluded = ind_cfg.get("excluded", [])
    industry = enriched.industry

    if industry and industry in tier1:
        raw_score += _W_INDUSTRY
        matched.append(f"industry:{industry} (tier_1)")
    elif industry and industry in tier2:
        raw_score += _W_INDUSTRY * 0.6
        matched.append(f"industry:{industry} (tier_2)")
        notes.append(f"Industry {industry!r} is Tier-2; 60% credit.")
    elif industry and industry in excluded:
        unmatched.append(f"industry:{industry} (EXCLUDED)")
        notes.append(f"Industry {industry!r} is explicitly excluded from ICP.")
    else:
        unmatched.append(f"industry:{industry or 'unknown'} (not in target list)")

    # ── 4. Geography (max 5) ──────────────────────────────────────────────────
    geo_cfg = firm.get("geography", {})
    preferred_geo = geo_cfg.get("preferred", [])
    accepted_geo = geo_cfg.get("accepted", [])
    location = enriched.location

    if location and location in preferred_geo:
        raw_score += _W_GEO
        matched.append(f"geography:{location} (preferred)")
    elif location and location in accepted_geo:
        raw_score += _W_GEO * 0.5
        matched.append(f"geography:{location} (accepted)")
    else:
        unmatched.append(f"geography:{location or 'unknown'} (not in target regions)")

    # ── 5. Role (max 20) ──────────────────────────────────────────────────────
    champion_roles = [r.lower() for r in role_cfg.get("champion_roles", [])]
    influencer_roles = [r.lower() for r in role_cfg.get("influencer_roles", [])]
    economic_roles = [r.lower() for r in role_cfg.get("economic_buyer_roles", [])]
    low_value_roles = [r.lower() for r in role_cfg.get("low_value_roles", [])]
    role = (enriched.lead.role or "").lower().strip()

    if _role_matches(role, champion_roles):
        raw_score += _W_ROLE
        matched.append(f"role:{enriched.lead.role} (champion)")
    elif _role_matches(role, influencer_roles):
        raw_score += _W_ROLE * 0.7
        matched.append(f"role:{enriched.lead.role} (influencer)")
        notes.append("Influencer role — not a direct champion but strong influence on purchase.")
    elif _role_matches(role, economic_roles):
        raw_score += _W_ROLE * 0.4
        matched.append(f"role:{enriched.lead.role} (economic_buyer)")
        notes.append("Economic buyer role — controls budget but may need a technical champion.")
    elif _role_matches(role, low_value_roles):
        unmatched.append(f"role:{enriched.lead.role} (low_value — explicitly excluded)")
        notes.append(f"Role {enriched.lead.role!r} is on the low-value role list; zero credit.")
    else:
        unmatched.append(f"role:{enriched.lead.role} (unknown/unrecognised)")

    # ── 6. Tech stack (max 10) ────────────────────────────────────────────────
    positive_tech = [t.lower() for t in tech_cfg.get("positive_signals", [])]
    negative_tech = [t.lower() for t in tech_cfg.get("negative_signals", [])]
    lead_stack = [t.lower() for t in enriched.tech_stack]

    positive_hits = [t for t in lead_stack if t in positive_tech]
    negative_hits = [t for t in lead_stack if t in negative_tech]

    if positive_hits or negative_hits:
        # Score scales from 0 to max based on positive hits (cap at 3 for full score),
        # then is reduced by negative hits.
        tech_score = min(len(positive_hits) / 3, 1.0) * _W_TECH
        tech_score -= len(negative_hits) * (_W_TECH * 0.25)
        tech_score = max(tech_score, 0.0)
        raw_score += tech_score

        if positive_hits:
            matched.append(f"tech_stack:positive_signals={positive_hits}")
        if negative_hits:
            unmatched.append(f"tech_stack:negative_signals={negative_hits}")
            notes.append(f"Negative tech stack signals detected: {negative_hits}.")
    else:
        if enriched.tech_stack:
            unmatched.append(f"tech_stack:no_recognised_signals (stack={enriched.tech_stack})")
        else:
            unmatched.append("tech_stack:unknown (no data)")

    # ── 7. Buying signals (max 10) ────────────────────────────────────────────
    high_intent_kws = [k.lower() for k in signal_cfg.get("high_intent", [])]
    moderate_intent_kws = [k.lower() for k in signal_cfg.get("moderate_intent", [])]
    low_intent_kws = [k.lower() for k in signal_cfg.get("low_intent", [])]

    # buying_signals are labelled strings like "form:high_intent:POC"
    signals_lower = [s.lower() for s in enriched.buying_signals]

    has_high = any(
        any(kw in sig for kw in high_intent_kws) or "high_intent" in sig
        for sig in signals_lower
    )
    has_moderate = any(
        any(kw in sig for kw in moderate_intent_kws) or "moderate_intent" in sig
        for sig in signals_lower
    )
    has_low = any(
        any(kw in sig for kw in low_intent_kws) or "low_intent" in sig
        for sig in signals_lower
    )

    if has_high:
        raw_score += _W_SIGNALS
        matched.append(f"buying_signals:high_intent ({len(enriched.buying_signals)} signals detected)")
    elif has_moderate:
        raw_score += _W_SIGNALS * 0.5
        matched.append(f"buying_signals:moderate_intent ({len(enriched.buying_signals)} signals)")
        notes.append("Only moderate-intent signals present; half credit.")
    elif has_low:
        unmatched.append("buying_signals:low_intent_only (no purchase urgency)")
        notes.append("Only low-intent signals; indicates curiosity, not buying intent.")
    else:
        if enriched.buying_signals:
            # Signals present but none mapped — partial credit
            raw_score += _W_SIGNALS * 0.25
            matched.append(f"buying_signals:unmapped ({enriched.buying_signals})")
        else:
            unmatched.append("buying_signals:none_detected")

    # ── Confidence penalty ────────────────────────────────────────────────────
    confidence = enriched.enrichment_confidence
    if confidence < 0.5:
        penalised = raw_score * confidence
        notes.append(
            f"Confidence penalty applied: raw={raw_score:.1f} × confidence={confidence:.2f} "
            f"→ {penalised:.1f}. Enrichment data is unreliable."
        )
        raw_score = penalised

    final_score = min(round(raw_score, 1), 100.0)

    return ICPScore(
        lead_id=enriched.lead.lead_id,
        score=final_score,
        matched_criteria=matched,
        unmatched_criteria=unmatched,
        scoring_notes=" | ".join(notes) if notes else None,
    )


# ---------------------------------------------------------------------------
# LangGraph node wrapper
# ---------------------------------------------------------------------------

def score_node(state: dict) -> dict:
    """
    LangGraph node: load ICP config and score the enriched lead in state.

    Reads ``state['enriched_lead']``, writes ``state['icp_score']``.
    """
    enriched: EnrichedLead = state["enriched_lead"]
    icp = _load_icp()
    icp_score = score_lead(enriched, icp)
    return {**state, "icp_score": icp_score}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_icp() -> dict[str, Any]:
    """Load and return the parsed ICP configuration."""
    with _ICP_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _role_matches(role_lower: str, role_list: list[str]) -> bool:
    """
    Fuzzy role matching: return True if the lead's role contains or is
    contained by any role in role_list (case-insensitive).
    """
    for ref in role_list:
        if ref in role_lower or role_lower in ref:
            return True
    return False
