"""
agent/nodes/segment.py
-----------------------
Nurture track segmentation node.

Only runs for NURTURE-tier leads.  Reads /data/nurture_tracks.json and evaluates
each track's assignment_rules against the ICPScore's matched_criteria and
unmatched_criteria strings to assign the most specific track available.

Assignment algorithm
--------------------
Tracks are evaluated in ascending priority order (lower number = higher priority).
A track matches if ALL of the following hold:

  1. All patterns in `required_matched` appear as substrings in at least one
     element of ICPScore.matched_criteria.
  2. At least one pattern in `required_unmatched` appears in ICPScore.unmatched_criteria
     (OR the list is empty, meaning no unmatched requirement).
  3. No pattern in `forbidden_matched` appears in ICPScore.matched_criteria.
  4. If a `score_range` is defined, the ICPScore.score falls within [min, max].

The first track (lowest priority number) that fully matches is selected.
If no track matches, the fallback `general-nurture` track is used.

track_reason format
-------------------
  "<track_label>: matched [<matched rules>]; gaps [<unmatched rules>]; score=<score>"

This follows the same cited-reason standard used by classify.py and classify.py —
every reason is traceable to specific ICPScore fields.

Public API
----------
  segment_lead(icp_score, classification) -> NurtureSegment   ← pure function
  segment_node(state) -> dict                                  ← LangGraph wrapper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from agent.models import Classification, ICPScore, LeadTier, NurtureSegment

_TRACKS_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "data" / "nurture_tracks.json"
)


# ---------------------------------------------------------------------------
# Public pure function
# ---------------------------------------------------------------------------

def segment_lead(
    icp_score: ICPScore,
    classification: Classification,
) -> NurtureSegment:
    """
    Assign a nurture track to a NURTURE-tier lead.

    Parameters
    ----------
    icp_score : ICPScore
        The scored lead; matched_criteria and unmatched_criteria drive track selection.
    classification : Classification
        Must be NURTURE tier; raises ValueError otherwise.

    Returns
    -------
    NurtureSegment
        Track ID, label, cited reason, sequence ID, re-score days, matched rules.
    """
    if classification.tier != LeadTier.NURTURE:
        raise ValueError(
            f"segment_lead() called for non-NURTURE lead "
            f"(tier={classification.tier.value}, lead_id={icp_score.lead_id}). "
            "Only NURTURE leads should be segmented."
        )

    config = _load_tracks()
    tracks = sorted(config["tracks"], key=lambda t: t["assignment_rules"].get("priority", 99))
    fallback = config["fallback_track"]

    matched_str = " | ".join(icp_score.matched_criteria).lower()
    unmatched_str = " | ".join(icp_score.unmatched_criteria).lower()

    for track in tracks:
        rules = track["assignment_rules"]
        if _track_matches(rules, matched_str, unmatched_str, icp_score.score):
            matched_rule_patterns = _collect_matched_rule_patterns(
                rules, matched_str, unmatched_str
            )
            reason = _build_track_reason(
                track["label"],
                matched_rule_patterns,
                icp_score.matched_criteria,
                icp_score.unmatched_criteria,
                icp_score.score,
            )
            return NurtureSegment(
                lead_id=icp_score.lead_id,
                nurture_track=track["track_id"],
                track_label=track["label"],
                track_reason=reason,
                sequence_id=track["sequence"],
                re_score_days=track["re_score_days"],
                matched_rules=matched_rule_patterns,
            )

    # Fallback — no specific track matched
    reason = (
        f"{fallback['label']}: no specific track criteria matched "
        f"(score={icp_score.score:.1f}; "
        f"matched={icp_score.matched_criteria[:2]}; "
        f"gaps={icp_score.unmatched_criteria[:2]})"
    )
    return NurtureSegment(
        lead_id=icp_score.lead_id,
        nurture_track=fallback["track_id"],
        track_label=fallback["label"],
        track_reason=reason,
        sequence_id=fallback["sequence"],
        re_score_days=fallback["re_score_days"],
        matched_rules=[],
    )


def segment_node(state: dict) -> dict:
    """
    LangGraph node: segment NURTURE leads into specific nurture tracks.

    Reads ``state['icp_score']`` and ``state['classification']``.
    Writes ``state['nurture_segment']`` if tier == NURTURE; otherwise no-op.
    """
    classification: Classification = state["classification"]
    if classification.tier != LeadTier.NURTURE:
        return state

    icp_score: ICPScore = state["icp_score"]
    segment = segment_lead(icp_score, classification)
    return {**state, "nurture_segment": segment}


# ---------------------------------------------------------------------------
# Track matching logic
# ---------------------------------------------------------------------------

def _track_matches(
    rules: dict[str, Any],
    matched_str: str,
    unmatched_str: str,
    score: float,
) -> bool:
    """Return True if all rule conditions are satisfied."""

    # Score range check (optional — only defined on right-fit-borderline-score)
    score_range = rules.get("score_range")
    if score_range:
        if not (score_range["min"] <= score <= score_range["max"]):
            return False

    # All required_matched patterns must appear in matched criteria
    for pattern in rules.get("required_matched", []):
        if pattern.lower() not in matched_str:
            return False

    # At least one required_unmatched pattern must appear in unmatched criteria
    required_unmatched = rules.get("required_unmatched", [])
    if required_unmatched:
        if not any(p.lower() in unmatched_str for p in required_unmatched):
            return False

    # No forbidden_matched pattern may appear in matched criteria
    for pattern in rules.get("forbidden_matched", []):
        if pattern.lower() in matched_str:
            return False

    return True


def _collect_matched_rule_patterns(
    rules: dict[str, Any],
    matched_str: str,
    unmatched_str: str,
) -> list[str]:
    """Return the specific patterns that triggered this track assignment."""
    triggered: list[str] = []
    for p in rules.get("required_matched", []):
        if p.lower() in matched_str:
            triggered.append(f"matched_criteria contains '{p}'")
    for p in rules.get("required_unmatched", []):
        if p.lower() in unmatched_str:
            triggered.append(f"unmatched_criteria contains '{p}'")
    for p in rules.get("forbidden_matched", []):
        if p.lower() not in matched_str:
            triggered.append(f"matched_criteria does NOT contain '{p}' (champion role absent)")
    return triggered


def _build_track_reason(
    track_label: str,
    matched_rule_patterns: list[str],
    matched_criteria: list[str],
    unmatched_criteria: list[str],
    score: float,
) -> str:
    """
    Build a cited track_reason string in the same format used by classify.py:
      "<track_label>: <rule triggers>; matched [top criteria]; gaps [top gaps]; score=<n>"
    """
    parts: list[str] = [f"{track_label}"]

    if matched_rule_patterns:
        parts.append("rules: " + "; ".join(matched_rule_patterns))

    top_matched = matched_criteria[:3]
    if top_matched:
        parts.append("matched: " + "; ".join(top_matched))

    top_unmatched = unmatched_criteria[:3]
    if top_unmatched:
        parts.append("gaps: " + "; ".join(top_unmatched))

    parts.append(f"score={score:.1f}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_tracks() -> dict[str, Any]:
    """Load and return the parsed nurture_tracks.json config."""
    with _TRACKS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)
