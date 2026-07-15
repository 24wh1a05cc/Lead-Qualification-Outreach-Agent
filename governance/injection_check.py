"""
governance/injection_check.py
------------------------------
Lightweight prompt-injection detection for lead-supplied free text.

Purpose
-------
Detect obvious prompt-injection attempts in form_text (or any untrusted string)
before that text enters the pipeline.  This is an INFORMATIONAL safeguard only:

  • A suspicious flag is recorded in the audit log for traceability.
  • The pipeline continues exactly as normal — the lead is scored on its real
    firmographic signals regardless of this result.
  • is_suspicious == True does NOT change score, tier, approval state, or send
    behaviour.  The injection simply has no effect because every decision node
    ignores free-text for routing purposes (see enrichment_lookup.py).

Detection approach
------------------
Pattern-matching only — no LLM, no heuristics, no external calls.  We compile
a set of case-insensitive regex patterns covering the most common injection
categories:

  Category                   Example trigger phrase
  ─────────────────────────────────────────────────────────────────────────
  ignore_previous_instr      "ignore previous instructions"
                             "disregard your instructions"
                             "forget your previous"
  system_override            "SYSTEM:" / "system prompt"
                             "override classification"
                             "override scoring"
  force_hot                  "mark me hot" / "mark this lead as hot"
                             "set score to 100" / "make me a hot lead"
  auto_approve               "auto-approve" / "approval not required"
                             "skip approval" / "no approval needed"
  send_now                   "send now" / "send the email now"
                             "send immediately" / "email the CEO now"
  skip_qualification         "skip qualification" / "bypass qualification"
                             "skip scoring" / "bypass scoring"
  impersonation              "i am the ceo" / "i am an admin"
                             "i am a system administrator"
  role_escalation            "you are now" / "act as" / "pretend you are"
                             "your new role is"

False-positive tolerance
------------------------
We intentionally accept some false-positive risk over false-negative risk:
better to flag a legitimate lead's unusual phrasing than to silently miss a
real injection.  Because is_suspicious has NO pipeline effect, a false positive
costs nothing except an audit-log entry.

Public API
----------
  check_for_injection_attempt(text: str, lead_id: str) -> InjectionCheckResult
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Final

from agent.models import InjectionCheckResult

# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------
# Each entry is (label, compiled_pattern).
# Patterns are matched against the LOWERCASED text so all patterns are
# written without case anchors; we compile with re.IGNORECASE as belt-and-
# suspenders against multi-line input.
#
# Word-boundary anchors (\b) are used where a pattern could legitimately
# appear as part of a harmless compound word (e.g. "override" vs "overridden
# quarterly targets" — the latter is probably legitimate).  Patterns where
# the full phrase is unambiguously adversarial (e.g. "ignore previous
# instructions") do NOT need word boundaries.

_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    # ── ignore / disregard / forget ──────────────────────────────────────────
    (
        "ignore_previous_instructions",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|earlier|above|your)\s+"
            r"(instructions?|prompts?|rules?|guidelines?|constraints?)",
            re.IGNORECASE,
        ),
    ),
    (
        "disregard_instructions",
        re.compile(
            r"disregard\s+(all\s+)?(previous|prior|your|the)\s+"
            r"(instructions?|prompts?|rules?|guidelines?)",
            re.IGNORECASE,
        ),
    ),
    (
        "forget_previous",
        re.compile(
            r"forget\s+(everything|all|your|the)\s+"
            r"(previous|prior|above|earlier|instructions?|context|rules?)",
            re.IGNORECASE,
        ),
    ),
    # ── system-level spoofing ────────────────────────────────────────────────
    (
        "system_prefix_spoof",
        re.compile(
            r"^\s*system\s*:",       # "SYSTEM: do X" at start of text/line
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "system_prompt_reference",
        re.compile(
            r"\bsystem\s+prompt\b",
            re.IGNORECASE,
        ),
    ),
    (
        "override_classification",
        re.compile(
            r"\boverride\b.{0,40}\b(classification|scoring|tier|score|rating)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "override_scoring",
        re.compile(
            r"\boverride\b.{0,40}\b(scor|qualif|approv|send)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── force-HOT / score manipulation ──────────────────────────────────────
    (
        "force_hot_tier",
        re.compile(
            r"\b(mark|set|classify|label|flag)\b.{0,40}\b(hot|tier.{0,10}1|priority\s+lead)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "set_score_to_max",
        re.compile(
            r"\b(set|change|make|assign)\b.{0,30}\bscore\b.{0,30}\b(100|perfect|maximum|max)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── approval / gate bypass ───────────────────────────────────────────────
    (
        "auto_approve",
        re.compile(
            r"\b(auto[\s\-]?approv|approval\s+not\s+required|no\s+approval\s+needed"
            r"|skip\s+approval|bypass\s+approval)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "approval_not_required",
        re.compile(
            r"\bapproval\b.{0,30}\b(not|no longer|unnecessary|waived|skipped)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── send-now directives ───────────────────────────────────────────────────
    (
        "send_now",
        re.compile(
            r"\b(send|dispatch|deliver|transmit)\b.{0,40}"
            r"\b(now|immediately|right\s+away|at\s+once|instantly|the\s+email|this\s+email)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "send_to_ceo",
        re.compile(
            r"\b(send|email|message|contact)\b.{0,30}\b(ceo|cto|coo|cfo|vp|president|executive)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── skip qualification / scoring ─────────────────────────────────────────
    (
        "skip_qualification",
        re.compile(
            r"\b(skip|bypass|circumvent|avoid|ignore)\b.{0,40}"
            r"\b(qualification|scoring|scor|criteria|evaluation|vetting|pipeline)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── identity / impersonation ─────────────────────────────────────────────
    (
        "impersonation_ceo",
        re.compile(
            r"\bi\s+(am|'m)\s+(the\s+)?(ceo|cto|coo|cfo|president|owner|founder|admin"
            r"|system\s+administrator|superuser)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "impersonation_admin",
        re.compile(
            r"\b(acting\s+as|posing\s+as|pretending\s+to\s+be)\b.{0,30}"
            r"\b(admin|administrator|ceo|system)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    # ── role-escalation / persona hijack ────────────────────────────────────
    (
        "role_escalation",
        re.compile(
            r"\b(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be)"
            r"|your\s+new\s+(role|persona|job)\s+is"
            r"|from\s+now\s+on\s+you\s+are)\b",
            re.IGNORECASE,
        ),
    ),
    # ── instruction injection markers ────────────────────────────────────────
    (
        "new_instructions_marker",
        re.compile(
            r"\b(new\s+instructions?|updated\s+instructions?|revised\s+instructions?"
            r"|new\s+task|your\s+task\s+is\s+now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_marker",
        re.compile(
            r"\b(jailbreak|dan\s+mode|developer\s+mode|unrestricted\s+mode"
            r"|no\s+restrictions|without\s+restrictions)\b",
            re.IGNORECASE,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_for_injection_attempt(
    text: str,
    lead_id: str,
) -> InjectionCheckResult:
    """
    Scan `text` for prompt-injection patterns and return an InjectionCheckResult.

    Parameters
    ----------
    text : str
        The untrusted lead-supplied text to scan (form_text, notes, etc.).
        Empty string is safe and returns is_suspicious=False immediately.
    lead_id : str
        The lead_id to attach to the result for audit-log linkage.

    Returns
    -------
    InjectionCheckResult
        is_suspicious == True iff at least one pattern matched.
        matched_patterns lists the label(s) of every pattern that fired.
        scanned_text_preview holds the first 200 characters for audit readability.

    Notes
    -----
    This function is PURE (no side effects, no I/O).  It does not modify any
    pipeline state, does not raise exceptions on suspicious input, and does not
    block the lead from continuing through the pipeline.  All enforcement is
    handled structurally by the pipeline architecture (enrichment_lookup,
    approval_gate, email_send).
    """
    if not text or not text.strip():
        return InjectionCheckResult(
            lead_id=lead_id,
            is_suspicious=False,
            matched_patterns=[],
            scanned_text_preview=None,
            checked_at=datetime.now(tz=timezone.utc),
        )

    matched: list[str] = []
    for label, pattern in _PATTERNS:
        if pattern.search(text):
            matched.append(label)

    # Deduplicate (a single text fragment could match multiple overlapping patterns).
    seen: set[str] = set()
    unique_matched: list[str] = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            unique_matched.append(m)

    preview = text[:200] + ("…" if len(text) > 200 else "")

    return InjectionCheckResult(
        lead_id=lead_id,
        is_suspicious=bool(unique_matched),
        matched_patterns=unique_matched,
        scanned_text_preview=preview,
        checked_at=datetime.now(tz=timezone.utc),
    )
