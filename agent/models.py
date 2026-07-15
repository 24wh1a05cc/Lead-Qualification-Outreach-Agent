"""
agent/models.py
---------------
Core Pydantic models shared across every node in the LangGraph pipeline.

Data flows in one direction:
  Lead  →  EnrichedLead  →  ICPScore  →  Classification  →  DraftedEmail

AuditRecord captures the full picture for every pipeline run.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# 1. Raw inbound lead (what arrives from a web form, CSV import, or CRM hook)
# ---------------------------------------------------------------------------

class Lead(BaseModel):
    """Represents the raw, un-enriched lead as captured from the inbound source."""

    lead_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this lead across the entire pipeline.",
    )
    name: str = Field(..., description="Full name of the prospect.")
    email: EmailStr = Field(..., description="Work email address.")
    company: str = Field(..., description="Company or organisation name.")
    role: str = Field(..., description="Job title or role as self-reported.")
    # Free-text captured from an inbound form, chat transcript, or SDR notes.
    form_text: str | None = Field(
        default=None,
        description="Raw form submission or freeform notes from the lead.",
    )
    source: str = Field(
        ...,
        description="Acquisition channel, e.g. 'webinar', 'inbound_form', 'linkedin'.",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the lead entered the system.",
    )


# ---------------------------------------------------------------------------
# 2. Enriched lead (Lead + data returned by the enrichment tool)
# ---------------------------------------------------------------------------

class EnrichedLead(BaseModel):
    """Lead record augmented with firmographic and behavioural enrichment data."""

    # Carry the original lead forward so downstream nodes have full context.
    lead: Lead

    # Firmographic fields populated by the enrichment tool.
    company_size: str | None = Field(
        default=None,
        description="Employee head-count band, e.g. '51-200', '201-1000'.",
    )
    industry: str | None = Field(
        default=None,
        description="Normalised industry label, e.g. 'SaaS', 'FinTech', 'Healthcare IT'.",
    )
    annual_revenue_usd: str | None = Field(
        default=None,
        description="Revenue band, e.g. '$10M-$50M'.",
    )
    location: str | None = Field(
        default=None,
        description="Primary operating geography, e.g. 'US', 'EU', 'APAC'.",
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Known technologies in use (CRM, data warehouse, etc.).",
    )

    # Behavioural / intent signals detected during enrichment.
    buying_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Observed buying-intent signals, e.g. 'visited pricing page', "
            "'downloaded ROI calculator', 'job post for data engineer'."
        ),
    )

    # Quality indicator so the scoring node can down-weight poor enrichment.
    enrichment_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in enrichment accuracy, 0.0 (none) to 1.0 (high).",
    )
    enrichment_source: str | None = Field(
        default=None,
        description="Which enrichment provider or mock fixture supplied the data.",
    )


# ---------------------------------------------------------------------------
# 3. ICP score (output of the scoring node)
# ---------------------------------------------------------------------------

class ICPScore(BaseModel):
    """Quantified fit of an enriched lead against the Ideal Customer Profile."""

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Aggregate fit score from 0 (no fit) to 100 (perfect fit).",
    )
    matched_criteria: list[str] = Field(
        default_factory=list,
        description="ICP criteria that the lead satisfied.",
    )
    unmatched_criteria: list[str] = Field(
        default_factory=list,
        description="ICP criteria that the lead did not satisfy.",
    )
    scoring_notes: str | None = Field(
        default=None,
        description="Free-text rationale from the scoring node.",
    )


# ---------------------------------------------------------------------------
# 4. Classification (output of the classification node)
# ---------------------------------------------------------------------------

class LeadTier(str, Enum):
    """Three-tier routing classification for leads."""
    HOT = "HOT"            # High fit → immediate SDR outreach
    NURTURE = "NURTURE"    # Moderate fit → marketing nurture sequence
    DISQUALIFY = "DISQUALIFY"  # Poor fit → close / do not contact


class Classification(BaseModel):
    """Tier assignment and the evidence that drove it."""

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    tier: LeadTier = Field(..., description="Routing tier assigned to this lead.")
    reason: str = Field(
        ...,
        description="One-sentence explanation of why this tier was assigned.",
    )
    cited_signals: list[str] = Field(
        default_factory=list,
        description="Specific signals from EnrichedLead that drove the decision.",
    )
    icp_score_used: float = Field(
        ...,
        description="Snapshot of the ICPScore.score that informed classification.",
    )


# ---------------------------------------------------------------------------
# 5. Drafted email (output of the drafting node)
# ---------------------------------------------------------------------------

class DraftedEmail(BaseModel):
    """Personalised outreach email produced by the drafting node."""

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Full email body (plain text).")
    grounded_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Specific enrichment facts that were cited in the email body. "
            "Used to verify grounding and prevent hallucination."
        ),
    )
    tone: str = Field(
        default="professional",
        description="Intended tone, e.g. 'professional', 'warm', 'consultative'.",
    )


# ---------------------------------------------------------------------------
# 6. Nurture segment (output of the segmentation node, NURTURE leads only)
# ---------------------------------------------------------------------------

class NurtureSegment(BaseModel):
    """
    Specific nurture track assignment for a NURTURE-tier lead.

    Produced by segment.py after classification confirms NURTURE tier.
    Every field is required — there is no path that writes a NURTURE lead to
    the CRM without a specific, cited track assignment.
    """

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    nurture_track: str = Field(
        ...,
        description="Machine-readable track ID from nurture_tracks.json, e.g. 'right-industry-no-signal'.",
    )
    track_label: str = Field(
        ...,
        description="Human-readable track label.",
    )
    track_reason: str = Field(
        ...,
        description=(
            "Cited reason for this track assignment — must reference specific "
            "matched/unmatched criteria from ICPScore, not a vague summary."
        ),
    )
    sequence_id: str = Field(
        ...,
        description="ID of the marketing/CRM sequence to enrol this lead in.",
    )
    re_score_days: int = Field(
        ...,
        description="Number of days until this lead should be re-scored.",
    )
    matched_rules: list[str] = Field(
        default_factory=list,
        description="The specific assignment rule patterns that matched for this track.",
    )


# ---------------------------------------------------------------------------
# 7. CRM record (typed, validated output of crm_write.py)
# ---------------------------------------------------------------------------

class CRMRecord(BaseModel):
    """
    Typed, validated CRM record written for every lead at the end of the pipeline.

    Schema varies by tier — use the `tier` field to understand which optional
    fields are populated:

      HOT        : score, send_status, approving_rep_id (if sent), message_id
      NURTURE    : nurture_track, track_label, track_reason, sequence_id, re_score_days
      DISQUALIFY : disqualify_reason (same as classification.reason)

    All tiers share: crm_record_id, lead_id, lead_email, lead_name,
    company, role, tier, classification_reason, written_at.
    """

    # ── Shared fields (all tiers) ─────────────────────────────────────────────
    crm_record_id: str = Field(..., description="Unique ID for this CRM record.")
    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    lead_email: str = Field(..., description="Lead email address.")
    lead_name: str = Field(..., description="Lead full name.")
    company: str = Field(..., description="Lead company name.")
    role: str = Field(..., description="Lead job title.")
    tier: str = Field(..., description="Classification tier: HOT, NURTURE, or DISQUALIFY.")
    classification_reason: str = Field(
        ...,
        description="The full reason string from Classification — always non-empty.",
    )
    icp_score: float = Field(..., description="ICP score at time of classification.")
    written_at: str = Field(..., description="ISO-8601 UTC timestamp of the CRM write.")
    mock: bool = Field(default=True, description="True in all mock/dev environments.")

    # ── HOT-only fields ───────────────────────────────────────────────────────
    send_status: str | None = Field(
        default=None,
        description="HOT only: 'sent', 'rejected_by_rep', or 'pending'.",
    )
    approving_rep_id: str | None = Field(
        default=None,
        description="HOT only: rep_id of the person who approved the send.",
    )
    message_id: str | None = Field(
        default=None,
        description="HOT only: message ID returned by email_send on successful send.",
    )

    # ── NURTURE-only fields ───────────────────────────────────────────────────
    nurture_track: str | None = Field(
        default=None,
        description="NURTURE only: track ID from nurture_tracks.json.",
    )
    track_label: str | None = Field(
        default=None,
        description="NURTURE only: human-readable track label.",
    )
    track_reason: str | None = Field(
        default=None,
        description="NURTURE only: cited reason for this specific track assignment.",
    )
    sequence_id: str | None = Field(
        default=None,
        description="NURTURE only: marketing sequence to enrol the lead in.",
    )
    re_score_days: int | None = Field(
        default=None,
        description="NURTURE only: days until re-scoring.",
    )

    # ── DISQUALIFY-only fields ────────────────────────────────────────────────
    disqualify_reason: str | None = Field(
        default=None,
        description="DISQUALIFY only: detailed reason — same as classification_reason.",
    )


# ---------------------------------------------------------------------------
# 8. Fairness check result (output of governance/fairness_check.py)
# ---------------------------------------------------------------------------

class FairnessResult(BaseModel):
    """
    Result of the identity-blind fairness check for a single lead.

    The check re-runs enrichment → scoring → classification on an anonymised
    copy of the lead (name replaced, email local-part replaced, domain kept)
    and compares the output to the original pipeline run.

    passed == True  iff  original_score == anonymized_score
                    AND  original_tier  == anonymized_tier

    Any discrepancy is a governance fail: discrepancy_details explains exactly
    what differed and which pipeline field caused it.
    """

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    original_score: float = Field(..., description="ICP score from the main pipeline run.")
    anonymized_score: float = Field(..., description="ICP score from the anonymised re-run.")
    original_tier: str = Field(..., description="Classification tier from the main pipeline run.")
    anonymized_tier: str = Field(..., description="Classification tier from the anonymised re-run.")
    passed: bool = Field(
        ...,
        description=(
            "True if score and tier are identical between original and anonymised runs. "
            "False is a governance fail — discrepancy_details will explain the gap."
        ),
    )
    discrepancy_details: str | None = Field(
        default=None,
        description=(
            "Required when passed == False. Describes exactly what differed "
            "and which pipeline field is the likely source."
        ),
    )
    anonymized_lead_id: str = Field(
        ...,
        description="lead_id of the anonymised Lead used for the re-run.",
    )
    checked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the fairness check.",
    )


# ---------------------------------------------------------------------------
# 9. Injection check result (output of governance/injection_check.py)
# ---------------------------------------------------------------------------

class InjectionCheckResult(BaseModel):
    """
    Result of a lightweight prompt-injection detection scan on lead-supplied text.

    The check scans form_text (or any untrusted string) for patterns that
    commonly appear in prompt-injection attempts: instruction overrides, system
    spoofing, approval bypasses, and send-now directives.

    is_suspicious == True  iff at least one pattern matched.

    This is an INFORMATIONAL signal only.  It is recorded in the audit log and
    can trigger eval-suite alerts, but it NEVER changes scoring, classification,
    or send behaviour.  The pipeline continues exactly as normal regardless of
    this result — the injection simply has no effect because every decision node
    ignores free-text for routing purposes.
    """

    lead_id: str = Field(..., description="Foreign key back to Lead.lead_id.")
    is_suspicious: bool = Field(
        ...,
        description=(
            "True if at least one injection-pattern matched in the scanned text. "
            "False means no obvious injection attempt was detected."
        ),
    )
    matched_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable labels for each pattern that matched "
            "(e.g. 'ignore_previous_instructions', 'system_override', 'send_now'). "
            "Empty when is_suspicious == False."
        ),
    )
    scanned_text_preview: str | None = Field(
        default=None,
        description=(
            "First 200 characters of the scanned text, for audit-log readability. "
            "Never stores the full text to avoid log bloat."
        ),
    )
    checked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the injection check.",
    )


# ---------------------------------------------------------------------------
# 10. Approval result (output of the human gate node)
# ---------------------------------------------------------------------------

class ApprovalResult(BaseModel):
    """
    Record of the human reviewer's decision at the approval gate.

    Produced by approval_gate.request_approval() and consumed by the pipeline
    to decide whether to call send_email() or discard the draft.

    Invariant: send_email() MUST NOT be called unless approved == True AND
    approval_token is a non-empty string.  The token is generated inside the
    approval gate and is single-use per lead per run.
    """

    approved: bool = Field(
        ...,
        description="True if the rep approved the email for sending.",
    )
    edited_email: DraftedEmail | None = Field(
        default=None,
        description=(
            "If the rep edited the draft before approving, this holds the "
            "edited version.  None if approved as-is or not approved."
        ),
    )
    approval_token: str | None = Field(
        default=None,
        description=(
            "Single-use UUID token generated at approval time.  "
            "Required by send_email() to proceed.  None when not approved."
        ),
    )
    rep_id: str = Field(
        ...,
        description="Identifier of the rep who acted on this approval request.",
    )
    decision: str = Field(
        ...,
        description="Human-readable decision: 'approved', 'approved_with_edits', or 'rejected'.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Required when approved == False; reason the rep rejected the draft.",
    )
    decided_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the approval decision.",
    )


# ---------------------------------------------------------------------------
# 11. Audit record (written to audit_log.jsonl after every pipeline run)
# ---------------------------------------------------------------------------

class AuditRecord(BaseModel):
    """
    Immutable record of a complete pipeline run for a single lead.

    Every stage writes its input/output here so there is a full, inspectable
    audit trail for compliance, debugging, and evaluation.
    """

    audit_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this audit entry.",
    )
    lead_id: str = Field(..., description="The lead this record describes.")
    pipeline_run_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the pipeline run started.",
    )

    # ── Stage inputs / outputs (None until that stage executes) ──────────────
    raw_lead: Lead | None = Field(default=None)
    enriched_lead: EnrichedLead | None = Field(default=None)
    icp_score: ICPScore | None = Field(default=None)
    classification: Classification | None = Field(default=None)
    drafted_email: DraftedEmail | None = Field(default=None)

    # ── Final decision ────────────────────────────────────────────────────────
    final_decision: str | None = Field(
        default=None,
        description=(
            "Terminal state of the lead: 'email_sent', 'crm_updated', "
            "'disqualified', 'pending_human_review', etc."
        ),
    )

    # ── Human-in-the-loop gate ────────────────────────────────────────────────
    human_action: str | None = Field(
        default=None,
        description="Action taken by a human reviewer, if the human gate was invoked.",
    )
    human_reviewer_id: str | None = Field(
        default=None,
        description="Identifier of the human who acted on this lead.",
    )
    human_reviewed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the human review action.",
    )

    # ── Generic metadata bag for future extensibility ─────────────────────────
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key/value metadata for debugging or future use.",
    )
