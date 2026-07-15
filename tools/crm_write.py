"""
tools/crm_write.py
------------------
Mock CRM write tool — produces and validates a typed CRMRecord for every lead.

Public API
----------
    write_crm(lead, classification, action,
              icp_score, nurture_segment=None,
              send_result=None, approval=None) -> CRMRecord

Every call returns a fully-populated, Pydantic-validated CRMRecord.
The record schema varies by tier but all required shared fields are always present.

Tier-specific requirements
---------------------------
  HOT        : send_status (sent/rejected_by_rep/pending), approving_rep_id if sent,
               message_id if sent.  These come from send_result + approval.
  NURTURE    : nurture_track, track_label, track_reason, sequence_id, re_score_days.
               These come from the NurtureSegment produced by segment.py.
               CRMWriteError is raised if nurture_segment is None.
  DISQUALIFY : disqualify_reason (copy of classification.reason).

Validation
----------
CRMRecord is a Pydantic model — Pydantic validates all fields on construction.
An additional _validate_crm_record() check enforces tier-specific field presence
before the mock write proceeds.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Final

from agent.models import (
    ApprovalResult,
    Classification,
    CRMRecord,
    ICPScore,
    Lead,
    LeadTier,
    NurtureSegment,
)

_VALID_ACTIONS: Final[frozenset[str]] = frozenset({"archive", "nurture", "hot_sent", "hot_rejected"})


class CRMWriteError(Exception):
    """Raised when write_crm() is called with invalid or incomplete arguments."""


def write_crm(
    lead: Lead,
    classification: Classification,
    action: str,
    icp_score: ICPScore,
    *,
    nurture_segment: NurtureSegment | None = None,
    send_result: dict | None = None,
    approval: ApprovalResult | None = None,
) -> CRMRecord:
    """
    Build, validate, and mock-write a CRM record for a lead.

    Parameters
    ----------
    lead : Lead
    classification : Classification
    action : str
        One of: 'archive', 'nurture', 'hot_sent', 'hot_rejected'.
    icp_score : ICPScore
    nurture_segment : NurtureSegment | None
        Required when action == 'nurture'.
    send_result : dict | None
        Required when action == 'hot_sent' — payload from email_send.send_email().
    approval : ApprovalResult | None
        Passed for all HOT leads to record who approved/rejected.

    Returns
    -------
    CRMRecord
        Fully-populated, Pydantic-validated CRM record.

    Raises
    ------
    CRMWriteError
        On missing required fields, unrecognised action, or empty reason.
    """
    if action not in _VALID_ACTIONS:
        raise CRMWriteError(
            f"write_crm() called with unknown action={action!r}. "
            f"Valid actions: {sorted(_VALID_ACTIONS)}"
        )

    if not classification.reason or not classification.reason.strip():
        raise CRMWriteError(
            "write_crm() requires a non-empty classification.reason. "
            "Every CRM record must carry a cited reason."
        )

    tier = classification.tier
    written_at = datetime.now(tz=timezone.utc).isoformat()
    crm_record_id = f"crm-{uuid.uuid4()}"

    # ── Build tier-specific fields ────────────────────────────────────────────

    # HOT fields
    send_status: str | None = None
    approving_rep_id: str | None = None
    message_id: str | None = None

    if tier == LeadTier.HOT:
        if action == "hot_sent":
            if send_result is None:
                raise CRMWriteError(
                    "write_crm(action='hot_sent') requires send_result — "
                    "pass the dict returned by email_send.send_email()."
                )
            send_status = "sent"
            message_id = send_result.get("message_id")
            approving_rep_id = approval.rep_id if approval else None
        elif action == "hot_rejected":
            send_status = "rejected_by_rep"
            approving_rep_id = approval.rep_id if approval else None
        else:
            send_status = "pending"

    # NURTURE fields
    nurture_track: str | None = None
    track_label: str | None = None
    track_reason: str | None = None
    sequence_id: str | None = None
    re_score_days: int | None = None

    if tier == LeadTier.NURTURE:
        if nurture_segment is None:
            raise CRMWriteError(
                "write_crm() for a NURTURE lead requires nurture_segment. "
                "Run segment.segment_lead() before calling write_crm()."
            )
        nurture_track = nurture_segment.nurture_track
        track_label = nurture_segment.track_label
        track_reason = nurture_segment.track_reason
        sequence_id = nurture_segment.sequence_id
        re_score_days = nurture_segment.re_score_days

    # DISQUALIFY fields
    disqualify_reason: str | None = None
    if tier == LeadTier.DISQUALIFY:
        disqualify_reason = classification.reason

    # ── Construct and validate CRMRecord ──────────────────────────────────────
    record = CRMRecord(
        crm_record_id=crm_record_id,
        lead_id=lead.lead_id,
        lead_email=str(lead.email),
        lead_name=lead.name,
        company=lead.company,
        role=lead.role,
        tier=tier.value,
        classification_reason=classification.reason,
        icp_score=icp_score.score,
        written_at=written_at,
        mock=True,
        # HOT
        send_status=send_status,
        approving_rep_id=approving_rep_id,
        message_id=message_id,
        # NURTURE
        nurture_track=nurture_track,
        track_label=track_label,
        track_reason=track_reason,
        sequence_id=sequence_id,
        re_score_days=re_score_days,
        # DISQUALIFY
        disqualify_reason=disqualify_reason,
    )

    # Post-construction validation (belt-and-suspenders beyond Pydantic)
    _validate_crm_record(record)

    # ── Mock print ────────────────────────────────────────────────────────────
    _print_crm_record(record)

    return record


def _validate_crm_record(record: CRMRecord) -> None:
    """
    Enforce tier-specific field presence rules that Pydantic's Optional
    typing cannot enforce on its own.
    """
    tier = record.tier

    if tier == "HOT":
        if record.send_status is None:
            raise CRMWriteError("HOT CRMRecord missing send_status.")
    elif tier == "NURTURE":
        missing = [
            f for f in ("nurture_track", "track_label", "track_reason", "sequence_id")
            if getattr(record, f) is None
        ]
        if missing:
            raise CRMWriteError(f"NURTURE CRMRecord missing fields: {missing}")
    elif tier == "DISQUALIFY":
        if not record.disqualify_reason:
            raise CRMWriteError("DISQUALIFY CRMRecord missing disqualify_reason.")


def _print_crm_record(record: CRMRecord) -> None:
    """Pretty-print the CRM record to stdout (mock output)."""
    print(f"\n  🗄  CRM RECORD — tier={record.tier}  id={record.crm_record_id}")
    print(f"     Lead      : {record.lead_name} <{record.lead_email}>")
    print(f"     Company   : {record.company}  ({record.role})")
    print(f"     ICP score : {record.icp_score:.1f}")
    reason_preview = record.classification_reason[:110]
    if len(record.classification_reason) > 110:
        reason_preview += "…"
    print(f"     Reason    : {reason_preview}")

    if record.tier == "HOT":
        print(f"     Send      : {record.send_status}")
        if record.approving_rep_id:
            print(f"     Approved by: {record.approving_rep_id}")
        if record.message_id:
            print(f"     Msg ID    : {record.message_id}")

    elif record.tier == "NURTURE":
        print(f"     Track     : {record.nurture_track}  ({record.track_label})")
        print(f"     Sequence  : {record.sequence_id}")
        print(f"     Re-score  : {record.re_score_days} days")
        track_reason_preview = (record.track_reason or "")[:110]
        if record.track_reason and len(record.track_reason) > 110:
            track_reason_preview += "…"
        print(f"     Track reason: {track_reason_preview}")

    elif record.tier == "DISQUALIFY":
        dq_preview = (record.disqualify_reason or "")[:110]
        if record.disqualify_reason and len(record.disqualify_reason) > 110:
            dq_preview += "…"
        print(f"     DQ reason : {dq_preview}")

    print(f"     Written at: {record.written_at}")
