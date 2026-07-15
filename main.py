"""
main.py
-------
CLI entry point for the Lead Qualification & Outreach Agent.

Step 6: enrich → score → classify → segment (NURTURE only) →
        [draft → approve → send → crm(HOT)] |
        [segment → crm(NURTURE)]            |
        [crm(DISQUALIFY)]

Audit log captures the full CRMRecord for every lead.

Usage
-----
    python main.py                     # runs the first sample lead (index 0)
    python main.py --lead-index 2      # runs sample lead at index 2 (0-based)
    python main.py --all               # runs every sample lead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

_TIER_COLOUR = {
    "HOT":        "\033[92m",
    "NURTURE":    "\033[93m",
    "DISQUALIFY": "\033[91m",
}
_RESET = "\033[0m"
_CYAN = "\033[96m"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sample_leads() -> list[dict]:
    sample_path = DATA_DIR / "sample_leads.json"
    if not sample_path.exists():
        print(f"[ERROR] Sample leads file not found: {sample_path}", file=sys.stderr)
        sys.exit(1)
    with sample_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_icp() -> dict:
    icp_path = DATA_DIR / "icp.json"
    if not icp_path.exists():
        print(f"[ERROR] ICP file not found: {icp_path}", file=sys.stderr)
        sys.exit(1)
    with icp_path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_enriched(enriched) -> None:
    print(f"  {'Industry':<22} {enriched.industry or '—'}")
    print(f"  {'Company size':<22} {enriched.company_size or '—'}")
    print(f"  {'Revenue band':<22} {enriched.annual_revenue_usd or '—'}")
    print(f"  {'Location':<22} {enriched.location or '—'}")
    print(f"  {'Tech stack':<22} {', '.join(enriched.tech_stack) or '—'}")
    print(f"  {'Enrichment confidence':<22} {enriched.enrichment_confidence:.0%}")
    if enriched.buying_signals:
        print(f"  {'Buying signals':<22}")
        for sig in enriched.buying_signals:
            print(f"    • {sig}")
    else:
        print(f"  {'Buying signals':<22} none detected")


def _print_score(score) -> None:
    print(f"  {'ICP score':<22} {score.score:.1f} / 100")
    if score.matched_criteria:
        print(f"  {'Matched':<22}")
        for c in score.matched_criteria:
            print(f"    ✓ {c}")
    if score.unmatched_criteria:
        print(f"  {'Unmatched':<22}")
        for c in score.unmatched_criteria:
            print(f"    ✗ {c}")
    if score.scoring_notes:
        print(f"  {'Notes':<22} {score.scoring_notes}")


def _print_classification(cls) -> None:
    tier_val = cls.tier.value
    colour = _TIER_COLOUR.get(tier_val, "")
    print(f"  {'Tier':<22} {colour}{tier_val}{_RESET}")
    print(f"  {'Score used':<22} {cls.icp_score_used:.1f} / 100")
    reason_lines = _wrap(cls.reason, width=54, indent="    ")
    print(f"  {'Reason':<22}")
    for line in reason_lines:
        print(line)
    if cls.cited_signals:
        print(f"  {'Cited signals':<22}")
        for sig in cls.cited_signals:
            print(f"    › {sig}")


def _print_segment(seg) -> None:
    print(f"  {'Track ID':<22} {seg.nurture_track}")
    print(f"  {'Track label':<22} {seg.track_label}")
    print(f"  {'Sequence':<22} {seg.sequence_id}")
    print(f"  {'Re-score in':<22} {seg.re_score_days} days")
    reason_lines = _wrap(seg.track_reason, width=54, indent="    ")
    print(f"  {'Track reason':<22}")
    for line in reason_lines:
        print(line)
    if seg.matched_rules:
        print(f"  {'Matched rules':<22}")
        for r in seg.matched_rules:
            print(f"    › {r}")


def _print_draft(draft) -> None:
    print(f"  {'Subject':<22} {draft.subject}")
    print(f"  {'Tone':<22} {draft.tone}")
    print(f"  {'Grounded facts':<22}")
    for fact in draft.grounded_facts:
        print(f"    • {fact}")
    print(f"  {'Body':<22}")
    for line in draft.body.splitlines():
        print(f"    {line}")


def _print_fairness(fairness) -> None:
    status_colour = "\033[92m" if fairness.passed else "\033[91m"
    status_label = "PASS ✓" if fairness.passed else "FAIL ✗"
    print(f"  {'Result':<22} {status_colour}{status_label}{_RESET}")
    print(f"  {'Original score':<22} {fairness.original_score:.1f}")
    print(f"  {'Anonymised score':<22} {fairness.anonymized_score:.1f}")
    print(f"  {'Original tier':<22} {fairness.original_tier}")
    print(f"  {'Anonymised tier':<22} {fairness.anonymized_tier}")
    if not fairness.passed and fairness.discrepancy_details:
        disc_lines = _wrap(fairness.discrepancy_details, width=54, indent="    ")
        print(f"  {'Discrepancy':<22}")
        for line in disc_lines:
            print(line)


def _print_injection(injection) -> None:
    _YELLOW = "\033[93m"
    _GREEN = "\033[92m"
    if injection.is_suspicious:
        status_colour = _YELLOW
        status_label = "SUSPICIOUS ⚠"
    else:
        status_colour = _GREEN
        status_label = "CLEAN ✓"
    print(f"  {'Result':<22} {status_colour}{status_label}{_RESET}")
    if injection.is_suspicious:
        print(f"  {'Matched patterns':<22}")
        for pattern in injection.matched_patterns:
            print(f"    ⚠ {pattern}")
        if injection.scanned_text_preview:
            preview_lines = _wrap(injection.scanned_text_preview, width=54, indent="    ")
            print(f"  {'Text preview':<22}")
            for line in preview_lines:
                print(line)
        print(f"  {'Pipeline effect':<22} none — scoring uses real signals only")


def _print_crm_summary(crm: "CRMRecord") -> None:
    colour = _TIER_COLOUR.get(crm.tier, "")
    print(f"  {'CRM record ID':<22} {crm.crm_record_id}")
    print(f"  {'Tier':<22} {colour}{crm.tier}{_RESET}")
    print(f"  {'ICP score':<22} {crm.icp_score:.1f}")
    if crm.tier == "HOT":
        print(f"  {'Send status':<22} {crm.send_status}")
        if crm.approving_rep_id:
            print(f"  {'Approved by':<22} {crm.approving_rep_id}")
        if crm.message_id:
            print(f"  {'Message ID':<22} {crm.message_id}")
    elif crm.tier == "NURTURE":
        print(f"  {'Nurture track':<22} {crm.nurture_track}")
        print(f"  {'Track label':<22} {crm.track_label}")
        print(f"  {'Sequence':<22} {crm.sequence_id}")
        print(f"  {'Re-score in':<22} {crm.re_score_days} days")
    elif crm.tier == "DISQUALIFY":
        dq_lines = _wrap(crm.disqualify_reason or "", width=54, indent="    ")
        print(f"  {'DQ reason':<22}")
        for line in dq_lines:
            print(line)
    print(f"  {'Written at':<22} {crm.written_at}")


def _wrap(text: str, width: int = 60, indent: str = "  ") -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current)
            current = indent + word
        else:
            current = current + " " + word if current.strip() else indent + word
    if current.strip():
        lines.append(current)
    return lines


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(lead_data: dict, icp: dict) -> None:
    """
    Full pipeline (Step 6):
      enrich → score → classify
        HOT       → draft → approve → send → crm(hot_sent/hot_rejected)
        NURTURE   → segment → crm(nurture)
        DISQUALIFY→ crm(archive)

    Every path writes a typed CRMRecord to the audit log.
    """
    from agent.models import Lead, LeadTier, CRMRecord
    from tools.enrichment_lookup import enrich_lead
    from agent.nodes.score import score_lead
    from agent.nodes.classify import classify_lead
    from agent.nodes.segment import segment_lead
    from agent.nodes.draft import draft_email
    from agent.nodes.approval_gate import request_approval
    from tools.email_send import send_email
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    clean = {k: v for k, v in lead_data.items() if not k.startswith("_")}
    lead = Lead(**clean)
    divider = "─" * 64

    print(f"\n{divider}")
    print(f"  LEAD  {lead.name} <{lead.email}>")
    print(f"  {'Company':<22} {lead.company}")
    print(f"  {'Role':<22} {lead.role}")
    print(f"  {'Source':<22} {lead.source}")
    print(f"  {'Lead ID':<22} {lead.lead_id}")
    if lead.form_text:
        preview = lead.form_text[:110] + ("…" if len(lead.form_text) > 110 else "")
        print(f"  {'Form text':<22} {preview}")

    # ── Stage 0: Injection check (runs before enrichment, informational only) ─
    print(f"\n  ┌─ INJECTION CHECK {'─' * 43}")
    from governance.injection_check import check_for_injection_attempt
    injection_result = check_for_injection_attempt(
        text=lead.form_text or "",
        lead_id=lead.lead_id,
    )
    _print_injection(injection_result)

    # ── Stage 1: Enrich ───────────────────────────────────────────────────────
    print(f"\n  ┌─ ENRICHMENT {'─' * 48}")
    enriched = enrich_lead(lead)
    _print_enriched(enriched)

    # ── Stage 2: Score ────────────────────────────────────────────────────────
    print(f"\n  ┌─ SCORING {'─' * 51}")
    icp_score = score_lead(enriched, icp)
    _print_score(icp_score)

    # ── Stage 3: Classify ─────────────────────────────────────────────────────
    print(f"\n  ┌─ CLASSIFICATION {'─' * 44}")
    classification = classify_lead(enriched, icp_score)
    _print_classification(classification)

    tier = classification.tier

    # ── Stage 4: Fairness check (runs for EVERY lead, not a gate) ────────────
    print(f"\n  ┌─ FAIRNESS CHECK {'─' * 44}")
    from governance.fairness_check import run_fairness_check
    fairness = run_fairness_check(lead, icp_score, classification)
    _print_fairness(fairness)

    # ── Audit payload (built up incrementally) ────────────────────────────────
    audit_payload: dict = {
        "stage": "full_pipeline",
        "pipeline_step": 8,
        "raw_lead": lead.model_dump(mode="json"),
        "injection_check": injection_result.model_dump(mode="json"),
        "enriched_lead": enriched.model_dump(mode="json"),
        "icp_score": icp_score.model_dump(mode="json"),
        "classification": classification.model_dump(mode="json"),
        "fairness_result": fairness.model_dump(mode="json"),
    }

    crm_record: CRMRecord | None = None

    # ══════════════════════════════════════════════════════════════════════════
    # HOT: draft → human gate → send (if approved) → crm
    # ══════════════════════════════════════════════════════════════════════════
    if tier == LeadTier.HOT:
        # Stage 4: Draft
        print(f"\n  ┌─ EMAIL DRAFT {'─' * 47}")
        drafted = draft_email(enriched, classification)
        _print_draft(drafted)
        audit_payload["drafted_email"] = drafted.model_dump(mode="json")

        # Stage 5: Human approval gate
        print(f"\n  ┌─ APPROVAL GATE {'─' * 45}")
        approval = request_approval(drafted, lead)
        audit_payload["approval_decision"] = approval.model_dump(mode="json")

        if approval.approved:
            email_to_send = approval.edited_email if approval.edited_email else drafted
            send_result = send_email(email_to_send, lead, approval.approval_token)  # type: ignore[arg-type]
            audit_payload["send_result"] = send_result
            audit_payload["status"] = "email_sent"
            print(f"\n  ┌─ OUTCOME {'─' * 51}")
            print(f"  {'Status':<22} \033[92mEmail sent\033[0m")
            print(f"  {'Message ID':<22} {send_result['message_id']}")

            # Stage 6: CRM write (hot_sent)
            print(f"\n  ┌─ CRM RECORD {'─' * 48}")
            crm_record = write_crm(
                lead, classification, "hot_sent", icp_score,
                send_result=send_result, approval=approval,
            )
        else:
            audit_payload["status"] = "email_rejected_by_rep"
            print(f"\n  ┌─ OUTCOME {'─' * 51}")
            print(f"  {'Status':<22} \033[91mRejected — not sent\033[0m")
            print(f"  {'Rejected by':<22} {approval.rep_id}")
            print(f"  {'Reason':<22} {approval.rejection_reason}")

            # Stage 6: CRM write (hot_rejected)
            print(f"\n  ┌─ CRM RECORD {'─' * 48}")
            crm_record = write_crm(
                lead, classification, "hot_rejected", icp_score,
                approval=approval,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # NURTURE: segment → crm(nurture)
    # ══════════════════════════════════════════════════════════════════════════
    elif tier == LeadTier.NURTURE:
        print(f"\n  ┌─ EMAIL DRAFT {'─' * 47}")
        colour = _TIER_COLOUR["NURTURE"]
        print(f"  {'Skipped':<22} lead is {colour}NURTURE{_RESET} — no draft or email.")

        # Stage 4: Segment
        print(f"\n  ┌─ NURTURE SEGMENTATION {'─' * 39}")
        nurture_segment = segment_lead(icp_score, classification)
        _print_segment(nurture_segment)
        audit_payload["nurture_segment"] = nurture_segment.model_dump(mode="json")

        # Stage 5: CRM write (nurture)
        print(f"\n  ┌─ CRM RECORD {'─' * 48}")
        crm_record = write_crm(
            lead, classification, "nurture", icp_score,
            nurture_segment=nurture_segment,
        )
        audit_payload["status"] = "crm_nurture_enrolled"

    # ══════════════════════════════════════════════════════════════════════════
    # DISQUALIFY: crm(archive) — no draft, no email, ever
    # ══════════════════════════════════════════════════════════════════════════
    else:
        print(f"\n  ┌─ EMAIL DRAFT {'─' * 47}")
        colour = _TIER_COLOUR["DISQUALIFY"]
        print(f"  {'Skipped':<22} lead is {colour}DISQUALIFY{_RESET} — no draft or email.")

        # Stage 4: CRM write (archive)
        print(f"\n  ┌─ CRM RECORD {'─' * 48}")
        crm_record = write_crm(
            lead, classification, "archive", icp_score,
        )
        audit_payload["status"] = "archived_disqualified"

    # ── Outcome summary ───────────────────────────────────────────────────────
    if crm_record:
        audit_payload["crm_record"] = crm_record.model_dump(mode="json")
        print(f"\n  ┌─ OUTCOME {'─' * 51}")
        _print_crm_summary(crm_record)

    print(f"\n{divider}\n")

    append_audit_record(lead_id=lead.lead_id, record=audit_payload)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lead Qualification & Outreach Agent — Step 8"
    )
    parser.add_argument(
        "--lead-index",
        type=int,
        default=0,
        metavar="N",
        help="0-based index of the sample lead to run (default: 0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every sample lead through the pipeline",
    )
    args = parser.parse_args()

    leads = _load_sample_leads()
    icp = _load_icp()

    if args.all:
        print(f"Running {len(leads)} leads through full pipeline (Step 6)…")
        for lead_data in leads:
            run_pipeline(lead_data, icp)
    else:
        if args.lead_index >= len(leads):
            print(
                f"[ERROR] --lead-index {args.lead_index} is out of range "
                f"(only {len(leads)} leads available).",
                file=sys.stderr,
            )
            sys.exit(1)
        run_pipeline(leads[args.lead_index], icp)


if __name__ == "__main__":
    main()
