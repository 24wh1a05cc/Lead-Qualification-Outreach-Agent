"""
ui/app.py
---------
Streamlit front-end for the Lead Qualification & Outreach Agent.

Thin wrapper around the existing pipeline — every business-logic call goes
directly to agent/, tools/, governance/.  No pipeline logic lives here.

Layout
------
  Sidebar  : lead selector (sample leads dropdown) OR manual entry form
             + "Run Eval Suite" button
  Main     : pipeline stage cards rendered as the run progresses
             (injection check → enrichment → scoring → classification →
              draft/approval [HOT] | nurture segment [NURTURE] | archive [DISQUALIFY])
             + audit trail expander at the bottom

Usage
-----
    streamlit run ui/app.py
    streamlit run ui/app.py --server.headless true   # CI / smoke-test
"""

from __future__ import annotations

import io
import json
import sys
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ── Project root on sys.path so local imports work when run from any cwd ──────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Lazy pipeline imports (kept here to avoid module-level side-effects) ──────
# Imported inside functions so Streamlit's import caching doesn't interfere
# with patching in the eval suite.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
AUDIT_LOG_PATH = _PROJECT_ROOT / "audit_log.jsonl"

_TIER_COLOUR = {
    "HOT": "#22c55e",        # green
    "NURTURE": "#f59e0b",    # amber
    "DISQUALIFY": "#ef4444", # red
}
_TIER_EMOJI = {
    "HOT": "🔥",
    "NURTURE": "🌱",
    "DISQUALIFY": "🚫",
}

# ---------------------------------------------------------------------------
# Page configuration (must be FIRST Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lead Qualification Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, purposeful
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Tier badge pills */
.badge-hot      { background:#166534; color:#bbf7d0; padding:3px 10px;
                  border-radius:12px; font-weight:700; font-size:0.85rem; }
.badge-nurture  { background:#78350f; color:#fde68a; padding:3px 10px;
                  border-radius:12px; font-weight:700; font-size:0.85rem; }
.badge-disq     { background:#7f1d1d; color:#fecaca; padding:3px 10px;
                  border-radius:12px; font-weight:700; font-size:0.85rem; }

/* Stage header bar */
.stage-header { background:#1e293b; border-left:4px solid #38bdf8;
                padding:6px 14px; border-radius:4px; margin-bottom:8px;
                font-weight:600; font-size:0.9rem; color:#e2e8f0; }

/* Score bar track */
.score-track { background:#334155; border-radius:6px; height:14px;
               width:100%; overflow:hidden; }
.score-fill  { height:14px; border-radius:6px; }

/* Audit log entry */
.audit-entry { font-family:monospace; font-size:0.78rem; color:#94a3b8;
               border-left:2px solid #475569; padding:4px 10px;
               margin-bottom:4px; }

/* Email body block */
.email-body { background:#0f172a; border:1px solid #334155; border-radius:6px;
              padding:14px; font-family:monospace; font-size:0.84rem;
              color:#e2e8f0; white-space:pre-wrap; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (cached so they don't re-read on every Streamlit rerun)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_sample_leads() -> list[dict]:
    with (DATA_DIR / "sample_leads.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def _load_icp() -> dict:
    with (DATA_DIR / "icp.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(lead_data: dict) -> None:
    """
    Execute the full pipeline for one lead and store every stage result in
    st.session_state so the render functions can read them.

    Mirrors main.py's run_pipeline() but:
      • Writes to session_state instead of stdout.
      • Does NOT call request_approval() — the UI presents the approval UI
        directly after the pipeline run completes.
      • Still calls append_audit_record() so the live audit log is updated.
    """
    from agent.models import Lead, LeadTier
    from tools.enrichment_lookup import enrich_lead
    from agent.nodes.score import score_lead
    from agent.nodes.classify import classify_lead
    from agent.nodes.segment import segment_lead
    from agent.nodes.draft import draft_email
    from governance.injection_check import check_for_injection_attempt
    from governance.fairness_check import run_fairness_check
    from governance.audit_log import append_audit_record

    icp = _load_icp()

    # Strip internal _keys used for metadata in sample_leads.json
    clean = {k: v for k, v in lead_data.items() if not k.startswith("_")}
    lead = Lead(**clean)

    # Stage 0 — Injection check
    inj = check_for_injection_attempt(lead.form_text or "", lead.lead_id)

    # Stage 1 — Enrich
    enriched = enrich_lead(lead)

    # Stage 2 — Score
    icp_score = score_lead(enriched, icp)

    # Stage 3 — Classify
    classification = classify_lead(enriched, icp_score)

    # Stage 4 — Fairness check (always)
    fairness = run_fairness_check(lead, icp_score, classification)

    # Stage 5 — Tier-specific downstream
    tier = classification.tier
    drafted = None
    nurture_segment = None

    if tier == LeadTier.HOT:
        drafted = draft_email(enriched, classification)
    elif tier == LeadTier.NURTURE:
        nurture_segment = segment_lead(icp_score, classification)

    # ── Persist to session_state ──────────────────────────────────────────────
    st.session_state["pipeline"] = {
        "lead": lead,
        "enriched": enriched,
        "icp_score": icp_score,
        "classification": classification,
        "fairness": fairness,
        "injection": inj,
        "drafted": drafted,
        "nurture_segment": nurture_segment,
        "tier": tier,
        "approval_state": "pending",   # pending | approved | edited | rejected
        "approval_result": None,
        "send_result": None,
        "crm_record": None,
        "audit_payload": {
            "stage": "full_pipeline",
            "pipeline_step": 10,
            "raw_lead": lead.model_dump(mode="json"),
            "injection_check": inj.model_dump(mode="json"),
            "enriched_lead": enriched.model_dump(mode="json"),
            "icp_score": icp_score.model_dump(mode="json"),
            "classification": classification.model_dump(mode="json"),
            "fairness_result": fairness.model_dump(mode="json"),
        },
    }
    # Write initial audit record (will be updated after approval decision)
    append_audit_record(lead_id=lead.lead_id, record=st.session_state["pipeline"]["audit_payload"])


# ---------------------------------------------------------------------------
# Render helpers — one function per pipeline stage card
# ---------------------------------------------------------------------------

def _stage_header(label: str, icon: str = "▶") -> None:
    st.markdown(f'<div class="stage-header">{icon} {label}</div>', unsafe_allow_html=True)


def _tier_badge(tier_val: str) -> str:
    css = {"HOT": "badge-hot", "NURTURE": "badge-nurture", "DISQUALIFY": "badge-disq"}.get(tier_val, "badge-disq")
    emoji = _TIER_EMOJI.get(tier_val, "")
    return f'<span class="{css}">{emoji} {tier_val}</span>'


def _score_bar(score: float) -> None:
    colour = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    pct = int(score)
    st.markdown(
        f'<div class="score-track"><div class="score-fill" '
        f'style="width:{pct}%; background:{colour};"></div></div>',
        unsafe_allow_html=True,
    )


def render_injection(p: dict) -> None:
    inj = p["injection"]
    _stage_header("Stage 0 — Injection Check", "🛡️")
    if inj.is_suspicious:
        st.warning(f"⚠️ **Suspicious content detected** in form text — flagged for audit. "
                   f"Pipeline continues on real signals only.")
        for pat in inj.matched_patterns:
            st.code(pat, language=None)
    else:
        st.success("✅ Form text clean — no injection patterns detected.")


def render_enrichment(p: dict) -> None:
    enriched = p["enriched"]
    _stage_header("Stage 1 — Enrichment", "🔍")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Industry", enriched.industry or "—")
        st.metric("Company size", enriched.company_size or "—")
    with col2:
        st.metric("Revenue band", enriched.annual_revenue_usd or "—")
        st.metric("Location", enriched.location or "—")
    with col3:
        conf_pct = int(enriched.enrichment_confidence * 100)
        st.metric("Enrichment confidence", f"{conf_pct}%")
        stack = ", ".join(enriched.tech_stack) if enriched.tech_stack else "—"
        st.caption(f"**Tech stack:** {stack}")

    if enriched.buying_signals:
        with st.expander(f"Buying signals ({len(enriched.buying_signals)})"):
            for sig in enriched.buying_signals:
                st.markdown(f"- `{sig}`")
    else:
        st.caption("No buying signals detected.")


def render_scoring(p: dict) -> None:
    icp_score = p["icp_score"]
    _stage_header("Stage 2 — ICP Scoring", "📊")
    col_score, col_detail = st.columns([1, 2])
    with col_score:
        st.metric("ICP Score", f"{icp_score.score:.1f} / 100")
        _score_bar(icp_score.score)
    with col_detail:
        if icp_score.matched_criteria:
            with st.expander(f"✅ Matched ({len(icp_score.matched_criteria)})", expanded=True):
                for c in icp_score.matched_criteria:
                    st.markdown(f"- {c}")
        if icp_score.unmatched_criteria:
            with st.expander(f"❌ Unmatched ({len(icp_score.unmatched_criteria)})"):
                for c in icp_score.unmatched_criteria:
                    st.markdown(f"- {c}")
    if icp_score.scoring_notes:
        st.caption(f"ℹ️ {icp_score.scoring_notes}")


def render_classification(p: dict) -> None:
    cls = p["classification"]
    tier_val = cls.tier.value
    _stage_header("Stage 3 — Classification", "🏷️")
    col_tier, col_reason = st.columns([1, 3])
    with col_tier:
        st.markdown(_tier_badge(tier_val), unsafe_allow_html=True)
        st.metric("Score used", f"{cls.icp_score_used:.1f}")
    with col_reason:
        st.markdown(f"**Reason:** {cls.reason}")
        if cls.cited_signals:
            st.caption("Cited signals: " + " · ".join(f"`{s}`" for s in cls.cited_signals[:4]))


def render_fairness(p: dict) -> None:
    f = p["fairness"]
    _stage_header("Stage 4 — Fairness Check", "⚖️")
    col1, col2, col3 = st.columns(3)
    with col1:
        icon = "✅" if f.passed else "❌"
        st.metric("Result", f"{icon} {'PASS' if f.passed else 'FAIL'}")
    with col2:
        st.metric("Original score", f"{f.original_score:.1f}")
        st.metric("Anonymised score", f"{f.anonymized_score:.1f}")
    with col3:
        st.metric("Original tier", f.original_tier)
        st.metric("Anonymised tier", f.anonymized_tier)
    if not f.passed and f.discrepancy_details:
        st.error(f"⚠️ {f.discrepancy_details}")


def render_hot_path(p: dict) -> None:
    """Render the draft email and the interactive approval gate for HOT leads."""
    from agent.models import ApprovalResult, DraftedEmail
    from tools.email_send import send_email
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    drafted = p["drafted"]
    lead = p["lead"]
    cls = p["classification"]
    icp_score = p["icp_score"]
    approval_state = p["approval_state"]

    # ── Draft display ─────────────────────────────────────────────────────────
    _stage_header("Stage 5 — Email Draft (HOT)", "✉️")
    st.markdown(f"**Subject:** {drafted.subject}")
    st.markdown(f'<div class="email-body">{drafted.body}</div>', unsafe_allow_html=True)
    with st.expander(f"Grounded facts ({len(drafted.grounded_facts)})"):
        for fact in drafted.grounded_facts:
            st.markdown(f"- `{fact}`")

    # ── Approval gate ─────────────────────────────────────────────────────────
    _stage_header("Stage 6 — Approval Gate", "🔐")

    if approval_state == "pending":
        st.info("📋 Email is **held for approval**. Choose an action below.")
        col_a, col_e, col_r = st.columns(3)

        with col_a:
            if st.button("✅ Approve & Send", use_container_width=True, type="primary"):
                _do_approve(p, drafted, lead, cls, icp_score, edited=False)
                st.rerun()

        with col_e:
            if st.button("✏️ Edit Draft", use_container_width=True):
                st.session_state["editing"] = True
                st.rerun()

        with col_r:
            if st.button("🚫 Reject", use_container_width=True):
                st.session_state["rejecting"] = True
                st.rerun()

        # Edit sub-form
        if st.session_state.get("editing"):
            st.markdown("---")
            st.markdown("**Edit the draft before approving:**")
            new_subj = st.text_input("Subject", value=drafted.subject, key="edit_subj")
            new_body = st.text_area("Body", value=drafted.body, height=250, key="edit_body")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Approve edited draft", type="primary"):
                    from agent.models import DraftedEmail as DE
                    edited = DE(
                        lead_id=drafted.lead_id,
                        subject=new_subj,
                        body=new_body,
                        grounded_facts=drafted.grounded_facts,
                        tone=drafted.tone,
                    )
                    _do_approve(p, edited, lead, cls, icp_score, edited=True)
                    st.session_state["editing"] = False
                    st.rerun()
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state["editing"] = False
                    st.rerun()

        # Reject sub-form
        if st.session_state.get("rejecting"):
            st.markdown("---")
            reason = st.text_input("Rejection reason (required):", key="reject_reason")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Confirm rejection", type="primary"):
                    if not reason.strip():
                        st.error("A rejection reason is required.")
                    else:
                        _do_reject(p, lead, cls, icp_score, reason)
                        st.session_state["rejecting"] = False
                        st.rerun()
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state["rejecting"] = False
                    st.rerun()

    elif approval_state in ("approved", "edited"):
        send_result = p["send_result"]
        label = "approved with edits" if approval_state == "edited" else "approved"
        st.success(f"✅ Email **{label}** and sent.")
        st.markdown(f"**Message ID:** `{send_result['message_id']}`")
        st.markdown(f"**To:** {send_result['to_name']} `<{send_result['to_email']}>`")
        st.markdown(f"**Subject:** {send_result['subject']}")
        crm = p["crm_record"]
        if crm:
            st.caption(f"CRM record `{crm.crm_record_id}` — tier=HOT, send=sent")

    elif approval_state == "rejected":
        approval = p["approval_result"]
        st.error(f"🚫 Email **rejected** by `{approval.rep_id}`.")
        st.markdown(f"**Reason:** {approval.rejection_reason}")
        crm = p["crm_record"]
        if crm:
            st.caption(f"CRM record `{crm.crm_record_id}` — tier=HOT, send=rejected_by_rep")


def _do_approve(p: dict, email_to_send, lead, cls, icp_score, *, edited: bool) -> None:
    """Execute approval: generate token, call send_email, write_crm, update audit."""
    from agent.models import ApprovalResult
    from tools.email_send import send_email
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    token = str(uuid.uuid4())
    approval = ApprovalResult(
        approved=True,
        edited_email=email_to_send if edited else None,
        approval_token=token,
        rep_id="ui-rep",
        decision="approved_with_edits" if edited else "approved",
        rejection_reason=None,
    )
    send_result = send_email(email_to_send, lead, token)
    action = "hot_sent"
    crm_record = write_crm(lead, cls, action, icp_score, send_result=send_result, approval=approval)

    p["approval_state"] = "edited" if edited else "approved"
    p["approval_result"] = approval
    p["send_result"] = send_result
    p["crm_record"] = crm_record
    p["audit_payload"]["approval_decision"] = approval.model_dump(mode="json")
    p["audit_payload"]["send_result"] = send_result
    p["audit_payload"]["crm_record"] = crm_record.model_dump(mode="json")
    p["audit_payload"]["status"] = "email_sent"
    append_audit_record(lead_id=lead.lead_id, record={"stage": "approval", **p["audit_payload"]})


def _do_reject(p: dict, lead, cls, icp_score, reason: str) -> None:
    from agent.models import ApprovalResult
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    approval = ApprovalResult(
        approved=False,
        edited_email=None,
        approval_token=None,
        rep_id="ui-rep",
        decision="rejected",
        rejection_reason=reason,
    )
    crm_record = write_crm(lead, cls, "hot_rejected", icp_score, approval=approval)

    p["approval_state"] = "rejected"
    p["approval_result"] = approval
    p["crm_record"] = crm_record
    p["audit_payload"]["approval_decision"] = approval.model_dump(mode="json")
    p["audit_payload"]["crm_record"] = crm_record.model_dump(mode="json")
    p["audit_payload"]["status"] = "email_rejected_by_rep"
    append_audit_record(lead_id=lead.lead_id, record={"stage": "rejection", **p["audit_payload"]})


def render_nurture_path(p: dict) -> None:
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    seg = p["nurture_segment"]
    lead = p["lead"]
    cls = p["classification"]
    icp_score = p["icp_score"]

    _stage_header("Stage 5 — Nurture Segmentation", "🌱")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Track", seg.nurture_track)
        st.metric("Sequence", seg.sequence_id)
    with col2:
        st.metric("Label", seg.track_label)
        st.metric("Re-score in", f"{seg.re_score_days} days")
    with col3:
        st.caption(f"**Track reason:** {seg.track_reason}")

    st.info("📭 **No email drafted or sent** — NURTURE leads enter the CRM sequence only.")

    # Write CRM record once (guard so it only runs once per session)
    if p.get("crm_record") is None:
        crm = write_crm(lead, cls, "nurture", icp_score, nurture_segment=seg)
        p["crm_record"] = crm
        p["audit_payload"]["nurture_segment"] = seg.model_dump(mode="json")
        p["audit_payload"]["crm_record"] = crm.model_dump(mode="json")
        p["audit_payload"]["status"] = "crm_nurture_enrolled"
        append_audit_record(lead_id=lead.lead_id, record={"stage": "nurture_crm", **p["audit_payload"]})

    crm = p["crm_record"]
    if crm:
        st.caption(f"CRM record `{crm.crm_record_id}` written — track `{crm.nurture_track}`")


def render_disqualify_path(p: dict) -> None:
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    lead = p["lead"]
    cls = p["classification"]
    icp_score = p["icp_score"]

    _stage_header("Stage 5 — Disqualified", "🚫")
    st.error("**No email drafted, no email sent, no nurture sequence.** "
             "This lead has been archived in the CRM.")
    st.markdown(f"**Reason:** {cls.reason}")

    if p.get("crm_record") is None:
        crm = write_crm(lead, cls, "archive", icp_score)
        p["crm_record"] = crm
        p["audit_payload"]["crm_record"] = crm.model_dump(mode="json")
        p["audit_payload"]["status"] = "archived_disqualified"
        append_audit_record(lead_id=lead.lead_id, record={"stage": "disqualify_crm", **p["audit_payload"]})

    crm = p["crm_record"]
    if crm:
        st.caption(f"CRM record `{crm.crm_record_id}` — archived, no outreach.")


def render_audit_trail(p: dict) -> None:
    """Show the live audit_log.jsonl entries for this lead."""
    from governance.audit_log import read_audit_records

    lead = p["lead"]
    with st.expander("📋 Live Audit Trail (from audit_log.jsonl)", expanded=False):
        records = read_audit_records(lead_id=lead.lead_id)
        if not records:
            st.caption("No audit records found yet for this lead.")
            return
        st.caption(f"{len(records)} audit record(s) for lead `{lead.lead_id}`")
        for rec in records:
            ts = rec.get("logged_at", "")[:19].replace("T", " ")
            stage = rec.get("stage", "—")
            status = rec.get("status", "")
            tier = rec.get("classification", {}).get("tier", "") if isinstance(rec.get("classification"), dict) else ""
            score_raw = rec.get("icp_score", {})
            score = score_raw.get("score", "") if isinstance(score_raw, dict) else ""
            summary = " · ".join(filter(None, [stage, tier, f"score={score}" if score else "", status]))
            st.markdown(f'<div class="audit-entry">{ts} UTC &nbsp;|&nbsp; {summary}</div>', unsafe_allow_html=True)
        with st.expander("Raw JSON (last record)"):
            st.json(records[-1])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict | None:
    """
    Renders the sidebar with lead selector / manual entry form.
    Returns the lead dict to run, or None if no run requested yet.
    """
    st.sidebar.title("🎯 Lead Agent")
    st.sidebar.markdown("---")

    leads = _load_sample_leads()
    # Build display labels for the dropdown
    labels = []
    for ld in leads:
        tier_hint = ld.get("_case", "").split("—")[0].strip()
        labels.append(f"{ld['name']} — {ld['company']} ({tier_hint})")

    mode = st.sidebar.radio("Input mode", ["Sample lead", "Manual entry"], index=0)
    lead_to_run: dict | None = None

    if mode == "Sample lead":
        idx = st.sidebar.selectbox("Choose lead", range(len(leads)), format_func=lambda i: labels[i])
        selected = leads[idx]
        st.sidebar.markdown(f"**Email:** {selected['email']}")
        st.sidebar.markdown(f"**Role:** {selected['role']}")
        if selected.get("_case"):
            st.sidebar.caption(selected["_case"])
        if st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True):
            lead_to_run = selected

    else:
        st.sidebar.markdown("**Enter lead details:**")
        name = st.sidebar.text_input("Full name", value="Jane Smith")
        email = st.sidebar.text_input("Work email", value="jane@example.com")
        company = st.sidebar.text_input("Company", value="Acme Corp")
        role = st.sidebar.text_input("Role / job title", value="VP of Data")
        source = st.sidebar.selectbox("Source", ["inbound_form", "linkedin", "webinar", "conference", "blog_post"])
        form_text = st.sidebar.text_area("Form text / notes (optional)", height=100)
        if st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True):
            lead_to_run = {
                "name": name, "email": email, "company": company,
                "role": role, "source": source,
                "form_text": form_text if form_text.strip() else None,
            }

    st.sidebar.markdown("---")

    # Eval suite runner
    st.sidebar.markdown("**Evaluation Suite**")
    if st.sidebar.button("🧪 Run Eval Suite", use_container_width=True):
        st.session_state["run_eval"] = True

    st.sidebar.markdown("---")
    st.sidebar.caption("Lead Qualification & Outreach Agent · Step 10")

    return lead_to_run


# ---------------------------------------------------------------------------
# Eval runner panel
# ---------------------------------------------------------------------------

def render_eval_panel() -> None:
    """Run eval/run_eval.py and display the PASS/FAIL report inline."""
    st.subheader("🧪 Eval Suite Results")
    placeholder = st.empty()
    placeholder.info("Running eval suite — this takes a few seconds…")

    buf = io.StringIO()
    try:
        # Suppress crm_write print noise during eval
        import contextlib
        import eval.run_eval as _eval_mod
        with contextlib.redirect_stdout(buf):
            _eval_mod.main()
        output = buf.getvalue()
        # Load the written JSON report for structured display
        report_path = _PROJECT_ROOT / "eval" / "eval_report.json"
        with report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)

        s = report["summary"]
        placeholder.empty()
        overall_ok = s["overall"] == "PASS"
        if overall_ok:
            st.success(f"✅ **{s['overall']}** — {s['passed']}/{s['total']} checks passed ({s['pass_rate']})")
        else:
            st.error(f"❌ **{s['overall']}** — {s['passed']}/{s['total']} checks passed ({s['pass_rate']})")

        # Scenarios table
        st.markdown("**Scenarios**")
        rows = []
        for c in report["scenarios"]:
            icon = "✅" if c["result"] == "PASS" else "❌"
            detail = c["assertions"][0] if c["assertions"] else (c["failures"][0] if c["failures"] else "")
            rows.append({"Check": c["name"], "Result": f"{icon} {c['result']}", "Detail": detail[:100]})
        st.table(rows)

        # Governance table
        st.markdown("**Governance Checks**")
        grows = []
        for c in report["governance_checks"]:
            icon = "✅" if c["result"] == "PASS" else "❌"
            detail = c["assertions"][0] if c["assertions"] else (c["failures"][0] if c["failures"] else "")
            grows.append({"Check": c["name"], "Result": f"{icon} {c['result']}", "Detail": detail[:100]})
        st.table(grows)

        with st.expander("Full eval_report.md"):
            md_path = _PROJECT_ROOT / "eval" / "eval_report.md"
            st.markdown(md_path.read_text(encoding="utf-8"))

    except SystemExit:
        # run_eval calls sys.exit(1) on failure — catch it, report was still written
        report_path = _PROJECT_ROOT / "eval" / "eval_report.json"
        if report_path.exists():
            with report_path.open(encoding="utf-8") as fh:
                report = json.load(fh)
            s = report["summary"]
            placeholder.empty()
            st.error(f"❌ **FAIL** — {s['passed']}/{s['total']} checks passed")
    except Exception as exc:
        placeholder.empty()
        st.error(f"Eval runner error: {exc}")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Sidebar — returns lead_data if Run Pipeline was clicked ──────────────
    lead_to_run = render_sidebar()

    # ── Eval panel (shown when requested, replaces pipeline view) ────────────
    if st.session_state.get("run_eval"):
        render_eval_panel()
        if st.button("← Back to pipeline"):
            del st.session_state["run_eval"]
            st.rerun()
        return

    # ── Trigger pipeline run ─────────────────────────────────────────────────
    if lead_to_run is not None:
        # Clear stale editing/rejecting flags from previous run
        for key in ("editing", "rejecting"):
            st.session_state.pop(key, None)
        with st.spinner("Running pipeline…"):
            run_pipeline(lead_to_run)

    # ── Render results ────────────────────────────────────────────────────────
    p: dict | None = st.session_state.get("pipeline")

    if p is None:
        st.title("🎯 Lead Qualification & Outreach Agent")
        st.markdown(
            "Select a sample lead from the sidebar (or enter one manually) "
            "and click **Run Pipeline** to see the full qualification flow."
        )
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔥 HOT leads")
            st.markdown("Strong ICP fit → email drafted → approval gate → send")
        with col2:
            st.markdown("### 🌱 NURTURE leads")
            st.markdown("Moderate fit → CRM sequence → no email")
        with col3:
            st.markdown("### 🚫 DISQUALIFY leads")
            st.markdown("Poor fit → archived → no email ever")
        return

    # Header
    lead = p["lead"]
    tier_val = p["tier"].value
    st.title(f"{_TIER_EMOJI.get(tier_val, '')} {lead.name}")
    col_h1, col_h2, col_h3 = st.columns([2, 2, 1])
    with col_h1:
        st.markdown(f"**{lead.company}** · {lead.role}")
        st.caption(f"Lead ID: `{lead.lead_id}`")
    with col_h2:
        st.markdown(f"📧 `{lead.email}` · source: `{lead.source}`")
        if lead.form_text:
            preview = lead.form_text[:120] + ("…" if len(lead.form_text) > 120 else "")
            st.caption(f"Form text: {preview}")
    with col_h3:
        st.markdown(_tier_badge(tier_val), unsafe_allow_html=True)
        st.metric("ICP Score", f"{p['icp_score'].score:.1f}")

    st.markdown("---")

    # Pipeline stages — always shown
    render_injection(p)
    st.markdown("")
    render_enrichment(p)
    st.markdown("")
    render_scoring(p)
    st.markdown("")
    render_classification(p)
    st.markdown("")
    render_fairness(p)
    st.markdown("")

    # Tier-specific downstream
    from agent.models import LeadTier
    tier = p["tier"]
    if tier == LeadTier.HOT:
        render_hot_path(p)
    elif tier == LeadTier.NURTURE:
        render_nurture_path(p)
    else:
        render_disqualify_path(p)

    st.markdown("")
    render_audit_trail(p)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:   # Streamlit always executes module body
    main()
