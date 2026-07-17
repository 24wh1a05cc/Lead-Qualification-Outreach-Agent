"""
ui/app.py
---------
Streamlit front-end for the Lead Qualification & Outreach Agent.
Enhanced UI with sidebar navigation, visual stepper, progressive reveal,
lead-picker cards, approval centerpiece, timeline audit trail, and eval dashboard.

Thin wrapper — all pipeline logic lives in agent/, tools/, governance/.
No pipeline logic is duplicated here.

Usage
-----
    streamlit run ui/app.py
"""

from __future__ import annotations

import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Load environment variables from .env ─────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
AUDIT_LOG_PATH = _PROJECT_ROOT / "audit_log.jsonl"

_TIER_COLOUR = {
    "HOT": "#22c55e",
    "NURTURE": "#f59e0b",
    "DISQUALIFY": "#ef4444",
}
_TIER_BG = {
    "HOT": "#052e16",
    "NURTURE": "#431407",
    "DISQUALIFY": "#450a0a",
}
_TIER_EMOJI = {
    "HOT": "🔥",
    "NURTURE": "🌱",
    "DISQUALIFY": "🚫",
}

# Pipeline stage definitions for the visual stepper
_STAGES = [
    ("🛡️", "Inject"),
    ("🔍", "Enrich"),
    ("📊", "Score"),
    ("🏷️", "Classify"),
    ("⚖️", "Fairness"),
    ("✉️", "Draft"),
    ("🔐", "Approve"),
]

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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Typography / global ─────────────────────────────────────── */
html, body, [class*="css"] { font-size: 15px; }

/* ── Tier badge pills ────────────────────────────────────────── */
.badge-hot      { background:#052e16; color:#86efac; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.92rem;
                  border:1px solid #16a34a; letter-spacing:.5px; }
.badge-nurture  { background:#431407; color:#fde68a; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.92rem;
                  border:1px solid #d97706; letter-spacing:.5px; }
.badge-disq     { background:#450a0a; color:#fca5a5; padding:4px 14px;
                  border-radius:20px; font-weight:700; font-size:0.92rem;
                  border:1px solid #dc2626; letter-spacing:.5px; }

/* ── Stage section header ────────────────────────────────────── */
.stage-header { background:#1e293b; border-left:4px solid #38bdf8;
                padding:8px 16px; border-radius:6px; margin-bottom:10px;
                font-weight:700; font-size:1rem; color:#e2e8f0;
                display:flex; align-items:center; gap:8px; }

/* ── Visual stepper ──────────────────────────────────────────── */
.stepper-wrap { display:flex; align-items:center; gap:0;
                background:#1e293b; border-radius:10px;
                padding:10px 18px; margin-bottom:22px; }
.step-node    { display:flex; flex-direction:column; align-items:center;
                gap:3px; min-width:64px; }
.step-icon    { font-size:1.3rem; }
.step-label   { font-size:0.68rem; color:#94a3b8; white-space:nowrap; }
.step-active  .step-label { color:#38bdf8; font-weight:700; }
.step-done    .step-label { color:#22c55e; }
.step-connector { flex:1; height:2px; background:#334155; margin:0 4px;
                   margin-bottom:16px; }
.step-connector-done { background:#22c55e; }

/* ── Score bar ───────────────────────────────────────────────── */
.score-track { background:#334155; border-radius:8px; height:16px;
               width:100%; overflow:hidden; }
.score-fill  { height:16px; border-radius:8px; transition:width .4s; }

/* ── Lead card (picker) ──────────────────────────────────────── */
.lead-card { background:#1e293b; border:1px solid #334155; border-radius:10px;
             padding:12px 14px; cursor:pointer; transition:border .15s;
             margin-bottom:6px; }
.lead-card:hover { border-color:#38bdf8; }
.lead-card-selected { border-color:#38bdf8 !important;
                      box-shadow:0 0 0 2px #38bdf826; }
.lead-card-name  { font-weight:700; font-size:0.95rem; color:#e2e8f0; }
.lead-card-meta  { font-size:0.78rem; color:#94a3b8; margin-top:2px; }
.lead-card-hint  { font-size:0.72rem; color:#64748b; margin-top:4px;
                   font-style:italic; }

/* ── Persistent header bar ───────────────────────────────────── */
.lead-header { background:#1e293b; border-radius:10px; padding:14px 20px;
               margin-bottom:20px; display:flex; align-items:center;
               justify-content:space-between; border:1px solid #334155; }
.lead-header-left h2 { margin:0; font-size:1.3rem; color:#f1f5f9; }
.lead-header-left p  { margin:0; font-size:0.82rem; color:#94a3b8; }

/* ── Email display block ─────────────────────────────────────── */
.email-card { background:#0f172a; border:1px solid #334155; border-radius:8px;
              padding:18px; margin:10px 0; }
.email-subject { font-size:1rem; font-weight:700; color:#38bdf8;
                 margin-bottom:8px; }
.email-body { font-family: 'Courier New', monospace; font-size:0.85rem;
              color:#e2e8f0; white-space:pre-wrap; line-height:1.6; }

/* ── Outcome confirmation banners ────────────────────────────── */
.outcome-sent     { background:#052e16; border:1px solid #16a34a;
                    border-radius:8px; padding:16px 20px; font-size:1rem;
                    color:#86efac; font-weight:600; }
.outcome-rejected { background:#450a0a; border:1px solid #dc2626;
                    border-radius:8px; padding:16px 20px; font-size:1rem;
                    color:#fca5a5; font-weight:600; }
.outcome-archived { background:#1c1917; border:1px solid #57534e;
                    border-radius:8px; padding:16px 20px; font-size:1rem;
                    color:#a8a29e; font-weight:600; }

/* ── Audit timeline ──────────────────────────────────────────── */
.timeline-row { display:flex; align-items:flex-start; gap:12px;
                padding:8px 0; border-bottom:1px solid #1e293b; }
.timeline-dot { width:10px; height:10px; border-radius:50%;
                background:#38bdf8; margin-top:5px; flex-shrink:0; }
.timeline-ts  { font-size:0.72rem; color:#64748b; min-width:135px;
                font-family:monospace; }
.timeline-stage { font-size:0.8rem; font-weight:600; color:#cbd5e1;
                  min-width:120px; }
.timeline-detail { font-size:0.78rem; color:#94a3b8; }

/* ── Eval dashboard ──────────────────────────────────────────── */
.eval-scorecard { background:#1e293b; border-radius:10px; padding:20px 24px;
                  text-align:center; border:1px solid #334155; }
.eval-big-pass  { font-size:3rem; font-weight:900; color:#22c55e;
                  line-height:1; }
.eval-big-fail  { font-size:3rem; font-weight:900; color:#ef4444;
                  line-height:1; }
.eval-label     { font-size:0.8rem; color:#64748b; margin-top:4px; }
.eval-badge-pass { background:#052e16; color:#86efac; padding:2px 10px;
                   border-radius:12px; font-weight:700; font-size:0.8rem;
                   border:1px solid #16a34a; }
.eval-badge-fail { background:#450a0a; color:#fca5a5; padding:2px 10px;
                   border-radius:12px; font-weight:700; font-size:0.8rem;
                   border:1px solid #dc2626; }

/* ── Check/cross criteria list ───────────────────────────────── */
.crit-row { display:flex; align-items:center; gap:8px;
            padding:5px 0; border-bottom:1px solid #1e293b;
            font-size:0.85rem; }
.crit-pass { color:#22c55e; font-weight:700; font-size:1rem; }
.crit-fail { color:#ef4444; font-weight:700; font-size:1rem; }
.crit-text-pass { color:#d1fae5; }
.crit-text-fail { color:#fecaca; }

/* ── Nav radio buttons (sidebar) ─────────────────────────────── */
div[data-testid="stRadio"] > div { gap:4px; }

/* ── General spacing ─────────────────────────────────────────── */
.section-gap { margin-top:20px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (cached)
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
# Shared HTML helpers
# ---------------------------------------------------------------------------

def _tier_badge_html(tier_val: str) -> str:
    css = {"HOT": "badge-hot", "NURTURE": "badge-nurture", "DISQUALIFY": "badge-disq"}.get(tier_val, "badge-disq")
    emoji = _TIER_EMOJI.get(tier_val, "")
    return f'<span class="{css}">{emoji} {tier_val}</span>'


def _score_bar_html(score: float) -> str:
    colour = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    pct = min(int(score), 100)
    return (
        f'<div class="score-track">'
        f'<div class="score-fill" style="width:{pct}%; background:{colour};"></div>'
        f'</div>'
    )


def _stage_header(label: str) -> None:
    st.markdown(f'<div class="stage-header">{label}</div>', unsafe_allow_html=True)


def _stepper_html(active_index: int) -> str:
    """Render the pipeline stepper bar. active_index is the last completed stage (0-based)."""
    parts = ['<div class="stepper-wrap">']
    for i, (icon, label) in enumerate(_STAGES):
        if i < active_index:
            cls = "step-node step-done"
            display_icon = "✅"
        elif i == active_index:
            cls = "step-node step-active"
            display_icon = icon
        else:
            cls = "step-node"
            display_icon = icon
        parts.append(
            f'<div class="{cls}">'
            f'<span class="step-icon">{display_icon}</span>'
            f'<span class="step-label">{label}</span>'
            f'</div>'
        )
        if i < len(_STAGES) - 1:
            conn_cls = "step-connector-done" if i < active_index else ""
            parts.append(f'<div class="step-connector {conn_cls}"></div>')
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(lead_data: dict) -> None:
    """
    Execute the full pipeline for one lead and persist every stage result in
    st.session_state["pipeline"].  All business logic delegated to agent/tools/governance.
    Does NOT call request_approval() — the UI presents the approval UI directly.
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

    clean = {k: v for k, v in lead_data.items() if not k.startswith("_")}
    lead = Lead(**clean)

    inj      = check_for_injection_attempt(lead.form_text or "", lead.lead_id)
    enriched = enrich_lead(lead)
    icp_score = score_lead(enriched, icp)
    classification = classify_lead(enriched, icp_score)
    fairness = run_fairness_check(lead, icp_score, classification)

    tier = classification.tier
    drafted        = None
    nurture_segment = None

    if tier == LeadTier.HOT:
        drafted = draft_email(enriched, classification)
    elif tier == LeadTier.NURTURE:
        nurture_segment = segment_lead(icp_score, classification)

    st.session_state["pipeline"] = {
        "lead":            lead,
        "enriched":        enriched,
        "icp_score":       icp_score,
        "classification":  classification,
        "fairness":        fairness,
        "injection":       inj,
        "drafted":         drafted,
        "nurture_segment": nurture_segment,
        "tier":            tier,
        "approval_state":  "pending",
        "approval_result": None,
        "send_result":     None,
        "crm_record":      None,
        "audit_payload": {
            "stage":         "full_pipeline",
            "pipeline_step": 10,
            "raw_lead":      lead.model_dump(mode="json"),
            "injection_check": inj.model_dump(mode="json"),
            "enriched_lead": enriched.model_dump(mode="json"),
            "icp_score":     icp_score.model_dump(mode="json"),
            "classification": classification.model_dump(mode="json"),
            "fairness_result": fairness.model_dump(mode="json"),
        },
    }
    append_audit_record(
        lead_id=lead.lead_id,
        record=st.session_state["pipeline"]["audit_payload"],
    )


# ---------------------------------------------------------------------------
# Approval helpers (called from approval gate render)
# ---------------------------------------------------------------------------

def _do_approve(p: dict, email_to_send, lead, cls, icp_score, *, edited: bool) -> None:
    from agent.models import ApprovalResult
    from tools.email_send import send_email
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    token    = str(uuid.uuid4())
    approval = ApprovalResult(
        approved=True,
        edited_email=email_to_send if edited else None,
        approval_token=token,
        rep_id="ui-rep",
        decision="approved_with_edits" if edited else "approved",
        rejection_reason=None,
    )
    send_result = send_email(email_to_send, lead, token)
    crm_record  = write_crm(lead, cls, "hot_sent", icp_score,
                             send_result=send_result, approval=approval)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    p["approval_state"]  = "edited" if edited else "approved"
    p["approval_result"] = approval
    p["send_result"]     = send_result
    p["crm_record"]      = crm_record
    p["sent_at"]         = now
    p["audit_payload"]["approval_decision"] = approval.model_dump(mode="json")
    p["audit_payload"]["send_result"]       = send_result
    p["audit_payload"]["crm_record"]        = crm_record.model_dump(mode="json")
    p["audit_payload"]["status"]            = "email_sent"
    append_audit_record(
        lead_id=lead.lead_id,
        record={"stage": "approval", **p["audit_payload"]},
    )


def _do_reject(p: dict, lead, cls, icp_score, reason: str) -> None:
    from agent.models import ApprovalResult
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    approval   = ApprovalResult(
        approved=False, edited_email=None, approval_token=None,
        rep_id="ui-rep", decision="rejected", rejection_reason=reason,
    )
    crm_record = write_crm(lead, cls, "hot_rejected", icp_score, approval=approval)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    p["approval_state"]  = "rejected"
    p["approval_result"] = approval
    p["crm_record"]      = crm_record
    p["rejected_at"]     = now
    p["audit_payload"]["approval_decision"] = approval.model_dump(mode="json")
    p["audit_payload"]["crm_record"]        = crm_record.model_dump(mode="json")
    p["audit_payload"]["status"]            = "email_rejected_by_rep"
    append_audit_record(
        lead_id=lead.lead_id,
        record={"stage": "rejection", **p["audit_payload"]},
    )


# ---------------------------------------------------------------------------
# Stage render functions — each wrapped in st.status() for progressive reveal
# ---------------------------------------------------------------------------

def render_injection(p: dict) -> None:
    inj = p["injection"]
    with st.status("🛡️ Stage 0 — Injection Check", expanded=False, state="complete"):
        if inj.is_suspicious:
            st.warning(
                "⚠️ **Suspicious content detected** in form text. "
                "Patterns flagged and logged — pipeline continues on real firmographic signals only."
            )
            for pat in inj.matched_patterns:
                st.code(pat, language=None)
            st.caption("ℹ️ Injection detection is informational: it never alters routing or scores.")
        else:
            st.success("✅ Form text clean — no injection patterns detected.")
        if lead_ft := (p["lead"].form_text or ""):
            with st.expander("Form text preview"):
                st.caption(lead_ft[:400] + ("…" if len(lead_ft) > 400 else ""))


def render_enrichment(p: dict) -> None:
    enriched = p["enriched"]
    with st.status("🔍 Stage 1 — Enrichment", expanded=False, state="complete"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Industry",      enriched.industry      or "—")
            st.metric("Company size",  enriched.company_size  or "—")
        with col2:
            st.metric("Revenue band",  enriched.annual_revenue_usd or "—")
            st.metric("Location",      enriched.location      or "—")
        with col3:
            conf_pct = int(enriched.enrichment_confidence * 100)
            st.metric("Confidence", f"{conf_pct}%")
            stack = ", ".join(enriched.tech_stack) if enriched.tech_stack else "—"
            st.caption(f"**Tech stack:** {stack}")
        if enriched.buying_signals:
            with st.expander(f"Buying signals detected ({len(enriched.buying_signals)})"):
                for sig in enriched.buying_signals:
                    st.markdown(f"- `{sig}`")
        else:
            st.caption("No buying signals detected.")


def render_scoring(p: dict) -> None:
    icp_score = p["icp_score"]
    with st.status("📊 Stage 2 — ICP Scoring", expanded=True, state="complete"):
        col_score, col_bar = st.columns([1, 3])
        with col_score:
            st.metric("ICP Score", f"{icp_score.score:.1f} / 100")
        with col_bar:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.markdown(_score_bar_html(icp_score.score), unsafe_allow_html=True)

        st.markdown("**ICP criteria breakdown:**")

        # Build unified sorted list: matched first, then unmatched
        matched_set   = set(icp_score.matched_criteria or [])
        unmatched_set = set(icp_score.unmatched_criteria or [])
        all_criteria  = sorted(matched_set | unmatched_set)

        rows_html = []
        for crit in sorted(matched_set):
            rows_html.append(
                f'<div class="crit-row">'
                f'<span class="crit-pass">✓</span>'
                f'<span class="crit-text-pass">{crit}</span>'
                f'</div>'
            )
        for crit in sorted(unmatched_set):
            rows_html.append(
                f'<div class="crit-row">'
                f'<span class="crit-fail">✗</span>'
                f'<span class="crit-text-fail">{crit}</span>'
                f'</div>'
            )
        if rows_html:
            st.markdown("".join(rows_html), unsafe_allow_html=True)

        if icp_score.scoring_notes:
            st.caption(f"ℹ️ {icp_score.scoring_notes}")


def render_classification(p: dict) -> None:
    cls      = p["classification"]
    tier_val = cls.tier.value
    with st.status("🏷️ Stage 3 — Classification", expanded=True, state="complete"):
        col_badge, col_detail = st.columns([1, 3])
        with col_badge:
            st.markdown(_tier_badge_html(tier_val), unsafe_allow_html=True)
            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.metric("Score used", f"{cls.icp_score_used:.1f}")
        with col_detail:
            st.markdown(f"**Reason:** {cls.reason}")
            if cls.cited_signals:
                chips = " &nbsp; ".join(
                    f'<code style="font-size:0.78rem">{s}</code>'
                    for s in cls.cited_signals[:5]
                )
                st.markdown(f"**Cited signals:** {chips}", unsafe_allow_html=True)


def render_fairness(p: dict) -> None:
    f = p["fairness"]
    passed = f.passed
    with st.status("⚖️ Stage 4 — Fairness Check", expanded=False, state="complete"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if passed:
                st.markdown(
                    '<span class="eval-badge-pass">✅ PASS</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="eval-badge-fail">❌ FAIL</span>',
                    unsafe_allow_html=True,
                )
            st.caption("Identity-blind re-score")
        with col2:
            st.metric("Original score",    f"{f.original_score:.1f}")
            st.metric("Anonymised score",  f"{f.anonymized_score:.1f}")
        with col3:
            st.metric("Original tier",    f.original_tier)
            st.metric("Anonymised tier",  f.anonymized_tier)
        if not passed and f.discrepancy_details:
            st.error(f"⚠️ Discrepancy: {f.discrepancy_details}")
        else:
            st.caption("Name/email anonymised; same firmographics → same score & tier ✓")


# ---------------------------------------------------------------------------
# Tier-specific downstream render functions
# ---------------------------------------------------------------------------

def render_hot_path(p: dict) -> None:
    """HOT path: draft display + approval gate (visual centerpiece)."""
    drafted        = p["drafted"]
    lead           = p["lead"]
    cls            = p["classification"]
    icp_score      = p["icp_score"]
    approval_state = p["approval_state"]

    # ── Draft display ─────────────────────────────────────────────────────────
    with st.status("✉️ Stage 5 — Email Draft", expanded=True, state="complete"):
        st.markdown(
            f'<div class="email-card">'
            f'<div class="email-subject">Subject: {drafted.subject}</div>'
            f'<div class="email-body">{drafted.body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"🔗 Grounded facts ({len(drafted.grounded_facts)}) — no hallucination"):
            for fact in drafted.grounded_facts:
                st.markdown(f"- `{fact}`")

    # ── Approval gate — VISUAL CENTERPIECE ────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "### 🔐 Stage 6 — Approval Gate",
    )

    if approval_state == "pending":
        st.info(
            "📋 **Email is held for human approval.** "
            "This is a hard gate — no email is sent without explicit rep action.",
            icon="🛑",
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

        col_a, col_e, col_r = st.columns(3, gap="large")

        with col_a:
            st.markdown("#### ✅ Approve & Send")
            st.caption("Send the drafted email exactly as written.")
            if st.button("✅ Approve & Send", use_container_width=True,
                         type="primary", key="btn_approve"):
                with st.spinner("Sending…"):
                    _do_approve(p, drafted, lead, cls, icp_score, edited=False)
                st.rerun()

        with col_e:
            st.markdown("#### ✏️ Edit then Approve")
            st.caption("Modify subject or body before sending.")
            if st.button("✏️ Edit Draft", use_container_width=True, key="btn_edit"):
                st.session_state["editing"]   = True
                st.session_state["rejecting"] = False
                st.rerun()

        with col_r:
            st.markdown("#### 🚫 Reject")
            st.caption("Reject this email — lead stays in CRM, no outreach.")
            if st.button("🚫 Reject", use_container_width=True, key="btn_reject"):
                st.session_state["rejecting"] = True
                st.session_state["editing"]   = False
                st.rerun()

        # ── Edit sub-form ──────────────────────────────────────────────────────
        if st.session_state.get("editing"):
            st.markdown("---")
            st.markdown("#### ✏️ Edit draft before approving")
            new_subj = st.text_input("Subject line", value=drafted.subject, key="edit_subj")
            new_body = st.text_area("Email body", value=drafted.body, height=280, key="edit_body")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Approve edited draft", type="primary", use_container_width=True):
                    from agent.models import DraftedEmail as DE
                    edited_draft = DE(
                        lead_id=drafted.lead_id, subject=new_subj, body=new_body,
                        grounded_facts=drafted.grounded_facts, tone=drafted.tone,
                    )
                    with st.spinner("Sending edited draft…"):
                        _do_approve(p, edited_draft, lead, cls, icp_score, edited=True)
                    st.session_state["editing"] = False
                    st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["editing"] = False
                    st.rerun()

        # ── Reject sub-form ────────────────────────────────────────────────────
        if st.session_state.get("rejecting"):
            st.markdown("---")
            st.markdown("#### 🚫 Reject this email")
            reason = st.text_input("Rejection reason (required):", key="reject_reason")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm rejection", type="primary", use_container_width=True):
                    if not reason.strip():
                        st.error("A rejection reason is required.")
                    else:
                        _do_reject(p, lead, cls, icp_score, reason)
                        st.session_state["rejecting"] = False
                        st.rerun()
            with c2:
                if st.button("Cancel", use_container_width=True, key="cancel_reject"):
                    st.session_state["rejecting"] = False
                    st.rerun()

    # ── Post-decision outcome banner ───────────────────────────────────────────
    elif approval_state in ("approved", "edited"):
        sr    = p["send_result"]
        label = "approved with edits" if approval_state == "edited" else "approved"
        ts    = p.get("sent_at", "—")
        st.markdown(
            f'<div class="outcome-sent">'
            f'✅ Email <strong>{label}</strong> and sent by <code>ui-rep</code> at {ts}<br>'
            f'&nbsp;&nbsp;📨 Message ID: <code>{sr["message_id"]}</code><br>'
            f'&nbsp;&nbsp;To: {sr["to_name"]} &lt;{sr["to_email"]}&gt;<br>'
            f'&nbsp;&nbsp;Subject: {sr["subject"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        crm = p["crm_record"]
        if crm:
            st.caption(f"CRM record `{crm.crm_record_id}` — tier=HOT · send_status=sent")

    elif approval_state == "rejected":
        approval = p["approval_result"]
        ts = p.get("rejected_at", "—")
        st.markdown(
            f'<div class="outcome-rejected">'
            f'❌ Email <strong>rejected</strong> by <code>{approval.rep_id}</code> at {ts}<br>'
            f'&nbsp;&nbsp;Reason: {approval.rejection_reason}<br>'
            f'&nbsp;&nbsp;Lead remains in CRM — no outreach sent.'
            f'</div>',
            unsafe_allow_html=True,
        )
        crm = p["crm_record"]
        if crm:
            st.caption(f"CRM record `{crm.crm_record_id}` — tier=HOT · send_status=rejected_by_rep")


def render_nurture_path(p: dict) -> None:
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    seg       = p["nurture_segment"]
    lead      = p["lead"]
    cls       = p["classification"]
    icp_score = p["icp_score"]

    with st.status("🌱 Stage 5 — Nurture Segmentation", expanded=True, state="complete"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Track ID",   seg.nurture_track)
            st.metric("Sequence",   seg.sequence_id)
        with col2:
            st.metric("Track label", seg.track_label)
            st.metric("Re-score in", f"{seg.re_score_days} days")
        with col3:
            st.markdown("**Track reason:**")
            st.caption(seg.track_reason)

        st.markdown(
            '<div class="outcome-archived">'
            '📭 No email drafted or sent — NURTURE leads enter the CRM sequence only.'
            '</div>',
            unsafe_allow_html=True,
        )

        if p.get("crm_record") is None:
            crm = write_crm(lead, cls, "nurture", icp_score, nurture_segment=seg)
            p["crm_record"] = crm
            p["audit_payload"]["nurture_segment"] = seg.model_dump(mode="json")
            p["audit_payload"]["crm_record"]      = crm.model_dump(mode="json")
            p["audit_payload"]["status"]          = "crm_nurture_enrolled"
            append_audit_record(
                lead_id=lead.lead_id,
                record={"stage": "nurture_crm", **p["audit_payload"]},
            )

        crm = p["crm_record"]
        if crm:
            st.caption(
                f"CRM record `{crm.crm_record_id}` written — "
                f"track `{crm.nurture_track}` · sequence `{seg.sequence_id}`"
            )


def render_disqualify_path(p: dict) -> None:
    from tools.crm_write import write_crm
    from governance.audit_log import append_audit_record

    lead      = p["lead"]
    cls       = p["classification"]
    icp_score = p["icp_score"]

    with st.status("🚫 Stage 5 — Disqualified", expanded=True, state="complete"):
        st.markdown(
            '<div class="outcome-archived">'
            '🚫 Lead archived — no email drafted, no email sent, no nurture sequence.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Reason:** {cls.reason}", unsafe_allow_html=False)

        if p.get("crm_record") is None:
            crm = write_crm(lead, cls, "archive", icp_score)
            p["crm_record"] = crm
            p["audit_payload"]["crm_record"] = crm.model_dump(mode="json")
            p["audit_payload"]["status"]     = "archived_disqualified"
            append_audit_record(
                lead_id=lead.lead_id,
                record={"stage": "disqualify_crm", **p["audit_payload"]},
            )

        crm = p["crm_record"]
        if crm:
            st.caption(f"CRM record `{crm.crm_record_id}` — archived, no outreach.")


# ---------------------------------------------------------------------------
# Audit Trail — clean timeline render
# ---------------------------------------------------------------------------

def render_audit_trail(p: dict) -> None:
    from governance.audit_log import read_audit_records

    lead    = p["lead"]
    records = read_audit_records(lead_id=lead.lead_id)

    with st.expander(f"📋 Audit Trail — {len(records)} record(s) for `{lead.lead_id}`", expanded=False):
        if not records:
            st.caption("No audit records found for this lead yet.")
            return

        rows_html = ['<div style="padding:4px 0;">']
        for rec in records:
            ts       = (rec.get("logged_at") or "")[:19].replace("T", " ")
            stage    = rec.get("stage", "—")
            status   = rec.get("status", "")
            tier_raw = rec.get("classification", {})
            tier     = tier_raw.get("tier", "") if isinstance(tier_raw, dict) else ""
            score_r  = rec.get("icp_score", {})
            score    = score_r.get("score", "") if isinstance(score_r, dict) else ""
            detail_parts = []
            if tier:
                detail_parts.append(f"tier={tier}")
            if score != "":
                detail_parts.append(f"score={score:.1f}" if isinstance(score, float) else f"score={score}")
            if status:
                detail_parts.append(status)
            detail = " · ".join(detail_parts) if detail_parts else "—"

            rows_html.append(
                f'<div class="timeline-row">'
                f'<div class="timeline-dot"></div>'
                f'<div class="timeline-ts">{ts} UTC</div>'
                f'<div class="timeline-stage">{stage}</div>'
                f'<div class="timeline-detail">{detail}</div>'
                f'</div>'
            )
        rows_html.append("</div>")
        st.markdown("".join(rows_html), unsafe_allow_html=True)

        with st.expander("🔍 Raw JSON — last audit record (debug)"):
            st.json(records[-1])


# ---------------------------------------------------------------------------
# Eval Dashboard (full page)
# ---------------------------------------------------------------------------

def render_eval_dashboard() -> None:
    st.markdown("## 🧪 Eval Suite Dashboard")
    st.markdown("Automated correctness, governance, and safety checks across all 11 sample leads.")
    st.markdown("---")

    report_path = _PROJECT_ROOT / "eval" / "eval_report.json"

    # ── Re-run button ──────────────────────────────────────────────────────────
    col_btn, col_spacer = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("▶ Re-run Eval Suite", type="primary", use_container_width=True)

    if run_btn or (st.session_state.get("run_eval") and not report_path.exists()):
        placeholder = st.empty()
        placeholder.info("⏳ Running eval suite — this takes ~10 seconds…")
        buf = io.StringIO()
        try:
            import contextlib
            import eval.run_eval as _eval_mod
            with contextlib.redirect_stdout(buf):
                _eval_mod.main()
        except SystemExit:
            pass
        except Exception as exc:
            placeholder.error(f"Eval runner error: {exc}")
            return
        placeholder.empty()
        st.rerun()

    # ── Load report (fall back to committed seed on a fresh deploy) ───────────
    seed_path = _PROJECT_ROOT / "eval" / "eval_report.seed.json"
    if not report_path.exists():
        if seed_path.exists():
            report_path = seed_path
            st.caption("ℹ️ Showing committed seed report — click **Re-run Eval Suite** to generate a live report.")
        else:
            st.info("No eval report found. Click **Re-run Eval Suite** to generate one.")
            return

    with report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)

    s           = report["summary"]
    total       = s["total"]
    passed      = s["passed"]
    failed      = total - passed
    overall_ok  = s["overall"] == "PASS"
    pass_rate   = s["pass_rate"]

    # ── Scorecard row ──────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        val_class = "eval-big-pass" if overall_ok else "eval-big-fail"
        icon      = "✅" if overall_ok else "❌"
        st.markdown(
            f'<div class="eval-scorecard">'
            f'<div class="{val_class}">{icon}</div>'
            f'<div class="eval-label">Overall</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f'<div class="eval-scorecard">'
            f'<div class="eval-big-pass">{passed}</div>'
            f'<div class="eval-label">PASSED</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc3:
        fail_class = "eval-big-fail" if failed > 0 else "eval-big-pass"
        st.markdown(
            f'<div class="eval-scorecard">'
            f'<div class="{fail_class}">{failed}</div>'
            f'<div class="eval-label">FAILED</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with sc4:
        st.markdown(
            f'<div class="eval-scorecard">'
            f'<div class="eval-big-pass">{pass_rate}</div>'
            f'<div class="eval-label">Pass rate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Scenario table ─────────────────────────────────────────────────────────
    st.markdown("### Scenarios")
    for c in report.get("scenarios", []):
        ok     = c["result"] == "PASS"
        badge  = '<span class="eval-badge-pass">✅ PASS</span>' if ok else '<span class="eval-badge-fail">❌ FAIL</span>'
        detail = (c["assertions"][0] if c.get("assertions") else
                  c["failures"][0]   if c.get("failures")   else "—")
        with st.expander(f"{c['name']} — {c['result']}"):
            st.markdown(badge, unsafe_allow_html=True)
            st.caption(detail)
            if c.get("failures"):
                for f in c["failures"]:
                    st.error(f)

    # ── Governance table ───────────────────────────────────────────────────────
    st.markdown("### Governance Checks")
    for c in report.get("governance_checks", []):
        ok     = c["result"] == "PASS"
        badge  = '<span class="eval-badge-pass">✅ PASS</span>' if ok else '<span class="eval-badge-fail">❌ FAIL</span>'
        detail = (c["assertions"][0] if c.get("assertions") else
                  c["failures"][0]   if c.get("failures")   else "—")
        with st.expander(f"{c['name']} — {c['result']}"):
            st.markdown(badge, unsafe_allow_html=True)
            st.caption(detail)
            if c.get("failures"):
                for f in c["failures"]:
                    st.error(f)

    # ── Full markdown report ───────────────────────────────────────────────────
    md_path = _PROJECT_ROOT / "eval" / "eval_report.md"
    if md_path.exists():
        with st.expander("📄 Full eval_report.md"):
            st.markdown(md_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sidebar — navigation + lead picker
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, dict | None]:
    """
    Returns (page, lead_to_run).
    page: "pipeline" | "eval" | "audit"
    lead_to_run: lead dict if Run Pipeline was clicked, else None
    """
    st.sidebar.markdown(
        '<div style="text-align:center; padding:10px 0 6px;">'
        '<span style="font-size:2rem;">🎯</span><br>'
        '<span style="font-weight:800; font-size:1.1rem; color:#e2e8f0;">Lead Agent</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── LLM mode indicator ────────────────────────────────────────────────────
    from llm_client import is_llm_enabled, get_model
    if is_llm_enabled():
        st.sidebar.success(f"🤖 LLM mode · `{get_model()}`")
    else:
        st.sidebar.info("📋 Template mode · set OPENROUTER_API_KEY in .env to enable LLM drafting")

    st.sidebar.markdown("---")

    # ── Top-level navigation ──────────────────────────────────────────────────
    page = st.sidebar.radio(
        "Navigate",
        options=["🚀 Run Pipeline", "🧪 Eval Dashboard", "📋 Audit Trail"],
        index=0,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    lead_to_run: dict | None = None

    if page == "🚀 Run Pipeline":
        leads = _load_sample_leads()

        input_mode = st.sidebar.radio("Lead source", ["Sample leads", "Custom lead"], index=0)

        if input_mode == "Sample leads":
            st.sidebar.markdown("**Click a lead to select it:**")

            # Lead card picker
            selected_idx = st.session_state.get("selected_lead_idx", 0)
            for i, ld in enumerate(leads):
                case_hint = ld.get("_case", "")
                # Shorten case hint: take text before " — " if present
                short_hint = case_hint.split(" — ")[0] if " — " in case_hint else case_hint
                short_hint = short_hint[:55] + "…" if len(short_hint) > 55 else short_hint
                is_selected = (i == selected_idx)
                card_cls = "lead-card lead-card-selected" if is_selected else "lead-card"
                # Render card as a clickable button-ish block using st.button
                clicked = st.sidebar.button(
                    f"{'→ ' if is_selected else '   '}{ld['name']} · {ld['company']}\n{ld['role']}",
                    key=f"lead_card_{i}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary",
                )
                if clicked:
                    st.session_state["selected_lead_idx"] = i
                    st.session_state.pop("editing", None)
                    st.session_state.pop("rejecting", None)
                    st.rerun()

            st.sidebar.markdown("---")

            # Show selected lead details
            sel = leads[selected_idx]
            st.sidebar.markdown(f"**Selected:** {sel['name']}")
            st.sidebar.caption(f"📧 {sel['email']}")
            st.sidebar.caption(f"🏢 {sel['company']} · {sel['role']}")
            if sel.get("_case"):
                st.sidebar.caption(f"ℹ️ {sel['_case'][:80]}…" if len(sel.get("_case","")) > 80 else f"ℹ️ {sel['_case']}")

            st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)
            if st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True, key="run_sample"):
                lead_to_run = sel

        else:  # Custom lead
            st.sidebar.markdown("**Enter lead details:**")
            name      = st.sidebar.text_input("Full name",    value="Jane Smith")
            email     = st.sidebar.text_input("Work email",   value="jane@example.com")
            company   = st.sidebar.text_input("Company",      value="Acme Corp")
            role      = st.sidebar.text_input("Role",         value="VP of Data")
            source    = st.sidebar.selectbox(
                "Source", ["inbound_form", "linkedin", "webinar", "conference", "blog_post"]
            )
            form_text = st.sidebar.text_area("Form text / notes (optional)", height=90)

            # Disable run button if required fields missing
            ready = bool(name.strip() and email.strip() and company.strip())
            if not ready:
                st.sidebar.caption("⚠️ Name, email, and company are required.")
            if st.sidebar.button(
                "▶ Run Pipeline",
                type="primary",
                use_container_width=True,
                disabled=not ready,
                key="run_custom",
            ):
                lead_to_run = {
                    "name": name, "email": email, "company": company,
                    "role": role, "source": source,
                    "form_text": form_text.strip() or None,
                }

    st.sidebar.markdown("---")
    st.sidebar.caption("Lead Qualification & Outreach Agent · v1.0")

    return page, lead_to_run


# ---------------------------------------------------------------------------
# Pipeline page — persistent header + stepper + stage cards
# ---------------------------------------------------------------------------

def render_pipeline_page(lead_to_run: dict | None) -> None:
    from agent.models import LeadTier

    # Trigger pipeline run
    if lead_to_run is not None:
        for key in ("editing", "rejecting"):
            st.session_state.pop(key, None)
        with st.spinner("⚙️ Running pipeline…"):
            run_pipeline(lead_to_run)

    p: dict | None = st.session_state.get("pipeline")

    # ── Empty state ────────────────────────────────────────────────────────────
    if p is None:
        st.markdown(
            '<h1 style="font-size:2rem; margin-bottom:4px;">🎯 Lead Qualification & Outreach Agent</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "Select a sample lead from the sidebar and click **▶ Run Pipeline** to see the full qualification flow.",
            unsafe_allow_html=False,
        )
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="eval-scorecard">'
                '<div style="font-size:2rem;">🔥</div>'
                '<div style="font-weight:700; color:#22c55e; margin-top:4px;">HOT</div>'
                '<div class="eval-label">Strong ICP fit → email drafted → approval gate → send</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="eval-scorecard">'
                '<div style="font-size:2rem;">🌱</div>'
                '<div style="font-weight:700; color:#f59e0b; margin-top:4px;">NURTURE</div>'
                '<div class="eval-label">Moderate fit → CRM sequence enrolled → no email</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="eval-scorecard">'
                '<div style="font-size:2rem;">🚫</div>'
                '<div style="font-weight:700; color:#ef4444; margin-top:4px;">DISQUALIFY</div>'
                '<div class="eval-label">Poor ICP fit → archived → no email ever</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        return

    # ── Persistent header bar ──────────────────────────────────────────────────
    lead     = p["lead"]
    tier_val = p["tier"].value
    score    = p["icp_score"].score

    tier_colour = _TIER_COLOUR.get(tier_val, "#94a3b8")
    st.markdown(
        f'<div class="lead-header">'
        f'<div class="lead-header-left">'
        f'<h2>{_TIER_EMOJI.get(tier_val,"")} {lead.name}</h2>'
        f'<p>{lead.company} · {lead.role} · '
        f'<code style="font-size:0.75rem">{lead.email}</code></p>'
        f'</div>'
        f'<div style="text-align:right; display:flex; flex-direction:column; gap:6px; align-items:flex-end;">'
        f'{_tier_badge_html(tier_val)}'
        f'<span style="font-size:0.82rem; color:#94a3b8;">ICP Score: '
        f'<strong style="color:{tier_colour};">{score:.1f}</strong> / 100</span>'
        f'<span style="font-size:0.72rem; color:#64748b;">ID: {lead.lead_id}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Visual stepper (how far through the pipeline) ─────────────────────────
    tier      = p["tier"]
    has_draft = p.get("drafted") is not None
    ap_state  = p.get("approval_state", "pending")

    if tier == LeadTier.HOT and ap_state in ("approved", "edited", "rejected"):
        active_step = 6
    elif tier == LeadTier.HOT and has_draft:
        active_step = 5
    else:
        active_step = 4

    st.markdown(_stepper_html(active_step), unsafe_allow_html=True)

    # ── Stage cards ────────────────────────────────────────────────────────────
    render_injection(p)
    render_enrichment(p)
    render_scoring(p)
    render_classification(p)
    render_fairness(p)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if tier == LeadTier.HOT:
        render_hot_path(p)
    elif tier == LeadTier.NURTURE:
        render_nurture_path(p)
    else:
        render_disqualify_path(p)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    render_audit_trail(p)


# ---------------------------------------------------------------------------
# Audit Trail page (standalone)
# ---------------------------------------------------------------------------

def render_audit_page() -> None:
    from governance.audit_log import read_audit_records

    st.markdown("## 📋 Audit Trail")
    st.markdown("All pipeline audit records from `audit_log.jsonl`. Rendered as a timeline per lead.")
    st.markdown("---")

    all_records = read_audit_records()
    seed_used = False
    if not all_records:
        # On a fresh deploy audit_log.jsonl doesn't exist yet — try the seed.
        seed_path = _PROJECT_ROOT / "audit_log.seed.jsonl"
        if seed_path.exists():
            from governance.audit_log import read_audit_records as _read
            all_records = _read(log_path=seed_path)
            seed_used = True
    if not all_records:
        st.info("No audit records yet — run the pipeline on some leads first.")
        return
    if seed_used:
        st.caption("ℹ️ Showing committed seed audit records — run the pipeline to generate live records.")

    # Group by lead_id
    by_lead: dict[str, list] = {}
    for rec in all_records:
        lid = rec.get("lead_id", "unknown")
        by_lead.setdefault(lid, []).append(rec)

    st.caption(f"{len(all_records)} total record(s) across {len(by_lead)} lead(s)")

    for lead_id, records in by_lead.items():
        # Derive lead name from first record's raw_lead if available
        raw = records[0].get("raw_lead", {})
        display = f"{raw.get('name', lead_id)} — {raw.get('company', '')}" if raw else lead_id

        with st.expander(f"🔖 {display} · `{lead_id}` · {len(records)} record(s)", expanded=False):
            rows_html = []
            for rec in records:
                ts     = (rec.get("logged_at") or "")[:19].replace("T", " ")
                stage  = rec.get("stage", "—")
                status = rec.get("status", "")
                tier_r = rec.get("classification", {})
                tier   = tier_r.get("tier", "") if isinstance(tier_r, dict) else ""
                detail_parts = list(filter(None, [tier, status]))
                detail = " · ".join(detail_parts) if detail_parts else "—"
                rows_html.append(
                    f'<div class="timeline-row">'
                    f'<div class="timeline-dot"></div>'
                    f'<div class="timeline-ts">{ts} UTC</div>'
                    f'<div class="timeline-stage">{stage}</div>'
                    f'<div class="timeline-detail">{detail}</div>'
                    f'</div>'
                )
            st.markdown("".join(rows_html), unsafe_allow_html=True)
            with st.expander("🔍 Raw JSON — last record"):
                st.json(records[-1])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    page, lead_to_run = render_sidebar()

    if page == "🚀 Run Pipeline":
        # Clear eval flag if navigating away
        st.session_state.pop("run_eval", None)
        render_pipeline_page(lead_to_run)

    elif page == "🧪 Eval Dashboard":
        st.session_state["run_eval"] = True
        render_eval_dashboard()

    elif page == "📋 Audit Trail":
        st.session_state.pop("run_eval", None)
        render_audit_page()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:   # Streamlit always executes module body
    main()
