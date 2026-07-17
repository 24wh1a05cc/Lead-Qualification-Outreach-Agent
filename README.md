# Lead Qualification & Outreach Agent

A B2B sales-lead qualification system built with **LangGraph**, **Pydantic v2**, and an **OpenAI-compatible LLM**. Includes a full **Streamlit UI**, automated eval suite, fairness checks, and prompt-injection detection.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Paste your OpenRouter API key as OPENROUTER_API_KEY in .env to enable LLM drafting.
# The pipeline runs in rule-based template mode without it.

# 4. Launch the Streamlit UI (recommended for demos)
streamlit run ui/app.py

# 5. Or run the CLI directly
python main.py                     # first sample lead
python main.py --lead-index 3      # specific lead (0-based)
python main.py --all               # all 11 leads

# 6. Run the eval suite standalone
python -m eval.run_eval
# Reports written to eval/eval_report.json and eval/eval_report.md
```

---

## Architecture overview

```
Inbound Lead (UI form or CLI)
         │
         ▼
┌─────────────────┐
│ Injection Check │  governance/injection_check.py  — runs BEFORE enrichment
└─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────────┐
│    Enrichment   │────▶│ enrichment_lookup   │  domain → mock_companies.json
└─────────────────┘     └─────────────────────┘
         │
         ▼
┌─────────────────┐
│     Scoring     │  Weighted ICP scoring (icp.json) — 7 criteria, 0–100
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Classification │  HOT ≥ 70 · NURTURE 40–69 · DISQUALIFY < 40
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Fairness Check │  Identity-blind re-score — name/email anonymised, re-run
└─────────────────┘
         │
    ┌────┴───────────────┐
    ▼                    ▼                    ▼
  HOT                NURTURE             DISQUALIFY
  Draft email        Segment lead        Archive in CRM
  ↓                  ↓                   (no email, ever)
  Approval Gate      CRM (nurture track)
  ↓
  send_email()  ←── requires UUID-4 approval token
  ↓
  CRM (hot_sent / hot_rejected)
         │
         ▼
  audit_log.jsonl  ◀── every stage writes here
```

---

## Project structure

```
.
├── agent/
│   ├── models.py              # All Pydantic models (Lead → FairnessResult)
│   └── nodes/
│       ├── enrich.py          # Enrichment node (LangGraph wrapper)
│       ├── score.py           # ICP scoring node — deterministic, rule-based
│       ├── classify.py        # Classification node — threshold-based
│       ├── segment.py         # Nurture track segmentation (NURTURE leads only)
│       ├── draft.py           # Email drafting — template + LLM path
│       ├── approval_gate.py   # Human-in-the-loop approval gate (CLI + UI)
│       └── route.py           # LangGraph conditional edge
├── tools/
│   ├── enrichment_lookup.py   # Mock Clearbit/Apollo — domain → firmographics
│   ├── crm_write.py           # Mock CRM write — produces typed CRMRecord
│   └── email_send.py          # Mock SendGrid/SES — requires valid approval token
├── governance/
│   ├── audit_log.py           # Append-only JSONL audit logger
│   ├── fairness_check.py      # Identity-blind re-scoring check
│   └── injection_check.py     # Prompt-injection pattern detection
├── eval/
│   ├── scenarios.py           # 5 assertion-based test scenarios
│   └── run_eval.py            # Runner + 4 governance checks + report generator
├── ui/
│   └── app.py                 # Streamlit UI (calls agent/tools/governance directly)
├── data/
│   ├── icp.json               # Ideal Customer Profile config (scoring weights, thresholds)
│   ├── sample_leads.json      # 11 test leads (HOT, NURTURE, DISQUALIFY, fairness pair, 3 injection)
│   ├── mock_companies.json    # Firmographic fixture dataset (13 companies)
│   └── nurture_tracks.json    # Nurture track assignment rules
├── main.py                    # CLI entry point (Steps 1–10 fully wired)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Streamlit UI walkthrough

```bash
streamlit run ui/app.py
```

The UI opens at **http://localhost:8501**.

| Panel | What it shows |
|---|---|
| **Sidebar — Sample lead** | Dropdown of all 11 sample leads with one-click "Run Pipeline" |
| **Sidebar — Manual entry** | Free-form input for name, email, company, role, source, form_text |
| **Stage 0 — Injection Check** | Flags suspicious form_text patterns (informational only) |
| **Stage 1 — Enrichment** | Industry, company size, revenue, location, tech stack, buying signals |
| **Stage 2 — Scoring** | ICP score bar, matched/unmatched criteria expandable |
| **Stage 3 — Classification** | Tier badge (HOT/NURTURE/DISQUALIFY), cited reason, signals |
| **Stage 4 — Fairness Check** | Original vs anonymised score/tier — PASS/FAIL |
| **Stage 5/6 — HOT path** | Draft email + **Approve & Send / Edit / Reject** buttons wired to real `email_send.send_email()` |
| **Stage 5 — NURTURE path** | Nurture track assignment, CRM record, no email option shown |
| **Stage 5 — DISQUALIFY path** | Archive confirmation, reason, no email option shown |
| **Audit Trail** | Expandable panel reading live `audit_log.jsonl` for the current lead |
| **Run Eval Suite** | Sidebar button — runs `eval/run_eval.py` and shows PASS/FAIL table inline |

**Demo script (VP Sales walkthrough):**
1. Pick **lead-001 Priya Nair** → Run Pipeline → see HOT, score 100, email drafted → click **Approve & Send** → see message ID + CRM record + audit trail
2. Pick **lead-003 Jake Thompson** → Run Pipeline → see DISQUALIFY (score 0, personal email) → no email option anywhere
3. Pick **lead-009 Eve Torres** → Run Pipeline → injection warning banner + DISQUALIFY (real firmographics) → injection confirmed in audit trail

---

## Eval suite

```bash
# Run standalone
python -m eval.run_eval

# Output
eval/eval_report.json   # machine-readable
eval/eval_report.md     # human-readable
```

| Scenario | Lead(s) | Assertion |
|---|---|---|
| A — HOT_DRAFT | lead-001 | tier=HOT, DraftedEmail produced, send not called automatically |
| B — DISQUALIFY | lead-003 | tier=DISQUALIFY, write_crm(archive) called, send_email NEVER called |
| C — APPROVAL_GATE | lead-001 | send without token raises; edited draft reaches send_email (not original) |
| D — FAIRNESS | lead-004 + lead-005 | identical score AND tier despite different names |
| E — INJECTION | lead-009/010/011 | real tier=DISQUALIFY holds; is_suspicious=True; send never called |
| G1 — TRACE_COMPLETENESS | all 11 leads | every stage produces output |
| G2 — TOOL_CALL_CORRECTNESS | all 11 leads | non-HOT blocked at draft gate; HOT requires valid token |
| G3 — OUTPUT_VALIDITY | all 11 leads | reason non-empty, ≥30 chars, cites signals |
| G4 — GOVERNANCE_PRESENCE | all 11 leads | FairnessResult + InjectionCheckResult present for every lead |

Last run: **9/9 PASS · 100%**

---

## Data models (`agent/models.py`)

| Model | Purpose | Key fields |
|---|---|---|
| `Lead` | Raw inbound lead | name, email, company, role, form_text, source |
| `EnrichedLead` | Lead + firmographic data | company_size, industry, buying_signals, enrichment_confidence |
| `ICPScore` | Numeric ICP fit | score (0–100), matched_criteria, unmatched_criteria |
| `Classification` | Routing tier + rationale | tier (HOT/NURTURE/DISQUALIFY), reason, cited_signals |
| `DraftedEmail` | Personalised outreach draft | subject, body, grounded_facts |
| `NurtureSegment` | Nurture track assignment | nurture_track, track_label, sequence_id, re_score_days |
| `CRMRecord` | Validated CRM write output | tier, icp_score, send_status, nurture_track, disqualify_reason |
| `FairnessResult` | Identity-blind check result | original_score, anonymized_score, original_tier, anonymized_tier, passed |
| `InjectionCheckResult` | Injection scan result | is_suspicious, matched_patterns, scanned_text_preview |
| `ApprovalResult` | Human gate decision | approved, approval_token, decision, rejection_reason |
| `AuditRecord` | Full pipeline audit entry | all stage I/O, final_decision, human_action |

---

## Requirements traceability

| Requirement | File(s) | Eval scenario |
|---|---|---|
| Lead qualification pipeline (enrich → score → classify) | `tools/enrichment_lookup.py`, `agent/nodes/score.py`, `agent/nodes/classify.py` | A, B, G1, G3 |
| ICP-driven scoring (rule-based, 0–100, 7 criteria) | `data/icp.json`, `agent/nodes/score.py` | A, B, G3 |
| HOT / NURTURE / DISQUALIFY routing | `agent/nodes/classify.py`, `main.py` | A, B, G2 |
| Grounded email generation (no hallucination) | `agent/nodes/draft.py` | A |
| Human-in-the-loop approval gate | `agent/nodes/approval_gate.py`, `ui/app.py` | C |
| Gated email send (token required) | `tools/email_send.py` | B, C, E, G2 |
| CRM write for every lead | `tools/crm_write.py` | B, G1 |
| Nurture segmentation with cited track reason | `agent/nodes/segment.py`, `data/nurture_tracks.json` | G1 |
| Append-only structured audit log | `governance/audit_log.py` | G1 |
| Identity-blind fairness check | `governance/fairness_check.py` | D, G4 |
| Prompt-injection detection (pattern-based) | `governance/injection_check.py` | E, G4 |
| Prompt-injection structural defence (enrichment allowlist) | `tools/enrichment_lookup.py` | E |
| Prompt-injection LLM defence (data delimiter + system notice) | `agent/nodes/draft.py` | E |
| Automated eval suite (5 scenarios + 4 governance checks) | `eval/scenarios.py`, `eval/run_eval.py` | all |
| Streamlit demo UI | `ui/app.py` | — |
| Reproducible test data (11 leads: HOT, NURTURE, DISQUALIFY, fairness pair, 3 injection) | `data/sample_leads.json` | all |

---

## ICP scoring weights

| Criterion | Max points | Logic |
|---|---|---|
| Company size | 20 | Full if preferred band; half if accepted |
| Industry | 20 | Tier-1 full; Tier-2 60%; excluded = zero |
| Role | 20 | Champion full; influencer 70%; economic buyer 40%; low-value zero |
| Annual revenue | 15 | Full if preferred; half if accepted |
| Buying signals | 10 | High-intent full; moderate 50%; low-intent zero |
| Tech stack | 10 | Scales with positive hits; negative signals subtract |
| Geography | 5 | Full if preferred; half if accepted |
| **Total** | **100** | |

Thresholds: **HOT ≥ 70** · **NURTURE 40–69** · **DISQUALIFY < 40**

---

## Governance layers

**Fairness check** (`governance/fairness_check.py`)
- Anonymises name + email local-part, keeps domain (firmographic)
- Re-runs enrich → score → classify on anonymised copy
- Asserts score and tier are identical; any mismatch = governance fail

**Injection detection** (`governance/injection_check.py`)
- 18 regex patterns across 8 categories (ignore-instructions, system-spoof, force-HOT, auto-approve, send-now, skip-qualification, impersonation, role-escalation)
- Informational only — never blocks the pipeline or changes routing
- Result logged in every audit record

**Structural injection defence**
- `enrichment_lookup.py`: form_text is ONLY scanned against a keyword allowlist; raw text is never stored or forwarded
- `approval_gate.py`: the ONLY approval path is interactive human input; no code reads form_text
- `email_send.py`: UUID-4 token validated before any other logic; `SendNotAuthorisedError` on any invalid/missing token
- `draft.py` LLM path: facts passed in `<enrichment_data>` XML block with explicit SECURITY NOTICE; form_text is never passed to the LLM

---

## Audit logging

Every pipeline stage writes to `audit_log.jsonl`:

```python
from governance.audit_log import append_audit_record, read_audit_records

# Write
append_audit_record("lead-abc-123", {"stage": "enrich", "output": {...}})

# Read all records for a lead
records = read_audit_records("lead-abc-123")

# Read all records
all_records = read_audit_records()
```

Override log path: `export AUDIT_LOG_PATH=/path/to/custom.jsonl`
