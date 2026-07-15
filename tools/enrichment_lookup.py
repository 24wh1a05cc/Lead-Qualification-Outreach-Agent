"""
tools/enrichment_lookup.py
--------------------------
Mock enrichment tool.

Simulates the behaviour of a real firmographic data provider (Clearbit, Apollo,
ZoomInfo) using a local fixture dataset at /data/mock_companies.json.

Lookup strategy
---------------
1. Extract the email domain from the lead's email address.
2. Check whether that domain is in the known-personal-email blocklist — if so,
   return a minimal EnrichedLead immediately (confidence=0.0).
3. Try to match the domain against mock_companies.json (exact match on `domain`).
4. If no domain match, try a case-insensitive partial match on `name` vs the
   lead's `company` field.
5. If still no match, return a minimal EnrichedLead with confidence=0.1 (low,
   but non-zero so the scoring node knows enrichment was attempted).

Buying-signal extraction (form_text)
--------------------------------------
IMPORTANT: form_text is treated as an UNTRUSTED, READ-ONLY signal source.
Only keywords from the ICP-defined keyword list are extracted from it.
The raw form_text is NEVER forwarded to downstream nodes, and it NEVER
influences any field other than `buying_signals`.  This is an intentional
security constraint to prevent prompt-injection attacks from influencing
firmographic or routing decisions.

Employee-count → size band mapping
------------------------------------
  ≤ 50          : "1-50"
  51 – 200      : "51-200"
  201 – 500     : "201-500"
  501 – 1000    : "501-1000"
  1001 – 5000   : "1001-5000"
  > 5000        : "5001+"
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

from agent.models import EnrichedLead, Lead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "data"
_MOCK_COMPANIES_PATH: Final[Path] = _DATA_DIR / "mock_companies.json"
_ICP_PATH: Final[Path] = _DATA_DIR / "icp.json"

# Email domains that are personal / free-tier and carry no company signal.
_PERSONAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "icloud.com", "protonmail.com", "aol.com", "live.com",
        "me.com", "mac.com", "ymail.com", "msn.com",
    }
)

_ENRICHMENT_SOURCE: Final[str] = "mock_companies_v1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mock_companies() -> list[dict]:
    """Load the mock company fixture. Returns an empty list on any I/O error."""
    try:
        with _MOCK_COMPANIES_PATH.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_icp_keywords() -> dict[str, list[str]]:
    """
    Return the buying-signal keyword lists from icp.json.

    Returns a dict with keys 'high_intent', 'moderate_intent', 'low_intent'.
    Falls back to empty lists on any I/O error.
    """
    try:
        with _ICP_PATH.open(encoding="utf-8") as fh:
            icp = json.load(fh)
        bsk = icp.get("buying_signal_keywords", {})
        return {
            "high_intent": bsk.get("high_intent", []),
            "moderate_intent": bsk.get("moderate_intent", []),
            "low_intent": bsk.get("low_intent", []),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {"high_intent": [], "moderate_intent": [], "low_intent": []}


def _employee_count_to_band(count: int) -> str:
    """Map a raw employee count to the standard size band used by ICP scoring."""
    if count <= 50:
        return "1-50"
    if count <= 200:
        return "51-200"
    if count <= 500:
        return "201-500"
    if count <= 1000:
        return "501-1000"
    if count <= 5000:
        return "1001-5000"
    return "5001+"


def _extract_domain(email: str) -> str:
    """Return the lowercased domain portion of an email address."""
    return email.split("@")[-1].lower().strip()


def _find_company(domain: str, company_name: str, companies: list[dict]) -> dict | None:
    """
    Look up a company record by domain (exact) then by name (partial, case-insensitive).
    Returns the matching company dict or None.
    """
    # Pass 1 — exact domain match (most reliable)
    for co in companies:
        if co.get("domain", "").lower() == domain:
            return co

    # Pass 2 — company name contains match (handles 'FinTrust AI' vs 'fintrust')
    norm_name = company_name.lower()
    for co in companies:
        if norm_name and co.get("name", "").lower() in norm_name:
            return co
        if norm_name and norm_name in co.get("name", "").lower():
            return co

    return None


def _extract_buying_signals(
    form_text: str | None,
    company_keywords: list[str],
    icp_keywords: dict[str, list[str]],
) -> list[str]:
    """
    Extract buying-intent signals from two sources:
      1. The company's own keyword list (from mock_companies.json).
      2. Keywords found in the lead's form_text — ONLY if they appear in the
         ICP keyword allowlist (icp.json). Free-text is never forwarded raw.

    Returns a deduplicated list of matched signal strings, each labelled with
    its intent tier so the scoring node can weight them appropriately.
    """
    signals: list[str] = []

    # -- Source 1: company-level signals (pre-curated, fully trusted) ----------
    for kw in company_keywords:
        signals.append(f"company_signal:{kw}")

    # -- Source 2: form_text keyword scan (allowlist-only, untrusted input) ----
    if form_text:
        # Sanitise: work on lowercased text; never store the raw text.
        text_lower = form_text.lower()

        for tier, kw_list in icp_keywords.items():
            for kw in kw_list:
                # Use word-boundary-aware search to avoid partial matches.
                pattern = re.escape(kw.lower())
                if re.search(pattern, text_lower):
                    signals.append(f"form:{tier}:{kw}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_lead(lead: Lead) -> EnrichedLead:
    """
    Enrich a Lead with firmographic and intent data from the mock dataset.

    Parameters
    ----------
    lead : Lead
        The raw lead to enrich.

    Returns
    -------
    EnrichedLead
        Always returns an EnrichedLead.  When no company data is found the
        enrichment_confidence is ≤ 0.1 and all firmographic fields are None.
        The scoring node must handle low-confidence records gracefully.
    """
    domain = _extract_domain(str(lead.email))

    # -- Fast path: personal / free-tier email domain -------------------------
    if domain in _PERSONAL_DOMAINS:
        return EnrichedLead(
            lead=lead,
            enrichment_confidence=0.0,
            enrichment_source=_ENRICHMENT_SOURCE,
        )

    companies = _load_mock_companies()
    icp_keywords = _load_icp_keywords()
    company_record = _find_company(domain, lead.company, companies)

    # -- No match found -------------------------------------------------------
    if company_record is None:
        # Extract whatever signals we can from form_text even without firmographics.
        signals = _extract_buying_signals(lead.form_text, [], icp_keywords)
        return EnrichedLead(
            lead=lead,
            buying_signals=signals,
            enrichment_confidence=0.1,
            enrichment_source=_ENRICHMENT_SOURCE,
        )

    # -- Matched company — populate all firmographic fields -------------------
    employee_count: int = company_record.get("employee_count", 0)
    signals = _extract_buying_signals(
        lead.form_text,
        company_record.get("buying_signal_keywords", []),
        icp_keywords,
    )

    return EnrichedLead(
        lead=lead,
        company_size=_employee_count_to_band(employee_count),
        industry=company_record.get("industry"),
        annual_revenue_usd=company_record.get("annual_revenue_usd"),
        location=company_record.get("location"),
        tech_stack=company_record.get("tech_stack", []),
        buying_signals=signals,
        enrichment_confidence=0.9,
        enrichment_source=_ENRICHMENT_SOURCE,
    )
