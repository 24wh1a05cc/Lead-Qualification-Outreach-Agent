"""
agent/nodes/approval_gate.py
----------------------------
Human-in-the-loop approval gate for outbound email.

This node presents the drafted email to the rep (via interactive CLI prompt)
and collects one of three decisions:

  (a) approve   — email is approved as-is; generate an approval_token
  (b) edit      — rep edits subject and/or body; then approve; generate token
  (c) reject    — rep rejects; no token generated; email is never sent

The approval_token is a UUID-4 string generated HERE, at approval time.
send_email() will refuse to operate without it — this is the only place in the
system that can produce a valid token.

Public API
----------
  request_approval(drafted_email, lead) -> ApprovalResult   ← interactive
  approval_gate_node(state) -> dict                          ← LangGraph wrapper
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import Final

from agent.models import ApprovalResult, Classification, DraftedEmail, Lead, LeadTier

_DIVIDER: Final[str] = "─" * 64
_GREEN: Final[str] = "\033[92m"
_YELLOW: Final[str] = "\033[93m"
_RED: Final[str] = "\033[91m"
_CYAN: Final[str] = "\033[96m"
_BOLD: Final[str] = "\033[1m"
_RESET: Final[str] = "\033[0m"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def request_approval(
    drafted_email: DraftedEmail,
    lead: Lead,
    *,
    rep_id: str = "cli-rep",
) -> ApprovalResult:
    """
    Present the drafted email to a human rep and collect their decision.

    Parameters
    ----------
    drafted_email : DraftedEmail
        The draft produced by the drafting node.
    lead : Lead
        The recipient lead (shown for context).
    rep_id : str
        Identifier of the rep reviewing this draft.  Defaults to 'cli-rep' for
        the interactive CLI session; override in tests or future UI integrations.

    Returns
    -------
    ApprovalResult
        Contains approved, edited_email (if edits made), approval_token (if
        approved), rep_id, decision, and rejection_reason (if rejected).
    """
    _display_draft_for_review(drafted_email, lead)

    while True:
        print(f"\n  {_BOLD}What would you like to do?{_RESET}")
        print(f"    {_GREEN}[a]{_RESET}  Approve and send as-is")
        print(f"    {_YELLOW}[e]{_RESET}  Edit subject / body, then approve")
        print(f"    {_RED}[r]{_RESET}  Reject (do not send)")
        print()

        choice = _prompt("  Your choice [a/e/r]: ").strip().lower()

        if choice == "a":
            return _build_approved(drafted_email, rep_id, edited=False)

        elif choice == "e":
            edited = _collect_edits(drafted_email)
            _display_edited_draft(edited)
            confirm = _prompt("\n  Confirm send of edited draft? [y/n]: ").strip().lower()
            if confirm == "y":
                return _build_approved(edited, rep_id, edited=True)
            else:
                print(f"  {_YELLOW}Edit not confirmed — returning to the menu.{_RESET}")
                continue

        elif choice == "r":
            reason = _prompt("  Rejection reason (required): ").strip()
            if not reason:
                print(f"  {_RED}A rejection reason is required.{_RESET}")
                continue
            return ApprovalResult(
                approved=False,
                edited_email=None,
                approval_token=None,
                rep_id=rep_id,
                decision="rejected",
                rejection_reason=reason,
                decided_at=datetime.now(tz=timezone.utc),
            )

        else:
            print(f"  {_RED}Invalid choice — please enter a, e, or r.{_RESET}")


def approval_gate_node(state: dict) -> dict:
    """
    LangGraph node: run the approval gate for HOT leads with a draft.

    Reads ``state['drafted_email']`` and ``state['enriched_lead']``.
    Writes ``state['approval_result']``.

    If no drafted_email is present (non-HOT lead), state passes through unchanged.
    """
    drafted_email: DraftedEmail | None = state.get("drafted_email")
    if drafted_email is None:
        return state

    lead: Lead = state["enriched_lead"].lead
    approval_result = request_approval(drafted_email, lead)
    return {**state, "approval_result": approval_result}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_draft_for_review(draft: DraftedEmail, lead: Lead) -> None:
    print(f"\n  {_BOLD}{'═' * 60}{_RESET}")
    print(f"  {_BOLD}  ✉  APPROVAL GATE — Review before sending{_RESET}")
    print(f"  {_BOLD}{'═' * 60}{_RESET}")
    print(f"\n  {_CYAN}Recipient :{_RESET}  {lead.name} <{lead.email}>")
    print(f"  {_CYAN}Company   :{_RESET}  {lead.company}  ({lead.role})")
    print(f"\n  {_CYAN}Subject   :{_RESET}  {draft.subject}")
    print()
    print(f"  {_CYAN}Body:{_RESET}")
    for line in draft.body.splitlines():
        print(f"    {line}")
    print()
    print(f"  {_CYAN}Grounded facts:{_RESET}")
    for fact in draft.grounded_facts:
        print(f"    • {fact}")
    print(f"\n  {_BOLD}{'─' * 60}{_RESET}")


def _display_edited_draft(draft: DraftedEmail) -> None:
    print(f"\n  {_YELLOW}── EDITED DRAFT ──────────────────────────────────────{_RESET}")
    print(f"  {_CYAN}Subject:{_RESET} {draft.subject}")
    print()
    print(f"  {_CYAN}Body:{_RESET}")
    for line in draft.body.splitlines():
        print(f"    {line}")
    print(f"  {_YELLOW}──────────────────────────────────────────────────────{_RESET}")


# ---------------------------------------------------------------------------
# Edit flow
# ---------------------------------------------------------------------------

def _collect_edits(original: DraftedEmail) -> DraftedEmail:
    """
    Prompt the rep to optionally replace the subject and/or body.
    Any field left blank keeps the original value.
    """
    print(f"\n  {_YELLOW}EDIT MODE{_RESET} — press Enter to keep the original value.\n")

    print(f"  Current subject: {original.subject}")
    new_subject = _prompt("  New subject (or Enter to keep): ").strip()
    subject = new_subject if new_subject else original.subject

    print(f"\n  Current body:")
    for line in original.body.splitlines():
        print(f"    {line}")
    print()
    print("  Enter new body (type END on its own line to finish, or Enter to keep):")
    new_body = _read_multiline()
    body = new_body if new_body.strip() else original.body

    return DraftedEmail(
        lead_id=original.lead_id,
        subject=subject,
        body=body,
        # grounded_facts are preserved from original — rep edits are tracked
        # separately in ApprovalResult.edited_email vs the original.
        grounded_facts=original.grounded_facts,
        tone=original.tone,
    )


def _read_multiline() -> str:
    """Read multiple lines from stdin until the user types 'END' alone."""
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token generation and ApprovalResult builder
# ---------------------------------------------------------------------------

def _build_approved(
    email: DraftedEmail,
    rep_id: str,
    *,
    edited: bool,
) -> ApprovalResult:
    """Generate a fresh UUID-4 approval token and return an approved result."""
    token = str(uuid.uuid4())
    decision = "approved_with_edits" if edited else "approved"
    print(f"\n  {_GREEN}✓ Approved{' (with edits)' if edited else ''}.{_RESET}")
    print(f"  {_GREEN}  Approval token: {token}{_RESET}\n")
    return ApprovalResult(
        approved=True,
        edited_email=email if edited else None,
        approval_token=token,
        rep_id=rep_id,
        decision=decision,
        rejection_reason=None,
        decided_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# I/O helper (thin wrapper so tests can monkeypatch input())
# ---------------------------------------------------------------------------

def _prompt(message: str) -> str:
    """Write a prompt to stdout and read a line from stdin."""
    try:
        return input(message)
    except EOFError:
        # Non-interactive context (piped input exhausted) — treat as rejection.
        return "r"
