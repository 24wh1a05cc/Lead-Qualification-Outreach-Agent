"""
agent/nodes/enrich.py
---------------------
LangGraph node wrapper stub for the enrichment stage.

NOTE: The live pipeline (main.py and ui/app.py) calls
      tools.enrichment_lookup.enrich_lead() directly — this wrapper is a
      dormant LangGraph integration stub.  It is intentionally not invoked
      by any live code path.  enrich_node() raises NotImplementedError to
      make that explicit; it does NOT affect pipeline behaviour.
"""

from __future__ import annotations

from agent.models import EnrichedLead, Lead


def enrich_node(state: dict) -> dict:  # noqa: ARG001
    """
    LangGraph node: Enrich a Lead with external firmographic/intent data.

    Parameters
    ----------
    state:
        LangGraph state dict; expected to contain a ``lead`` key of type Lead.

    Returns
    -------
    dict
        Updated state with an ``enriched_lead`` key added.
    """
    raise NotImplementedError("enrich_node is not implemented yet — coming in Step 2.")
