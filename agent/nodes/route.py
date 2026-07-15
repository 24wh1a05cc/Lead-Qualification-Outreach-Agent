"""
agent/nodes/route.py
--------------------
LangGraph conditional-edge stub for the routing stage.

NOTE: The live pipeline (main.py and ui/app.py) implements routing via a plain
      Python if/elif branch on Classification.tier — this wrapper is a dormant
      LangGraph integration stub.  route_node() raises NotImplementedError to
      make that explicit; it does NOT affect pipeline behaviour.
"""

from __future__ import annotations

from agent.models import LeadTier


def route_node(state: dict) -> str:
    """
    LangGraph conditional-edge function: determine the next node based on tier.

    Parameters
    ----------
    state:
        LangGraph state dict; expected to contain a ``classification`` key.

    Returns
    -------
    str
        The name of the next node: ``'draft'``, ``'nurture'``, or ``'disqualify'``.
    """
    raise NotImplementedError("route_node is not implemented yet — coming in Step 3.")
