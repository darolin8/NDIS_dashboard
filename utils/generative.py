"""
Lightweight generative helpers for summaries and mitigations.
- Pure rule-based defaults (no external APIs).
- Optional LLM hook via `llm_call(prompt: str) -> str`.

Usage:
    from ndis_dashboard.utils.generative import generate_summary_and_mitigations
    summary, recs = generate_summary_and_mitigations(row_dict, narrative="...")

row_dict can be a pandas Series or a dict.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any

# ---------- small utilities ----------

def _to_dict(row: Any) -> Dict[str, Any]:
    try:
        return row.to_dict()  # pandas Series
    except Exception:
        return dict(row) if isinstance(row, dict) else {}

def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

# ---------- rule-based generators (no API needed) ----------

def summarize_incident_row(row_like: Any) -> str:
    """1–2 sentence, concise summary from available fields."""
    row = _to_dict(row_like)
    typ = str(row.get("incident_type", "incident")).strip().title() or "Incident"
    sev = str(row.get("severity", "Unknown")).strip()
    loc = str(row.get("location", "Unknown")).strip()
    carer = f"Carer {row.get('carer_id')}" if row.get("carer_id") else "Carer"
    date = str(row.get("incident_date", "")).split(" ")[0]  # yyyy-mm-dd if datetime-like
    delay = _safe_int(row.get("notification_delay_days"), None)

    delay_txt = f", notified after {delay} day(s)" if delay is not None else ""
    pieces = [
        f"On {date} at {loc}, a {typ} occurred (severity: {sev}){delay_txt}.",
        f"{carer} documented the incident; follow-up actions recorded where applicable."
    ]
    return " ".join(pieces).strip()

def recommend_mitigations(row_like: Any) -> List[str]:
    """Return a short, prioritized list of mitigations based on simple rules."""
    row = _to_dict(row_like)
    recs: List[str] = []

    sev_raw = str(row.get("severity", "")).lower()
    typ = str(row.get("incident_type", "")).lower()
    delay = row.get("notification_delay_days", None)
    delay = _safe_int(delay, None)

    # Timeliness
    if delay is not None and delay > 1:
        recs.append("Reinforce 24-hour notification SOP and add auto-reminders at 12 hours.")

    # Type-specific hooks
    if "self" in typ or "harm" in typ:
        recs.append("Initiate immediate risk review and update safety plan with clinician.")
    if "medication" in typ:
        recs.append("Run a double-check protocol refresher and update MAR audit checklist.")
    if "fall" in typ:
        recs.append("Conduct an environment assessment and install fall-prevention aids.")

    # Severity escalation
    if sev_raw in {"high", "critical", "4", "5"}:
        recs.append("Escalate to on-call senior and schedule a post-incident debrief within 48 hours.")

    if not recs:
        recs.append("Log incident, review care plan triggers, and schedule targeted refresher training.")

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for r in recs:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out

# ---------- orchestration with optional LLM ----------

def generate_summary_and_mitigations(
    row_like: Any,
    narrative: str | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> Tuple[str, List[str]]:
    """
    Returns (summary_text, mitigation_list).
    If llm_call is provided, it should accept a prompt and return a string.
    Falls back to rule-based outputs if llm_call is None.
    """
    if llm_call is None:
        return summarize_incident_row(row_like), recommend_mitigations(row_like)

    row = _to_dict(row_like)
    prompt = f"""You are an incident-management assistant for NDIS providers.

Narrative:
{narrative or row.get('narrative', '')}

Context (JSON-like):
{row}

Tasks:
1) Provide a concise 3–4 sentence executive summary (Australian English).
2) Provide a prioritized bullet list (3–5) of mitigation recommendations aligned to 24-hour notification and 5-day investigation SLAs.
Be specific and actionable. Avoid PHI. Keep total under 150 words."""
    text = llm_call(prompt).strip()

    # If you later want to parse bullets from the LLM text, do that here.
    # For now, return the whole text as "summary" and keep rule-based mitigations:
    return text, recommend_mitigations(row)
