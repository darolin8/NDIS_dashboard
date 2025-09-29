# ndis_dashboard/utils/alerts.py
from __future__ import annotations
import os, time, hashlib
from typing import Dict, Any, List, Tuple, Optional

# ---- Policy: what triggers an alert? (customise as needed)
def should_trigger_alert(row: Dict[str, Any]) -> Tuple[bool, str]:
    sev = str(row.get("severity", "")).lower()
    typ = str(row.get("incident_type", "")).lower()
    delay = row.get("notification_delay_days", None)
    rep = int(row.get("reportable_bin", 0))

    # Example early-risk rules
    if sev in {"high", "critical", "4", "5"}:
        return True, "high_severity"
    if "self" in typ or "harm" in typ:
        return True, "self_harm_risk"
    if delay is not None and delay > 1:
        return True, "late_notification"
    if rep == 1 and sev in {"moderate", "3"}:
        return True, "reportable_moderate"
    return False, ""

# ---- Safe, minimal SMS/email text (no PHI)
def format_alert_message(row: Dict[str, Any], reason: str) -> str:
    iid = str(row.get("incident_id", "unknown"))
    sev = str(row.get("severity", "NA"))
    loc = str(row.get("location", "NA"))
    date = str(row.get("incident_date", ""))[:10]
    return (f"[NDIS Early Alert] Incident {iid} flagged ({reason}). "
            f"Severity: {sev} | Location: {loc} | Date: {date}. "
            f"Please review and escalate per SOP.")

# ---- Twilio SMS (optional)
def send_sms_via_twilio(message: str, to_number: str) -> Optional[str]:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    if not all([sid, token, from_number]):
        return "Twilio not configured (missing env vars)."
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        resp = client.messages.create(body=message, from_=from_number, to=to_number)
        return f"SMS sent: {resp.sid}"
    except Exception as e:
        return f"Twilio error: {e}"

# ---- Email fallback (very simple SMTP example)
def send_email_smtp(message: str, to_email: str) -> Optional[str]:
    host = os.getenv("SMTP_HOST"); user = os.getenv("SMTP_USER"); pwd = os.getenv("SMTP_PASS")
    port = int(os.getenv("SMTP_PORT", "587") or 587)
    sender = os.getenv("SMTP_SENDER", user or "alerts@example.com")
    if not all([host, user, pwd, to_email]):
        return "SMTP not configured (missing env vars)."
    import smtplib
    from email.mime.text import MIMEText
    msg = MIMEText(message)
    msg["Subject"] = "NDIS Early Alert"
    msg["From"] = sender
    msg["To"] = to_email
    try:
        with smtplib.SMTP(host, port) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(sender, [to_email], msg.as_string())
        return "Email sent"
    except Exception as e:
        return f"SMTP error: {e}"

# ---- Simple de-dupe + cooldown
class AlertGate:
    def __init__(self, cooldown_seconds: int = 3600):
        self.cooldown = cooldown_seconds
        self._last: Dict[str, float] = {}

    def _key(self, row: Dict[str, Any], reason: str, channel: str) -> str:
        iid = str(row.get("incident_id", "")) + reason + channel
        return hashlib.sha256(iid.encode()).hexdigest()

    def allow(self, row: Dict[str, Any], reason: str, channel: str) -> bool:
        k = self._key(row, reason, channel)
        t = time.time()
        last = self._last.get(k, 0)
        if t - last >= self.cooldown:
            self._last[k] = t
            return True
        return False

# ---- Orchestrator
def route_alerts(
    row: Dict[str, Any],
    to_sms: Optional[str] = None,
    to_email: Optional[str] = None,
    gate: Optional[AlertGate] = None,
) -> List[str]:
    ok, reason = should_trigger_alert(row)
    if not ok:
        return ["No alert: policy not matched."]
    msg = format_alert_message(row, reason)
    results: List[str] = []
    gate = gate or AlertGate()

    if to_sms:
        if gate.allow(row, reason, "sms"):
            results.append(send_sms_via_twilio(msg, to_sms) or "SMS attempted.")
        else:
            results.append("SMS skipped: cooldown.")
    if to_email:
        if gate.allow(row, reason, "email"):
            results.append(send_email_smtp(msg, to_email) or "Email attempted.")
        else:
            results.append("Email skipped: cooldown.")
    if not (to_sms or to_email):
        results.append("No channels configured.")
    return results
