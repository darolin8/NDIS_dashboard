# utils/factor_labels.py
import re
from typing import Optional

# Exact phrase → compact label. Keys must be lowercase.
PHRASE_MAP = {
    "natural causes": "Natural causes",
    "no signage on wet floor": "Wet floor",
    "wet floor not clearly marked": "Wet floor",
    "trip hazard not removed": "Trip hazard",
    "poor lighting in certain areas": "Lighting",
    "gate alarm not functioning": "Gate alarm",
    "gate left unlatched": "Gate unlatched",
    "chemical storage room left open": "Chem storage",
    "medication tray left unattended": "Med tray",
    "monitoring equipment malfunction": "Monitoring",
    "faulty equipment": "Equipment",
    "incorrect use of hoist": "Hoist misuse",
    "battery not charged before use": "Battery flat",
    "vehicle not parked securely": "Vehicle",
    "poor vehicle maintenance": "Vehicle",
    "noise levels in common area": "Noise",
    "sensory overload in group area": "Sensory",
    "environmental temperature": "Temperature",
    "environmental stressors (noise, crowds)": "Stressors",
    "environmental factors (weather events)": "Weather",
    "environmental factors (e.g., moisture impacting electronics)": "Moisture",
    "wear and tear reported but not addressed": "Wear/tear",
    "routine maintenance not completed": "Maintenance",
    "lack of regular maintenance": "Maintenance",
    "unsecured power cables": "Cables",
    "sharp object within reach": "Sharp object",
    "care plan not followed for hygiene support": "Care plan",
    "lack of 1:1 supervision as per care plan": "Supervision",
    "inadequate monitoring as per care plan": "Monitoring",
    "no approved behaviour support plan in place": "No BSP",
}

# Regex buckets (all lowercase; we lowercase input before matching)
_PATTERNS = [
    (r"\bmental (health|illness)|\banxiety|\bdepress|\bpsych",          "Mental health"),
    (r"\bpalliative\b",                                   "Palliative"),
    (r"\b(financ|expense|invoice|transaction)\b",         "Finance"),
    (r"\b(1:?1|one[- ]to[- ]one|supervis|monitoring)\b",  "Supervision"),
    (r"\b(fatigue|tired|low morale|long day)\b",          "Fatigue"),
    (r"\bstorage\b",                                      "Storage"),
    (r"\b(exploit|vulnerab)",                             "Exploitation"),
    (r"\b(routin|schedule|rostering|unstructured)\b",     "Routine"),
    (r"\bhygien",                                         "Hygiene"),
    (r"\b(access|support)\b",                             "Support"),
    (r"\b(underlying|existing|history).*(health|medic|illness|condition)", "Condition"),
    (r"\bisolation\b|social iso",                         "Isolation"),
    (r"\btrain(ing)?\b",                                  "Training"),
    (r"\b(protocol|procedure|guideline|policy|code of conduct)\b", "Protocol"),
    (r"\bcommunicat|handover\b",                          "Comms"),
    (r"\bmedicat",                                        "Medication"),
    (r"\blight(ing)?\b",                                  "Lighting"),
    (r"\b(fault|equipment|device|hoist|battery)\b",       "Equipment"),
    (r"\bmainten|wear and tear\b",                        "Maintenance"),
    (r"\b(vehicle|traffic)\b",                            "Transport"),
    (r"\b(allergen|wet floor|trip|floor|hazard|gate|unlatched|cable|sharp|alarm)\b", "Safety"),
    (r"\bbehavio(u)?r|escalation|de-?escalation|warning\b","Behaviour"),
    (r"\bboundar",                                        "Boundaries"),
    (r"\bbehaviou?r support plan|\bbsp\b",                "BSP"),  # <- fixed to lowercase
    (r"\breport\b|notifi",                                "Reporting"),
    (r"\binfection|hydration\b",                          "Clinical"),
    (r"\belectr",                                         "Electrical"),
    (r"\b(fire|drill|evac)\b",                            "Fire"),
    (r"\bvisitor\b",                                      "Visitor"),
    (r"\bstaffing|understaff",                            "Staffing"),
    (r"\bndis|funding\b",                                 "NDIS"),
    (r"\bbudget|cost\b",                                  "Budget"),
    (r"\bdesign|layout\b",                                "Design"),
    (r"\b(overcrowding|noise|sensory)\b",                 "Noise"),
    (r"\b(diagnos|testing)\b",                            "Diagnosis"),
    (r"\b(deterioration|sudden medical event|symptom)\b", "Deterioration"),
]
SHORT_LABELS_REGEX = [(re.compile(p), label) for p, label in _PATTERNS]

STOPWORDS = {
    "the","a","an","of","to","and","or","for","with","in","on","by","from","as","per",
    "not","no","being","due","at","its","their","his","her","is","are","was","were",
    "participant","participants","staff","plan","care","existing","previous","history",
    "recent","during","around","about","between","within","into","across","levels",
    "area","areas","group","common","certain","appropriate","specific","latest"
}

def shorten_factor(text: Optional[str]) -> str:
    """
    Compact a long factor string into a short label:
      1) Exact match via PHRASE_MAP
      2) Regex bucket from SHORT_LABELS_REGEX
      3) Fallback: first 1–2 informative tokens (title-cased)
    """
    if not isinstance(text, str) or not text.strip():
        return "Unknown"

    s = text.strip()
    s_low = s.lower()

    # 1) exact phrase
    if s_low in PHRASE_MAP:
        return PHRASE_MAP[s_low]

    # 2) regex buckets
    for pat, label in SHORT_LABELS_REGEX:
        if pat.search(s_low):
            return label

    # 3) fallback heuristic
    tokens = re.findall(r"[a-z0-9\-]+", s_low)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return "Other"
    return " ".join(t.title() for t in tokens[:2])
