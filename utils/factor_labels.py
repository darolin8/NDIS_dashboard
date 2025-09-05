
# utils/factor_labels.py
import re

def shorten_factor(text: str) -> str:
    """
    Collapse long factor labels to 1–2 words for compact plots.
    Example rules: keep first 1–2 words, strip punctuation, title-case.
    """
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    # remove non-word chars, split, keep up to 2 tokens
    tokens = re.sub(r"[^\w\s-]", "", text).split()
    return " ".join(tokens[:2]).title()


PHRASE_MAP = {
    # keys must be lowercase
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

SHORT_LABELS_REGEX = [
    (r"\bmental health|anxiety|depress|psych",          "Mental health"),
    (r"\bpalliative",                                   "Palliative"),
    (r"\b(financ|expense|invoice|transaction)\b",       "Finance"),
    (r"\b(1:?1|one[- ]to[- ]one|supervis|monitoring)\b","Supervision"),
    (r"\b(fatigue|tired|low morale|long day)\b",        "Fatigue"),
    (r"\bstorage\b",                                    "Storage"),
    (r"\b(exploit|vulnerab)",                           "Exploitation"),
    (r"\b(routin|schedule|rostering|unstructured)\b",   "Routine"),
    (r"\bhygien",                                       "Hygiene"),
    (r"\b(access|support)\b",                           "Support"),
    (r"\b(underlying|existing|history).*(health|medic|illness|condition)", "Condition"),
    (r"\bisolation\b|social iso",                       "Isolation"),
    (r"\btrain(ing)?\b",                                "Training"),
    (r"\b(protocol|procedure|guideline|policy|code of conduct)", "Protocol"),
    (r"\bcommunicat|handover\b",                        "Comms"),
    (r"\bmedicat",                                      "Medication"),
    (r"\blight(ing)?\b",                                "Lighting"),
    (r"\b(fault|equipment|device|hoist|battery)\b",     "Equipment"),
    (r"\bmainten|wear and tear\b",                      "Maintenance"),
    (r"\bvehicle|traffic\b",                            "Transport"),
    (r"\b(allergen|wet floor|trip|floor|hazard|gate|unlatched|cable|sharp|alarm)\b", "Safety"),
    (r"\bbehavio(u)?r|escalation|de-?escalation|warning\b", "Behaviour"),
    (r"\bboundar",                                      "Boundaries"),
    (r"\bbehaviou?r support plan|BSP\b",                "BSP"),
    (r"\breport\b|notifi",                              "Reporting"),
    (r"\binfection|hydration\b",                        "Clinical"),
    (r"\belectr",                                       "Electrical"),
    (r"\b(fire|drill|evac)\b",                          "Fire"),
    (r"\bvisitor\b",                                    "Visitor"),
    (r"\bstaffing|understaff",                          "Staffing"),
    (r"\bndis|funding\b",                               "NDIS"),
    (r"\bbudget|cost\b",                                "Budget"),
    (r"\bdesign|layout\b",                              "Design"),
    (r"\b(overcrowding|noise|sensory)\b",               "Noise"),
    (r"\b(diagnos|testing)\b",                          "Diagnosis"),
    (r"\b(deterioration|sudden medical event|symptom)\b","Deterioration"),
]

STOPWORDS = {
    "the","a","an","of","to","and","or","for","with","in","on","by","from","as","per",
    "not","no","being","due","at","its","their","his","her","is","are","was","were",
    "participant","participants","staff","plan","care","existing","previous","history",
    "recent","during","around","about","between","within","into","across","levels",
    "area","areas","group","common","certain","appropriate","specific","latest"
}

def shorten_factor(text: str) -> str:
    s = str(text).strip().lower()
    if not s:
        return "Unknown"

    # 1) exact phrase
    if s in PHRASE_MAP:
        return PHRASE_MAP[s]

    # 2) regex buckets
    for pat, label in SHORT_LABELS_REGEX:
        if re.search(pat, s):
            return label

    # 3) fallback: first two informative tokens
    tokens = re.findall(r"[a-z]+", s)
    tokens = [t for t in tokens if t not in STOPWORDS]
    if not tokens:
        return "Other"
    if len(tokens) == 1:
        return tokens[0].title()
    return f"{tokens[0].title()} {tokens[1].title()}"
