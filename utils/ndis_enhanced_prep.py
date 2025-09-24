
import pandas as pd
import numpy as np

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_MAP = {
    "low": "Low",
    "minor": "Low",
    "medium": "Medium",
    "moderate": "Medium",
    "high": "High",
    "major": "High",
    "critical": "Critical",
    "severe": "Critical",
}
SEVERITY_NUMERIC_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

TRUE_SET = {"yes","y","true","1","reportable","required"}
FALSE_SET = {"no","n","false","0","non-reportable","not required","none"}

def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    try:
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            return (sn.fillna(0) > 0).astype(int)
    except Exception:
        pass
    sm = s.astype(str).str.strip().str.lower()
    return sm.apply(lambda x: 1 if x in TRUE_SET else (0 if x in FALSE_SET else np.nan)).astype("Int64").fillna(0).astype(int)

def normalise_severity(s: pd.Series) -> pd.Series:
    sm = s.astype(str).str.strip().str.lower().map(SEVERITY_MAP).fillna(s.astype(str))
    sm = sm.where(sm.isin(SEVERITY_ORDER), "Medium")
    return pd.Categorical(sm, categories=SEVERITY_ORDER, ordered=True)

def parse_datetime(date_col: pd.Series, time_col: pd.Series | None = None) -> pd.Series:
    d = pd.to_datetime(date_col, errors="coerce")
    if time_col is not None and time_col.notna().any():
        try:
            dt = pd.to_datetime(d.astype(str) + " " + time_col.astype(str), errors="coerce")
        except Exception:
            dt = d
        return dt.fillna(d)
    return d

def add_histories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["participant_id","incident_datetime"])
    df["participant_prior_incidents"] = df.groupby("participant_id").cumcount()
    df = df.sort_values(["carer_id","incident_datetime"])
    df["carer_prior_incidents"] = df.groupby("carer_id").cumcount()
    df["participant_total_incidents"] = df.groupby("participant_id")["incident_id"].transform("count")
    df["carer_total_incidents"] = df.groupby("carer_id")["incident_id"].transform("count")
    return df

def compute_location_risk(df: pd.DataFrame) -> pd.DataFrame:
    loc_risk = (
        df.groupby("location", dropna=False)["severity_numeric"]
          .mean()
          .fillna(1.0)
          .rename("location_risk_score")
    )
    return df.merge(loc_risk, on="location", how="left")

def prepare_ndis_data(df: pd.DataFrame) -> pd.DataFrame:
    if "incident_id" not in df.columns:
        df["incident_id"] = np.arange(1, len(df)+1)

    df["incident_datetime"] = parse_datetime(df.get("incident_date"), df.get("incident_time"))
    df["incident_date"] = pd.to_datetime(df.get("incident_date"), errors="coerce")
    df["severity"] = normalise_severity(df.get("severity"))
    df["severity_numeric"] = df["severity"].map(SEVERITY_NUMERIC_MAP).astype(int)

    for col in ["participant_id","carer_id"]:
        if col not in df.columns:
            df[col] = (df.get("reported_by").fillna("unknown").astype(str).str.replace(r"\s+","_", regex=True) 
                       + "_" + df["incident_id"].astype(str))

    if "location" not in df.columns:
        df["location"] = "Unknown"
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip().str.title()

    if "medical_attention_required" in df.columns:
        df["medical_attention_required_bin"] = to_bool_series(df["medical_attention_required"])
    else:
        df["medical_attention_required_bin"] = 0

    if "reportable" in df.columns:
        df["reportable_bin"] = to_bool_series(df["reportable"])
    else:
        df["reportable_bin"] = 0

    df["hour"] = df["incident_datetime"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["incident_datetime"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["day_type"] = np.where(df["is_weekend"]==1, "weekend", "weekday")

    df = add_histories(df)
    df = compute_location_risk(df)

    df["is_kitchen"] = df["location"].str.contains("kitchen", case=False, na=False).astype(int)
    df["is_bathroom"] = df["location"].str.contains("bathroom|toilet|washroom|restroom", case=False, na=False).astype(int)

    return df

def create_comprehensive_features(df: pd.DataFrame):
    base = pd.DataFrame({
        "hour": df["hour"],
        "is_weekend": df["is_weekend"],
        "is_kitchen": df["is_kitchen"],
        "is_bathroom": df["is_bathroom"],
        "participant_incident_count": df["participant_prior_incidents"],
        "carer_incident_count": df["carer_prior_incidents"],
        "location_risk_score": df["location_risk_score"].astype(float),
        "severity_numeric": df["severity_numeric"].astype(float),
    }, index=df.index)

    cats = {}
    top_locs = df["location"].value_counts().head(10).index
    for loc in top_locs:
        cats[f"loc__{loc}"] = (df["location"] == loc).astype(int)

    if "incident_type" in df.columns:
        top_types = df["incident_type"].astype(str).value_counts().head(10).index
        for t in top_types:
            cats[f"type__{t}"] = (df["incident_type"].astype(str) == t).astype(int)

    cats_df = pd.DataFrame(cats, index=df.index) if cats else pd.DataFrame(index=df.index)
    features_df = pd.concat([base, cats_df], axis=1).fillna(0)

    feature_names = list(features_df.columns)
    X = features_df.to_numpy(dtype=float)
    return X, feature_names, features_df
