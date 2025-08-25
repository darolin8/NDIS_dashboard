#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py (main)
import streamlit as st
from dashboard_pages import PAGE_TO_RENDERER
# also import your ML helpers:
from your_module import (
    train_severity_prediction_model,
    prepare_ml_features,
    perform_anomaly_detection,
    find_association_rules,
    time_series_forecast,
    MLXTEND_AVAILABLE,
    STATSMODELS_AVAILABLE,
)

# ... your df & filtered_df computed as before

page = st.sidebar.selectbox(
    "Dashboard Pages",
    ["Executive Summary", "Operational Performance", "Compliance & Investigation", "🤖 Machine Learning Analytics", "Risk Analysis"]
)

renderer = PAGE_TO_RENDERER[page]
if page == "🤖 Machine Learning Analytics":
    renderer(
        filtered_df,
        train_severity_prediction_model=train_severity_prediction_model,
        prepare_ml_features=prepare_ml_features,
        perform_anomaly_detection=perform_anomaly_detection,
        find_association_rules=find_association_rules,
        time_series_forecast=time_series_forecast,
        MLXTEND_AVAILABLE=MLXTEND_AVAILABLE,
        STATSMODELS_AVAILABLE=STATSMODELS_AVAILABLE
    )
else:
    renderer(df, filtered_df) if page == "Executive Summary" else renderer(filtered_df)

