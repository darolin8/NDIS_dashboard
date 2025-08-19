import streamlit as st
st.write("Basic test works!")

# Test each import individually
try:
    import pandas as pd
    st.write("✅ Pandas loaded")
except Exception as e:
    st.write("❌ Pandas error:", e)

try:
    import plotly
    st.write("✅ Plotly loaded")
except Exception as e:
    st.write("❌ Plotly error:", e)