"""Customer churn prediction — production system demo.

Reproduces the train/serve + batch-scoring + risk-tiering flow of a
production churn system on top of the trained LightGBM artifact.
"""

import streamlit as st

st.set_page_config(
    page_title="Telco Churn — Production Demo",
    page_icon=":material/insights:",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("app_pages/batch_scoring_01.py", title="Churn risk & action queue", icon=":material/dns:"),
        st.Page("app_pages/single_customer_02.py", title="Customer lookup", icon=":material/person_search:"),
    ]
)
pages.run()
