import streamlit as st
from ui.sidebar import create_sidebar

st.set_page_config(
    page_title="Portfolio Optimization Tool",
    layout="wide"
)
st.title("Portfolio Optimization Dashboard")
create_sidebar()