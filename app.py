import streamlit as st
from data.processor import execution
from ui.sidebar import create_sidebar

# --- Page Config ---
st.set_page_config(
    page_title="Portfolio Optimization Tool",
    layout="wide"
)

st.title("Portfolio Optimization Dashboard")

# --- Hardcoded Top US Tickers (Top 100 proxy by market cap) ---
custom_tickers, selected_list, model_choice, apply_constraints, start_date, end_date = create_sidebar()


# --- Execution ---
execution(custom_tickers, selected_list, model_choice, apply_constraints, start_date, end_date)
    