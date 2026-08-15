import streamlit as st

# Import the UI components
from webapp.file_upload import render_file_upload
from webapp.chat_interface import render_chat_interface
from webapp.market_analysis import render_market_analysis

def main_app():
    # Create the three-column layout
    left_col, center_col, right_col = st.columns([1, 2.5, 1], gap="medium")

    # Render components into their designated columns
    with left_col:
        render_file_upload()

    with center_col:
        render_chat_interface()

    with right_col:
        render_market_analysis()