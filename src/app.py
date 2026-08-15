import streamlit as st
from core.logger import get_custom_logger
from rag.vector_store import vector_store
from webapp.main_screen import main_app

if __name__ == '__main__':
    
    logger = get_custom_logger(name=__name__)
    
    st.set_page_config(
        page_title="Financial AI Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    vectorstore_status = vector_store.check_status()
    logger.info(msg = f"Vector Store Status: {vectorstore_status}")
    main_app()