import streamlit as st
import asyncio 
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile
from service.rag_service import rag_service

def render_file_upload():
    st.subheader("Document Upload")
    st.caption("Upload financial reports or data.")
    
    if "uploaded_files_status" not in st.session_state:
        st.session_state.uploaded_files_status = {}

    # 1. Add the Dropdown for Collection Selection
    # Make sure these names exactly match the collection names in your settings.COLLECTION_NAMES
    selected_collection = st.selectbox(
        "Select Document Category",
        options=rag_service.collection_mapping.keys(), 
        index=0 # Sets "Finance" as the default
    )

    selected_files: List[UploadedFile] = st.file_uploader(
        "Choose file(s)", 
        type=["pdf", "txt", "csv"], 
        accept_multiple_files=True
    )

    if st.button("Upload", type="primary", use_container_width=True):
        if selected_files:
            for file in selected_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # 2. Pass the selected collection to the ingest method
                        asyncio.run(rag_service.ingest_file(file, collection_name=selected_collection))
                        
                        st.session_state.uploaded_files_status[file.file_id] = {
                            "file_name": file.name,
                            "status": f"Uploaded Successfully to {selected_collection}"
                        }
                    except Exception as e:
                        st.session_state.uploaded_files_status[file.file_id] = {
                            "file_name": file.name,
                            "status": f"Failed: {str(e)}"
                        }
            st.rerun()
        else:
            st.warning("Please select at least one file to upload.")

    st.divider()

    st.markdown("### Uploaded Files")
    if st.session_state.uploaded_files_status:
        for _, data in st.session_state.uploaded_files_status.items():
            # Fixed minor f-string quote collision here by using single quotes inside
            if "Successfully" in data.get("status"):
                st.success(f"**{data.get('file_name')}**\n\nStatus: {data.get('status')}")
            else:
                st.error(f"**{data.get('file_name')}**\n\nStatus: {data.get('status')}")
    else:
        st.info("No files uploaded yet.")