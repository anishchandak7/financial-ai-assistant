import streamlit as st
from typing import Any, Dict, List
from service.llm_service import LLMService
from service.workflow_service import WorkflowService

llm_service = LLMService()
workflow_service = WorkflowService()

def render_assistant_message(content: Any):
    """Renders the assistant response in a structured, user-friendly format."""
    # 1. Fallback for plain string messages (e.g. greetings or fallback errors)
    if isinstance(content, str):
        st.markdown(content)
        return

    # 2. If content is not a dictionary, render as markdown string
    if not isinstance(content, dict):
        st.markdown(str(content))
        return

    # Extract components from workflow state
    routing_decisions: List[str] = content.get("routing_decisions") or content.get("decision") or []
    agent_output: Dict[str, Any] = content.get("agent_output") or {}
    citations: List[Dict[str, Any]] = content.get("citations") or []

    # Display Routing Decision Badge/Caption
    if routing_decisions:
        decisions_str = " ➔ ".join([f"`{agent}`" for agent in routing_decisions])
        st.caption(f":material/alt_route: **Routed to:** {decisions_str}")

    # Display Agent Responses
    if not agent_output:
        st.markdown("No response was generated for this request.")
    else:
        for agent_name, output_data in agent_output.items():
            if len(agent_output) > 1:
                st.markdown(f"#### :material/smart_toy: {agent_name}")

            if isinstance(output_data, dict):
                response_text = output_data.get("response", "")
            else:
                response_text = str(output_data)

            if response_text:
                st.markdown(response_text)

    # Collect and deduplicate citations
    all_citations: List[Dict[str, Any]] = []
    if citations:
        all_citations.extend(citations)
    else:
        for output_data in agent_output.values():
            if isinstance(output_data, dict) and output_data.get("citations"):
                all_citations.extend(output_data.get("citations"))

    unique_citations = []
    seen = set()
    for cit in all_citations:
        key = (cit.get("source_index"), cit.get("quote"))
        if key not in seen and cit.get("quote"):
            seen.add(key)
            unique_citations.append(cit)

    # Display Citations Section if citations are present
    if unique_citations:
        with st.expander(f":material/menu_book: Sources & Citations ({len(unique_citations)})", expanded=False):
            for i, cit in enumerate(unique_citations):
                source_idx = cit.get("source_index", i)
                file_name = cit.get("file_name") or f"Source [{source_idx}]"
                quote = cit.get("quote", "")
                meta = cit.get("metadata") or {}

                st.markdown(f"**[{source_idx}] :material/description: {file_name}**")
                st.markdown(f"> *\"{quote}\"*")

                meta_details = []
                if meta.get("page") is not None:
                    meta_details.append(f"Page: {meta.get('page')}")
                if meta.get("file_type"):
                    meta_details.append(f"Type: {meta.get('file_type')}")
                if meta_details:
                    st.caption(" • ".join(meta_details))

                if i < len(unique_citations) - 1:
                    st.divider()

def render_chat_interface():
    # Top Header & Model Selection
    header_col, dropdown_col = st.columns([2, 1])
    with header_col:
        st.subheader("Chat Interface")
    
    with dropdown_col:
        selected_model = st.selectbox(
            "LLM Provider", 
            llm_service.available_models(details="display_name"),
            label_visibility="collapsed"
        )
        workflow_service.set_model(selected_model=selected_model)
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat Container (Fixed height makes it scrollable)
    chat_container = st.container(height=500)
    
    with chat_container:
        # Render existing chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_assistant_message(message["content"])
                else:
                    st.markdown(message["content"])

    # Chat Input Box
    if prompt := st.chat_input("Type your message here..."):
        
        # Add user prompt to session state and render in chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing request..."):
                    response = workflow_service.execute(prompt)
                render_assistant_message(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
