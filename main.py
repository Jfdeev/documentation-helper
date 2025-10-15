from typing import Set
from backend.core import run_llm
import streamlit as st

st.header("Documentation Helper")

prompt = st.text_input("Prompt", placeholder="Enter your question here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

def created_sources_string(sources_urls: Set[str]) -> str:
    if not sources_urls:
        return "No sources found."
    sources_list = list(sources_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"

    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(prompt)
        sources = set([doc.metadata["source"] for doc in generated_response['source_documents']])

        format_response = (
                f"{generated_response['result']}\n\n {created_sources_string(sources)}"
            )
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(format_response)


if st.session_state["chat_answer_history"]:
    for generated_response, user_prompt in zip(
        st.session_state["chat_answer_history"], st.session_state["user_prompt_history"]
    ):
        st.chat_message("User").write(user_prompt)
        st.chat_message("Assistant").write(generated_response)