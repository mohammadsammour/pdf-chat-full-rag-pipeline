import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research Paper Chat",
    page_icon="\U0001f4da",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    st.title("Your workspace")

    if st.button("New chat", width="stretch"):
        try:
            response = requests.delete(
                f"{API_URL}/chat/{st.session_state.chat_id}",
                timeout=15,
            )
            response.raise_for_status()
        except requests.RequestException:
            st.error("Could not start a new chat. Please check that the API is running.")
        else:
            st.session_state.chat_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.subheader("Add a document")
    st.caption("Upload a PDF to add it to your document library.")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if st.button(
        "Upload PDF",
        type="primary",
        width="stretch",
        disabled=uploaded_file is None,
    ):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }

        with st.spinner("Adding your document..."):
            try:
                response = requests.post(
                    f"{API_URL}/documents",
                    files=files,
                    timeout=1800,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException:
                st.error("Upload failed. Please check that the API is running and try again.")
            else:
                if data["filename"] not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(data["filename"])
                st.success(f"Added {data['filename']} ({data['pages']} pages).")

    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Uploaded this session")
        for file_name in st.session_state.uploaded_files:
            st.write(file_name)

st.title("Research Paper Chat")
st.caption("Explore your papers and technical documents with source references.")

welcome = st.empty()

if not st.session_state.messages:
    with welcome.container(border=True):
        st.subheader("What would you like to explore?")
        st.write(
            "Upload a PDF from the sidebar, or ask about documents "
            "already in your library."
        )
        st.caption("Try a question, a comparison, or a request for a summary.")

# Show the conversation again whenever Streamlit refreshes the page.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about your documents...",
    key="question_input",
    submit_mode="disable",
)

if question and question.strip():
    question = question.strip()
    welcome.empty()

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "chat_id": st.session_state.chat_id,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except requests.RequestException:
                answer = "Could not get a response. Please check that the API is running and try again."
                st.error(answer)
            else:
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
