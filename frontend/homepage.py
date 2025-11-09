import pandas as pd
import numpy as np
import streamlit as st


# Page config

st.set_page_config(
    page_title="ResumeAI – Candidate Ranking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Header

st.title("ResumeAI – AI-Powered Resume Screening")
st.caption("Paste a job description, upload up to 10 .doc/.docx resumes, and click Rank Candidates.")


# Home Page Content
st.subheader("Upload job description")
user_text = st.text_area(
    "Job Description (max 5,000 characters)",
    value="",
    max_chars=5000,
    height=220,
    help="Paste the JD here.",
)

st.subheader("Upload multiple resume files")
uploaded_docs = st.file_uploader(
    "Drag & drop up to 10 files (.doc/.docx only)",
    type=["doc", "docx"],
    accept_multiple_files=True,
    help="Only .doc and .docx files are accepted; max 10 files.",
)

# Max 10-file cap and preview list
if uploaded_docs:
    if len(uploaded_docs) > 10:
        st.warning("You selected more than 10 files — only the first 10 will be used.")
        uploaded_docs = uploaded_docs[:10]
    with st.container(border=True):
        st.write(f"**{len(uploaded_docs)}** file(s) ready:")
        st.write("  \n".join(f"• {f.name}" for f in uploaded_docs))

# Rank button
if st.button("🔎 Rank Candidates", type="primary"):
    if not user_text.strip():
        st.error("Please paste a job description before ranking.")
    elif not uploaded_docs or len(uploaded_docs) == 0:
        st.error("Please upload at least one resume (.doc).")
    else:
        st.success("Success!")

st.divider()
