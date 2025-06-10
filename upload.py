import streamlit as st
import tempfile
import os
import io
import contextlib
from pdf2vstore_base64 import process_pdf

st.set_page_config(page_title="📤 Upload PDF to VStore", layout="centered")
st.title("📤 Upload PDF and Store in Vector DB")

st.markdown("### Upload One or More PDFs")
uploaded_files = st.file_uploader("Drag or browse to upload", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"📄 **Processing:** `{uploaded_file.name}`")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Processing `{uploaded_file.name}`..."):
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                process_pdf(tmp_path)
            logs = output_buffer.getvalue()
            st.code(logs, language="text")
            st.success(f"✅ Finished: `{uploaded_file.name}`")

        os.remove(tmp_path)

