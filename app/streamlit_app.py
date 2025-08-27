import streamlit as st
import requests
import os
import logging

API_URL = os.getenv("BACKEND_URL", "http://backend_api:8000/query")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

st.title("Personal Knowledge Assistant")

# Chat input
question = st.text_input("Ask a question about your documents:")

if st.button("Submit") and question.strip():
    with st.spinner("Searching & generating answer..."):
        resp = requests.post(API_URL, json={"question": question})
        if resp.status_code == 200:
            data = resp.json()
            # Log to app logs
            logger.info(f"Received data: {data}")
            logger.info(f"Answer: {data.get('answer')}")
            
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("### Sources")
            for src in data["sources"]:
                st.write(f"- **{src['metadata'].get('source', '?')}** – score: {src['score']:.4f}")
        else:
            st.error(f"Error: {resp.status_code}")
