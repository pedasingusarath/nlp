import os
import requests
import streamlit as st
from local_inference import generate_answer

st.set_page_config(page_title="Academic Question Response Generator",
                   page_icon="📚", layout="centered")
st.title("📚 Academic Question Response Generator")
st.write("Enter an academic question and generate a short or medium explanation.")

question = st.text_area(
    "Enter your academic question",
    placeholder="Example: Explain the working of CNN",
    height=120
)
length = st.radio("Response length", ["short", "medium"], horizontal=True)

if st.button("Generate response", type="primary"):
    if not question.strip():
        st.error("Please enter a question.")
    elif len(question) > 2000:
        st.error("Question must be 2000 characters or fewer.")
    else:
        with st.spinner("Generating response..."):
            endpoint_url = os.getenv("SAGEMAKER_INVOKE_URL", "").strip()
            if endpoint_url:
                # For a deployed SageMaker endpoint, invoke through a small
                # authenticated backend/proxy. Do not put AWS credentials here.
                st.info("SageMaker URL is configured. Use the backend proxy described in README.")
                st.warning("Direct browser calls to SageMaker require secure AWS authentication.")
            else:
                try:
                    answer = generate_answer(question, length)
                    st.subheader("Generated response")
                    st.write(answer or "The model did not generate a response. Try another prompt.")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

st.caption("Notice: This is an AI-generated demonstration. Responses may be incomplete or incorrect. Verify them with your course materials.")
