import streamlit as st
from my_utilities import LLM, EMBEDDER

st.title("AnyDiet.AI")
with st.form("Tokens"):
    groq_token = st.text_input("GroqAPI Token")
    hf_token = st.text_input("HuggingFace Token")
    if st.form_submit_button("Entrar"):
        LLM.api_key = groq_token
        EMBEDDER.api_key = hf_token
        try:
            LLM.ask("Hi", max_tokens=1)
            EMBEDDER.embed("Hi", tries=5)
            st.session_state["groq_token"] = groq_token
            st.session_state["hf_token"] = hf_token
            st.switch_page("pages/inicio.py")
        except Exception as e:
            print(e)
            st.error("Os tokens que você inseriu não são válidos, tente novamente!")