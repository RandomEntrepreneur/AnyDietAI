import streamlit as st
from my_utilities import column_categorization_prompt, make_user_som

if not "groq_token" in st.session_state.keys() or not "hf_token" in st.session_state.keys():
    st.switch_page("login.py")

with st.form("informações"):
    st.title("Informações do usuário")
    nome = st.text_input("Qual é o seu nome?")
    st.markdown("Conte-nos um pouco sobre suas necessidades alimentares. O que você busca em sua alimentação? Sinta-se à vontade para compartilhar suas metas (como melhorar sua saúde ou ganhar mais energia), condições de saúde (como diabetes, hipertensão, sobrepeso) ou até mesmo desafios com a alimentação, como transtornos alimentares. Estamos aqui para ajudar de forma acolhedora e sem julgamentos!")
    description = st.text_area("Descrição")

    if st.form_submit_button("Confirmar"):
        if not nome or not description:
            st.error("Por favor, preencha todos os campos para que possamos te ajudar :)")
        else:
            with st.spinner("Extraindo informações alimentares..."):
                st.session_state["nome"] = nome
                st.session_state["description"] = description
                st.session_state["col_categorization"] = column_categorization_prompt(description, top_k=5)
                som_data, fig = make_user_som(st.session_state["col_categorization"])
                st.session_state["som_data"] = som_data
                st.session_state["fig"] = fig
                st.switch_page("pages/personalizado.py")
                