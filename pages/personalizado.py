import streamlit as st
import plotly.graph_objects as go
from my_utilities import meal_description_to_list, find_healthy_closest,suggest_meals

if not "fig" in st.session_state.keys():
    st.switch_page("pages/inicio.py")
nome = st.session_state["nome"]

# Mapa de Alimentos
st.title("Mapa de Alimentos")
st.markdown("Explore sua lista de alimentos personalizada! Os alimentos estão organizados por cores para facilitar sua escolha: os alimentos em azul são os melhores para você, enquanto os em vermelho são aqueles que é melhor evitar. Use este mapa para fazer escolhas alinhadas às suas necessidades!")
st.plotly_chart(st.session_state["fig"], use_container_width=True)

# Importância de Nutrientes
st.title("Importância de Nutrientes")
st.markdown("Veja como os nutrientes influenciam sua saúde! Este gráfico destaca as propriedades dos alimentos que têm impacto positivo ou negativo para você. Nutrientes em azul são benéficos e devem ser priorizados, enquanto os em vermelho merecem atenção para serem evitados. Use esta visão para guiar suas escolhas nutricionais.")
cc = st.session_state["col_categorization"]
categories = list(cc.keys())
values = [min(abs(value), 1.0) for value in cc.values()]
colors = ["blue" if value > 0 else "red" for value in cc.values()]
fig = go.Figure()
for category, value, color in zip(categories, values, colors):
    fig.add_trace(go.Bar(x=[category], y=[value], marker_color=color, name=category))
fig.update_layout(title="Visualização da Importância de Nutrientes", xaxis_title="Nutrientes", yaxis_title="Importância", yaxis=dict(range=[0, 1]), showlegend=False, plot_bgcolor="white", margin=dict(l=40, r=40, t=40, b=40))
st.plotly_chart(fig, use_container_width=True)

# Recomendações Personalizadas
st.title("Recomendações Personalizadas")
st.markdown(f"{nome}, conte para nós o que você gostaria de comer agora e ajudaremos a encontrar opções mais saudáveis! Escreva os alimentos separados por vírgulas e veja como é fácil ajustar sua alimentação sem abrir mão do prazer.")
with st.form("Sugestão"):
    meal_description = st.text_input("Refeição", placeholder="Feijão, Arroz, Purê, Carne, ...")
    if st.form_submit_button("Obter Sugestões"):
        if not meal_description.strip():
            st.error("Por favor, escreva o que deseja comer no momento.")
        else:
            with st.spinner(f"Gerando recomendações para {nome}..."):
                meal_list = meal_description_to_list(meal_description)
                healthy_closest = find_healthy_closest(meal_list, st.session_state["som_data"])
                suggestion = suggest_meals(st.session_state["description"], meal_list, healthy_closest)
                st.success(suggestion)