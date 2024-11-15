import re
import pandas as pd
from minisom import MiniSom
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from difflib import SequenceMatcher
import json
from typing import Tuple
from groq import Groq
import requests
import time

DF = pd.read_csv("alimentos.csv")
data = DF.drop(columns=["Category"]).values
names = DF["Category"].values.flatten()
columns = DF.columns.to_list()[1:]
with open("embeddings_MiniLMv2.json", "r") as f:
    EMBEDDINGS = json.load(f).values()

cosine_similarity = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
sequence_similarity = lambda a, b : SequenceMatcher(None, a, b).ratio()
euclidean_distance = lambda p1, p2: np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))
message_parser = lambda role, text : {"role": role, "content": text}

class MiniLMv2():
    def __init__(self, api_key: str=""):
        self.api_key = api_key

    def embed(self, text: str, tries: int=5) -> np.ndarray:
        header = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        body = {'inputs': [text]}
        for _ in range(tries):
            try:
                url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
                return np.array(requests.post(url, json=body, headers=header).json()[0])
            except:
                time.sleep(1)
        raise RuntimeError(f"API failed to respond after {tries} attempts.")

class LLAMA():
    def __init__(self, api_key: str="", model: str="llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
    
    def ask(self, instruction: str, max_tokens: int=100, tries: int=3) -> str:
        client = Groq(api_key=self.api_key)
        for _ in range(tries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[message_parser("user",instruction)],
                    model=self.model,
                    temperature=0,
                    max_tokens=max_tokens
                )
                return chat_completion.choices[0].message.content
            except:
                time.sleep(1)
        raise RuntimeError(f"API failed to respond after {tries} attempts.")

    def ask_messages(self, messages: list, max_tokens: int=100, tries: int=3) -> str:
        client = Groq(api_key=self.api_key)
        for _ in range(tries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    max_tokens=max_tokens
                )
                return chat_completion.choices[0].message.content
            except:
                time.sleep(1)
        raise RuntimeError(f"API failed to respond after {tries} attempts.")

LLM = LLAMA()
EMBEDDER = MiniLMv2()

def find_healthy_closest(meal_list: "list[str]", user_som_data: list, top_k=3) -> "list[str]":
    healthy_list = []
    for food in meal_list:
        food_data = list(filter(lambda d : d["name"] == food, user_som_data))[0]

        if max([d["score"] for d in user_som_data]) == food_data["score"]:
            healthy_list.append(food_data["name"])
        else:
            candidate_food = [d for d in user_som_data if d["score"] > food_data["score"]]

            distances = np.array([euclidean_distance([food_data["x"], food_data["y"]], [d["x"], d["y"]]) for d in candidate_food])
            delta_scores = np.array([d["score"] for d in candidate_food]) - food_data["score"]

            closeness_score = 1.0 - (distances / np.max(distances))
            compatibility_score = 1.0 - (delta_scores / np.max(delta_scores))
            absolute_scores = (closeness_score + compatibility_score) / 2

            most_compatible_indexes = np.argsort(-absolute_scores).tolist()[:top_k]
            healthy_list += [candidate_food[i]["name"] for i in most_compatible_indexes]

    return healthy_list

def meal_description_to_list(meal_description: str) -> "list[str]":
    foods = [f.strip().lower() for f in meal_description.split(",")]

    found_foods = []
    for food in foods:
        food_embedding = EMBEDDER.embed(food)
        similarities = np.array([cosine_similarity(food_embedding, e) + sequence_similarity(food, n.lower()) for n,e in zip(names, EMBEDDINGS)]) / 2.0
        idx = int(np.argmax(similarities))
        if similarities[idx] >= 0.7:
            found_foods.append(names[idx])

    return found_foods

def column_categorization_prompt(user_description: str, top_k=3) -> dict:
    few_shot_prompts = [
        message_parser("system", f'Atributos de alimentos: {columns}\nDada a descrição do usuário e os atributos dos alimentos, indique o impacto de cada atributo alimentar para o caso do usuário, especificando se é [MUITO BENÉFICO, BENÉFICO, NEUTRO, PREJUDICIAL, MUITO PREJUDICIAL]. A resposta deve incluir TODOS os atributos de alimentos listados anteriormente e seguir o formato abaixo:\n"Atributo: Impacto"\nUse uma nova linha para cada atributo. Responda levando em consideração o estilo de vida, questões de saúde e metas do usuário, escolhendo o que mais é adequado para ele (por exemplo, sal é MUITO PREJUDICIAL para quem tem hipertensão e carboidrato é MUITO PREJUDICIAL para quem tem diabetes).'),
        message_parser("user", "Eu estou começando a fazer exercícios e estou abaixo do peso. Eu gostaria de aprimorar meus ganhos em massa muscular"),
        message_parser("assistant", "Carboidratos: BENÉFICO\nColesterol: PREJUDICIAL\nFibra: BENÉFICO\nCalorias: MUITO BENÉFICO\nManganês: BENÉFICO\nProteína: MUITO BENÉFICO\nSelênio: BENÉFICO\nAçúcar: PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: BENÉFICO\nFósforo: BENÉFICO\nPotássio: BENÉFICO\nSódio: PREJUDICIAL\nZinco: NEUTRO\nVitamina C: BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: BENÉFICO\nVitamina A: BENÉFICO\nVitamina B: BENÉFICO"),
        message_parser("user", "Eu tenho pressão alta e gostaria de reduzir a quantidade de sódio e gorduras na minha dieta para melhorar minha saúde cardiovascular."),
        message_parser("assistant", "Carboidratos: NEUTRO\nColesterol: MUITO PREJUDICIAL\nFibra: MUITO BENÉFICO\nCalorias: PREJUDICIAL\nManganês: BENÉFICO\nProteína: BENÉFICO\nSelênio: BENÉFICO\nAçúcar: PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: BENÉFICO\nCobre: NEUTRO\nFerro: BENÉFICO\nMagnésio: BENÉFICO\nFósforo: BENÉFICO\nPotássio: MUITO BENÉFICO\nSódio: MUITO PREJUDICIAL\nZinco: NEUTRO\nVitamina C: MUITO BENÉFICO\nVitamina E: MUITO BENÉFICO\nVitamina K: NEUTRO\nVitamina A: NEUTRO\nVitamina B: BENÉFICO"),
        message_parser("user", "Recentemente, fui diagnosticado com osteoporose e preciso de alimentos que fortaleçam os ossos e melhorem minha densidade óssea."),
        message_parser("assistant", "Carboidratos: NEUTRO\nColesterol: PREJUDICIAL\nFibra: BENÉFICO\nCalorias: NEUTRO\nManganês: BENÉFICO\nProteína: BENÉFICO\nSelênio: NEUTRO\nAçúcar: PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: MUITO BENÉFICO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: MUITO BENÉFICO\nFósforo: MUITO BENÉFICO\nPotássio: BENÉFICO\nSódio: PREJUDICIAL\nZinco: NEUTRO\nVitamina C: NEUTRO\nVitamina E: NEUTRO\nVitamina K: MUITO BENÉFICO\nVitamina A: NEUTRO\nVitamina B: NEUTRO"),
        message_parser("user", "Estou tentando reduzir a ingestão de açúcares para controlar meu peso e prevenir picos de glicose."),
        message_parser("assistant", "Carboidratos: PREJUDICIAL\nColesterol: NEUTRO\nFibra: MUITO BENÉFICO\nCalorias: PREJUDICIAL\nManganês: NEUTRO\nProteína: BENÉFICO\nSelênio: NEUTRO\nAçúcar: MUITO PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: BENÉFICO\nFósforo: NEUTRO\nPotássio: BENÉFICO\nSódio: PREJUDICIAL\nZinco: NEUTRO\nVitamina C: BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: NEUTRO\nVitamina A: NEUTRO\nVitamina B: BENÉFICO"),
        message_parser("user", "Estou buscando melhorar minha saúde mental e reduzir o estresse. Gostaria de incluir nutrientes que promovam o bem-estar e a saúde cerebral."),
        message_parser("assistant", "Carboidratos: NEUTRO\nColesterol: PREJUDICIAL\nFibra: NEUTRO\nCalorias: NEUTRO\nManganês: BENÉFICO\nProteína: BENÉFICO\nSelênio: NEUTRO\nAçúcar: PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: BENÉFICO\nMagnésio: MUITO BENÉFICO\nFósforo: BENÉFICO\nPotássio: BENÉFICO\nSódio: NEUTRO\nZinco: BENÉFICO\nVitamina C: NEUTRO\nVitamina E: BENÉFICO\nVitamina K: NEUTRO\nVitamina A: NEUTRO\nVitamina B: MUITO BENÉFICO"),
        message_parser("user", "Tenho diabetes tipo 2 e quero controlar meus níveis de açúcar no sangue."),
        message_parser("assistant", "Carboidratos: PREJUDICIAL\nColesterol: NEUTRO\nFibra: MUITO BENÉFICO\nCalorias: NEUTRO\nManganês: NEUTRO\nProteína: BENÉFICO\nSelênio: NEUTRO\nAçúcar: MUITO PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: BENÉFICO\nFósforo: NEUTRO\nPotássio: BENÉFICO\nSódio: NEUTRO\nZinco: BENÉFICO\nVitamina C: BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: NEUTRO\nVitamina A: NEUTRO\nVitamina B: BENÉFICO"),
        message_parser("user", "Fui diagnosticado com hipertensão e quero reduzir meu consumo de sódio e melhorar minha circulação."),
        message_parser("assistant", "Carboidratos: NEUTRO\nColesterol: PREJUDICIAL\nFibra: MUITO BENÉFICO\nCalorias: NEUTRO\nManganês: BENÉFICO\nProteína: BENÉFICO\nSelênio: BENÉFICO\nAçúcar: PREJUDICIAL\nLipídios: PREJUDICIAL\nCálcio: BENÉFICO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: MUITO BENÉFICO\nFósforo: BENÉFICO\nPotássio: MUITO BENÉFICO\nSódio: MUITO PREJUDICIAL\nZinco: BENÉFICO\nVitamina C: MUITO BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: BENÉFICO\nVitamina A: NEUTRO\nVitamina B: BENÉFICO"),
        message_parser("user", "Estou grávida e quero garantir que minha dieta contenha nutrientes essenciais para o desenvolvimento do meu bebê."),
        message_parser("assistant", "Carboidratos: BENÉFICO\nColesterol: NEUTRO\nFibra: BENÉFICO\nCalorias: BENÉFICO\nManganês: BENÉFICO\nProteína: MUITO BENÉFICO\nSelênio: BENÉFICO\nAçúcar: PREJUDICIAL\nLipídios: NEUTRO\nCálcio: MUITO BENÉFICO\nCobre: BENÉFICO\nFerro: MUITO BENÉFICO\nMagnésio: MUITO BENÉFICO\nFósforo: MUITO BENÉFICO\nPotássio: BENÉFICO\nSódio: NEUTRO\nZinco: MUITO BENÉFICO\nVitamina C: MUITO BENÉFICO\nVitamina E: MUITO BENÉFICO\nVitamina K: BENÉFICO\nVitamina A: BENÉFICO\nVitamina B: MUITO BENÉFICO"),
        message_parser("user", "Estou tentando ganhar peso de forma saudável, mantendo um equilíbrio entre calorias e nutrientes essenciais."),
        message_parser("assistant", "Carboidratos: BENÉFICO\nColesterol: NEUTRO\nFibra: BENÉFICO\nCalorias: MUITO BENÉFICO\nManganês: BENÉFICO\nProteína: MUITO BENÉFICO\nSelênio: NEUTRO\nAçúcar: NEUTRO\nLipídios: NEUTRO\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: BENÉFICO\nFósforo: BENÉFICO\nPotássio: BENÉFICO\nSódio: NEUTRO\nZinco: BENÉFICO\nVitamina C: BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: NEUTRO\nVitamina A: BENÉFICO\nVitamina B: BENÉFICO"),
        message_parser("user", "Tenho colesterol alto e gostaria de reduzir meu consumo de gorduras saturadas."),
        message_parser("assistant", "Carboidratos: NEUTRO\nColesterol: MUITO PREJUDICIAL\nFibra: MUITO BENÉFICO\nCalorias: PREJUDICIAL\nManganês: BENÉFICO\nProteína: BENÉFICO\nSelênio: BENÉFICO\nAçúcar: NEUTRO\nLipídios: MUITO PREJUDICIAL\nCálcio: NEUTRO\nCobre: NEUTRO\nFerro: NEUTRO\nMagnésio: BENÉFICO\nFósforo: BENÉFICO\nPotássio: BENÉFICO\nSódio: PREJUDICIAL\nZinco: BENÉFICO\nVitamina C: BENÉFICO\nVitamina E: BENÉFICO\nVitamina K: BENÉFICO\nVitamina A: NEUTRO\nVitamina B: BENÉFICO"),
        message_parser("user", re.sub(r'\s+', '', user_description).strip())
    ]

    full_impact_dict = {c : [] for c in columns}

    for _ in range(top_k):
        impact_dict = {}
        resp = LLM.ask_messages(few_shot_prompts, 200).lower()

        for row in filter(lambda a : ":" in a, resp.split("\n")):
            if "muito benéfico" in row: impact = 1.0
            elif "benéfico" in row: impact = 0.5
            elif "prejudicial" in row: impact = -0.5
            elif "muito prejudicial" in row: impact = -1.0
            else: impact = 0.01

            for c in filter(lambda col : col.lower() in row, columns):
                impact_dict[c] = impact

        for c in impact_dict.keys():
            full_impact_dict[c].append(impact_dict[c])

    most_common_impacts = {}
    for k in full_impact_dict.keys():
        values = full_impact_dict[k]
        if not values:
            most_common_impacts[k] = 0.01
        else:
            set_values = sorted(list(set(values)))
            set_counts = [values.count(v) for v in set_values]
            idx = np.argmax(set_counts)
            most_common_impacts[k] = set_values[idx]

    return most_common_impacts

def suggest_meals(user_prompt: str, meal_list: "list[str]", healthy_closest: "list[str]") -> str:
    meal_text = ", ".join(meal_list)
    healthy_text = ", ".join(healthy_closest)

    prompts = [
        message_parser("system", f'Você é uma IA assistente de alimentação. Seu objetivo é tornar os usuários mais saudáveis de acordo com seus objetivos e restrições alimentares. O usuário que você vai ajudar tem a seguinte descrição: {user_prompt}'),
        message_parser("user", f"O usuário gostaria de comer uma refeição contendo {meal_text}. No entanto, há uma série de alimentos que poderiam ser substituídos nesta refeição para melhorar sua alimentação, o que inclui {healthy_text}. Faça uma breve sugestão de refeições que o usuário poderia fazer com alguns destes alimentos. Não escolha opções muito diferentes das que o usuário escolheu ou ele pode não querer consumir. Você pode recomendar mais de uma alternativa. Ele pode ler esse texto, converse como se estivesse falando diretamente com ele, mas seja direto ao ponto. Você deve responder com até 50 palavras, em português.")
    ]

    resp = LLM.ask_messages(prompts, 1000).lower()
    return resp.capitalize()

def make_user_som(column_categorization: dict) -> Tuple[list, go.Figure]:
    grid_x, grid_y = 15, 15
    norm_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    custom_data = norm_data * np.array([column_categorization[c] for c in columns])
    val_grid = np.zeros((grid_x, grid_y))
    hover_grid = [["" for x in range(grid_x)] for y in range(grid_y)]

    # SOM
    som = MiniSom(grid_x, grid_y, custom_data.shape[1], sigma=3, learning_rate=1e-1, topology="rectangular")
    som.random_weights_init(custom_data)
    som.train_random(custom_data, 1000, verbose=False)
    units = som.labels_map(custom_data, names)
    som_data = []

    for x, y in units:
        food_names = list(units[(x, y)].keys())
        indexes = np.array([names.tolist().index(n) for n in food_names], dtype=int)
        food_scores = np.sum(custom_data[indexes], axis=1).tolist()
        avg_score = float(np.mean(food_scores))
        val_grid[(x, y)] = avg_score
        hover_grid[x][y] = "<br>".join(food_names[:15])

        for food_name, food_score in zip(food_names, food_scores):
            som_data.append({"name": food_name, "score": food_score, "x": x, "y": y})

    val_grid_df = pd.DataFrame(val_grid, columns=range(grid_x), index=range(grid_y))

    # Plot with Plotly Express
    fig = px.imshow(
        val_grid_df,
        color_continuous_scale="RdBu",
        aspect="auto",
        labels={"color": "Average Score"},
        title="SOM Heatmap Based on Average Score per Cell",
        zmin=-1.5,
        zmax=1.5
    )

    fig.update_traces(hovertemplate="Average Score: %{z:.2f}<br>%{customdata}")
    fig.update(data=[{"customdata": hover_grid}])
    fig.update_layout(xaxis_title="SOM Grid X", yaxis_title="SOM Grid Y", width=768, height=576)

    return som_data, fig

