import streamlit as st
import pandas as pd

st.title("Análisis de Opiniones de Clientes")

uploaded_file = st.file_uploader("Sube un archivo CSV con 20 opiniones", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Vista previa de datos:", df.head())

### PARTE 2

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
import nltk
nltk.download('stopwords')

def clean_text(text):
    stop_words = set(stopwords.words("spanish"))
    words = text.lower().split()
    return [word for word in words if word.isalpha() and word not in stop_words]

if uploaded_file:
    all_words = []
    for opinion in df['opinion']:
        all_words.extend(clean_text(opinion))

    st.subheader("Nube de palabras")
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(all_words))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.subheader("Top 10 palabras más frecuentes")
    top_words = Counter(all_words).most_common(10)
    top_df = pd.DataFrame(top_words, columns=['Palabra', 'Frecuencia'])
    st.bar_chart(top_df.set_index("Palabra"))

## PARTE 3

from transformers import pipeline

@st.cache_resource
def cargar_modelo():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

if uploaded_file:
    clasificador = cargar_modelo()
    resultados = df['opinion'].apply(lambda x: clasificador(x)[0])

    df['sentimiento'] = resultados.apply(lambda r: "positivo" if int(r['label'][0]) >= 4 else ("negativo" if int(r['label'][0]) <= 2 else "neutro"))
    st.subheader("Opiniones clasificadas")
    st.write(df[['opinion', 'sentimiento']])

    st.subheader("Distribución de sentimientos")
    st.bar_chart(df['sentimiento'].value_counts(normalize=True))

### PARTE 4

from transformers import pipeline

@st.cache_resource
def cargar_modelo():
    # Forzamos el uso de PyTorch ("pt") como backend
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="pt")

if uploaded_file:
    st.subheader("Clasificación de sentimientos")

    # Cargar el modelo una sola vez
    clasificador = cargar_modelo()

    # Clasificar cada opinión
    resultados = df['opinion'].apply(lambda x: clasificador(x)[0])

    # Traducir el resultado del modelo a etiquetas: positivo, neutro, negativo
    def interpretar_sentimiento(etiqueta):
        estrella = int(etiqueta['label'][0])  # la etiqueta viene como '4 stars', '2 stars', etc.
        if estrella >= 4:
            return "positivo"
        elif estrella <= 2:
            return "negativo"
        else:
            return "neutro"

    df['sentimiento'] = resultados.apply(interpretar_sentimiento)

    # Mostrar resultados
    st.write("Opiniones clasificadas:")
    st.dataframe(df[['opinion', 'sentimiento']])

    # Gráfico de porcentaje por clase
    st.subheader("Distribución de opiniones por clase")
    st.bar_chart(df['sentimiento'].value_counts(normalize=True))
