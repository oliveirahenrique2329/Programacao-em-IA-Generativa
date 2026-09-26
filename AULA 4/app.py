import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download do léxico do VADER
nltk.download('vader_lexicon', quiet=True)

# Inicializa o analisador de sentimentos
sia = SentimentIntensityAnalyzer()

# Título do aplicativo
st.title("Sentiment Analysis App")

# Campo de texto para entrada em inglês
texto = st.text_area("Enter text in English:", "I absolutely love learning new technologies! This app is amazing.")

# Botão para disparar a análise
if st.button("Analyze Sentiment"):
    if texto.strip() != "":
        # Calcula as pontuações de sentimento
        scores = sia.polarity_scores(texto)
        compound = scores['compound']
        
        # Define a classificação do sentimento com base no score compound
        if compound >= 0.05:
            sentimento = "Positive 😊"
        elif compound <= -0.05:
            sentimento = "Negative 😡"
        else:
            sentimento = "Neutral 😐"
        
        # Exibe a classificação e a pontuação compound
        st.write(f"**Sentiment:** {sentimento}")
        st.write(f"**Compound Score:** {compound:.4f}")
    else:
        st.warning("Please enter some text to analyze.")