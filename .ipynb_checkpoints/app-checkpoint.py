import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (runs once)
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function (SAME as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Paste a news article below and check if it's real or fake.")

# Input
input_text = st.text_area("Enter News Article:")

# Button
if st.button("Predict"):

    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Clean input
        cleaned = clean_text(input_text)

        # Convert to vector
        vec = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        confidence = max(probability)

        # Output
        if prediction == 1:
            st.success(f"✅ REAL News (Confidence: {confidence:.2f})")
        else:
            st.error(f"🚨 FAKE News (Confidence: {confidence:.2f})")

        # Debug info (optional but useful)
        with st.expander("See processed text"):
            st.write(cleaned)

import requests

API_KEY = "6680505537844216b62c937316112ede"

def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data["articles"]

if st.button("Get Live News"):
    articles = fetch_news()

    for article in articles[:5]:
        title = article["title"]

        cleaned = clean_text(title)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success(f"📰 {title} → REAL")
        else:
            st.error(f"📰 {title} → FAKE")    
            


            