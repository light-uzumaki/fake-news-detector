# 📰 Real-Time Fake News Detection System

## 🔗 Live Demo
👉 https://fake-news-detector1912.streamlit.app

---

## 🚀 Overview
This project is a **machine learning-based fake news detection system** that classifies news as **REAL or FAKE**.

It also integrates a **real-time news API**, allowing users to test live headlines using the trained model.

---

## ✨ Features
- 🧠 ML model trained using TF-IDF + Logistic Regression  
- 🌐 Real-time news fetching using News API  
- ⚡ Instant prediction (REAL / FAKE)  
- 📊 Confidence score for predictions  
- 🖥️ Interactive web app built with Streamlit  

---

## 🧠 How It Works

1. Text is cleaned (lowercasing, removing stopwords, etc.)
2. Converted into numerical form using **TF-IDF Vectorization**
3. Passed into a **Logistic Regression model**
4. Model predicts whether news is **REAL or FAKE**

---

## 🛠️ Tech Stack
- Python  
- Scikit-learn  
- NLP (TF-IDF)  
- Streamlit  
- NewsAPI  
- NLTK  

---

## 🌐 Real-Time Feature
The app fetches live news headlines and runs predictions on them using the trained ML model.

---

## 📂 Project Structure

fake-news-detector/
│
├── app.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
