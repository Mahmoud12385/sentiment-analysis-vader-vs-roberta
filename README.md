# 🧠 Sentiment Analysis: VADER vs RoBERTa

## 🚀 Live Demo
👉 Try the app here:  
https://sentiment-analysis-vader-vs-roberta-uccfvmgycjrbjhrsxa4noq.streamlit.app/

---

## 📖 Project Overview
This project is a Natural Language Processing (NLP) application that performs sentiment analysis on text data using two different approaches:

- 🔹 **VADER (Valence Aware Dictionary and sEntiment Reasoner)** – rule-based sentiment analysis
- 🔹 **RoBERTa (Transformer-based model)** – deep learning approach using HuggingFace Transformers

The goal is to compare classical NLP methods with modern transformer-based models on real-world data.

---

## 🧠 Models Used

### 🔹 VADER
- Rule-based sentiment analysis model
- Works well on short texts like reviews and social media
- Outputs:
  - Positive
  - Neutral
  - Negative
  - Compound score (-1 to +1)

### 🔹 RoBERTa
- Pretrained transformer model
- Context-aware and more accurate
- Model used:
  - `cardiffnlp/twitter-roberta-base-sentiment`

---

## 📊 Dataset
- Amazon Fine Food Reviews Dataset (from Kaggle)
- Contains:
  - Text reviews
  - Star ratings (1–5)
- Due to size limitations, only a sample of the dataset is used in the app.

---

## ⚙️ Features

- ✍️ Input custom text for analysis
- 📊 Compare VADER vs RoBERTa results
- 📈 Visual bar chart comparison
- 🧠 Final sentiment prediction
- ⚡ Fast real-time inference

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🎈
- PyTorch 🔥
- Transformers 🤗
- NLTK 📚
- Pandas & NumPy 📊
- Matplotlib & Seaborn 📉

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-vader-vs-roberta.git

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
