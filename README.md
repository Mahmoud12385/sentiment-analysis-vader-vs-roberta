# 📊 Sentiment Analysis: VADER vs RoBERTa

## 📖 Overview
This project performs sentiment analysis on customer reviews using two approaches:
- VADER (rule-based)
- RoBERTa (transformer-based deep learning model)

The goal is to compare their performance and understand differences between classical NLP and modern deep learning approaches.

---

## 🧠 Models Used

### 🔹 VADER
- Lexicon and rule-based sentiment analysis
- Works well for short texts (e.g., social media)

### 🔹 RoBERTa
- Pretrained transformer model
- Context-aware sentiment understanding
- Model: `cardiffnlp/twitter-roberta-base-sentiment`

---

## 📊 Dataset
- Reviews dataset (first 500 samples used)
- Contains:
  - Text
  - Score (1–5 stars)

---

## ⚙️ Workflow

1. Data Loading & Cleaning  
2. Exploratory Data Analysis (EDA)  
3. VADER Sentiment Scoring  
4. RoBERTa Sentiment Prediction  
5. Merge Results  
6. Model Comparison (Visualization)

---

## 📈 Key Visualizations
- Sentiment distribution by star rating
- Average compound score vs rating
- Pairplot comparison between models

---

## 🚀 Installation

```bash
pip install -r requirements.txt
