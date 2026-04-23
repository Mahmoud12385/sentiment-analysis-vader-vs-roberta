import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import pandas as pd

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Sentiment Analysis App")
st.write("Compare VADER vs RoBERTa in real-time")

# ==============================
# تحميل NLTK
# ==============================
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sid = load_vader()

# ==============================
# تحميل RoBERTa
# ==============================
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Functions
# ==============================
def roberta_scores(text):
    encoded_text = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        output = model(**encoded_text)

    scores = output.logits[0].cpu().numpy()
    scores = softmax(scores)

    return {
        "Negative": float(scores[0]),
        "Neutral": float(scores[1]),
        "Positive": float(scores[2])
    }

def get_label(scores_dict):
    return max(scores_dict, key=scores_dict.get)

def label_to_emoji(label):
    if label == "Positive":
        return "😄 Positive"
    elif label == "Negative":
        return "😡 Negative"
    else:
        return "😐 Neutral"

# ==============================
# UI Input
# ==============================
text = st.text_area("✍️ Enter your text:", height=150)

analyze_btn = st.button("🚀 Analyze")

# ==============================
# Main Logic
# ==============================
if analyze_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        with st.spinner("Analyzing... 🔍"):

            # VADER
            vader_result = sid.polarity_scores(text)

            # RoBERTa
            try:
                roberta_result = roberta_scores(text)
            except RuntimeError:
                st.error("⚠️ Text too long! Try shorter input.")
                st.stop()

        # ==============================
        # Display Results
        # ==============================

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 VADER")
            st.json(vader_result)

            vader_label = (
                "Positive" if vader_result['compound'] >= 0.05 else
                "Negative" if vader_result['compound'] <= -0.05 else
                "Neutral"
            )

            st.success(f"Prediction: {label_to_emoji(vader_label)}")

        with col2:
            st.subheader("🤖 RoBERTa")
            st.json(roberta_result)

            roberta_label = get_label(roberta_result)
            st.success(f"Prediction: {label_to_emoji(roberta_label)}")

        # ==============================
        # Comparison Chart
        # ==============================
        df = pd.DataFrame({
            "VADER": [
                vader_result['neg'],
                vader_result['neu'],
                vader_result['pos']
            ],
            "RoBERTa": [
                roberta_result['Negative'],
                roberta_result['Neutral'],
                roberta_result['Positive']
            ]
        }, index=["Negative", "Neutral", "Positive"])

        st.subheader("📈 Model Comparison")
        st.bar_chart(df)

        # ==============================
        # Insights
        # ==============================
        st.subheader("🧠 Insight")

        if vader_label != roberta_label:
            st.warning("⚡ Models disagree! This is an interesting case.")
        else:
            st.info("✅ Both models agree on the sentiment.")

# ==============================
# Footer
# ==============================
st.markdown("---")

st.caption("Built with ❤️ using Streamlit, VADER, and RoBERTa")
st.caption("Author: Mahmoud Youssef - [GitHub](https://github.com/Mahmoud12385) - [LinkedIn](http://www.linkedin.com/in/mahmoud-youseif)")


# 