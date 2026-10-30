import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("models/disease_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/nlp_cleaned_clinical_trial.csv")

# -----------------------------
# NLP Tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Project Information")

st.sidebar.info(
    """
    Clinical Trial Disease Category Classification

    Model: Logistic Regression

    NLP Technique:
    - Lowercase Conversion
    - Stopword Removal
    - Lemmatization
    - TF-IDF Vectorization

    Accuracy: 94.25%
    """
)

# -----------------------------
# Main Title
# -----------------------------
st.title("🏥 Clinical Trial Disease Category Classification")

st.markdown(
    """
    This application predicts the disease category from a clinical trial summary
    using NLP and Machine Learning.
    """
)

# -----------------------------
# # -----------------------------
# Clinical Trial Summary Dropdown
# -----------------------------

# Create short preview for dropdown
df["summary_preview"] = (
    df["brief_summary"]
    .astype(str)
    .str[:100] + "..."
)

selected_preview = st.selectbox(
    "Select a Clinical Trial Summary",
    df["summary_preview"]
)

selected_summary = df.loc[
    df["summary_preview"] == selected_preview,
    "brief_summary"
].iloc[0]

st.subheader("Selected Clinical Trial Summary")
st.info(selected_summary)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Disease"):

    cleaned_text = clean_text(selected_summary)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)

    st.success(
    f"🎯 Predicted Disease Category: {prediction[0]}"
)

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("📈 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric(
        "Disease Categories",
        df["source_condition_query"].nunique()
    )

with col3:
    st.metric(
        "Model Accuracy",
        "94.25%"
    )

# -----------------------------
# Disease Distribution
# -----------------------------
st.subheader("Disease Categories")

disease_counts = (
    df["source_condition_query"]
    .value_counts()
)

st.bar_chart(disease_counts)