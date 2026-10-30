import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/nlp_cleaned_clinical_trial.csv")

X_text = df["cleaned_text"]
y = df["source_condition_query"]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "models/disease_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model and Vectorizer saved successfully!")