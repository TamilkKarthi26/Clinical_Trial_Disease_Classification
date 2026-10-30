import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../data/nlp_cleaned_clinical_trial.csv")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["cleaned_text"])

print("TF-IDF Shape:")
print(X.shape)

print("\nFirst 20 Features:")
print(vectorizer.get_feature_names_out()[:20])