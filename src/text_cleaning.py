import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("../data/cleaned_clinical_trial.csv")

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stop words and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
df["cleaned_text"] = df["brief_summary"].apply(clean_text)

# Show result
print("Original Text:\n")
print(df["brief_summary"].iloc[0])

print("\n" + "="*80)

print("\nCleaned Text:\n")
print(df["cleaned_text"].iloc[0])

# Save cleaned data
df.to_csv("../data/nlp_cleaned_clinical_trial.csv", index=False)

print("\nNLP cleaned dataset saved successfully!")