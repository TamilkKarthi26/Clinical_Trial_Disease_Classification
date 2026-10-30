# Clinical Trial Disease Category Classification

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning to predict the disease or medical condition associated with a clinical trial based on its brief summary.

The application allows users to select a clinical trial summary from a dropdown and predicts its disease category.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

## Project Workflow
1. Data Loading
2. Data Cleaning
3. Missing Value Removal
4. Duplicate Removal
5. Text Preprocessing
6. Lowercase Conversion
7. Stopword Removal
8. Lemmatization
9. TF-IDF Vectorization
10. Logistic Regression Model Training
11. Model Evaluation
12. Streamlit Application Development

## Dataset
Dataset source: GUVI.

The dataset contains clinical trial summaries and their associated disease categories.

The dataset files are not included in this repository. Please obtain the dataset from the authorized GUVI source.

## Machine Learning Model
**Algorithm:** Logistic Regression

**Text Vectorization:** TF-IDF

**Input Feature:** `brief_summary`

**Target Variable:** `source_condition_query`

**Reported Accuracy:** 94.25%

## Streamlit Application
The application provides:
- Clinical trial summary selection through a dropdown.
- Display of the selected clinical trial summary.
- Disease category prediction.
- Dataset statistics.
- Disease category distribution visualization.

## Project Structure

```text
Clinical_Trial_Disease_Classification/
│
├── app.py
├── data/
├── models/
│   ├── disease_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── data_loading.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── save_model.py
│   ├── text_cleaning.py
│   └── tfidf_vectorization.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Clinical_Trial_Disease_Classification

pip install -r requirements.txt
```

Obtain the authorized dataset from GUVI and place the required file in the `data/` folder.

Download the required NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Run the Application

```bash
streamlit run app.py
```

## Project Objective
To demonstrate how Natural Language Processing and Machine Learning can be applied to clinical trial text classification.

## Author
Karthikeyan Rajendran

GitHub: https://github.com/TamilkKarthi26