import pandas as pd

df = pd.read_csv("../data/clinical_trial.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())
print(df["source_condition_query"].head())

print("\nUnique Diseases:")
print(df["source_condition_query"].nunique())
print(df["conditions"].head())

print(df["source_condition_query"].head(10))

print("\nUnique values:")
print(df["source_condition_query"].unique())