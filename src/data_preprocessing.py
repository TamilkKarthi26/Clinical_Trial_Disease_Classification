import pandas as pd

df = pd.read_csv("../data/clinical_trial.csv")

print("Original Shape:", df.shape)

print("\nMissing Values in brief_summary:")
print(df["brief_summary"].isnull().sum())

print("\nMissing Values in source_condition_query:")
print(df["source_condition_query"].isnull().sum())

df = df.dropna(subset=["brief_summary", "source_condition_query"])

print("\nShape After Removing Missing Values:")
print(df.shape)

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

df = df[["brief_summary", "source_condition_query"]]

df.to_csv("../data/cleaned_clinical_trial.csv", index=False)

print("\nCleaned dataset saved successfully!")
print(df["brief_summary"].iloc[0])