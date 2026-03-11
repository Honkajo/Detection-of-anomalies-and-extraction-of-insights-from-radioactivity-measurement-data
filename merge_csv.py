import pandas as pd

df1 = pd.read_csv("dataset1_preprocessed.csv")
df2 = pd.read_csv("dataset3.csv", on_bad_lines="skip")
merged = pd.merge(df1, df2, on="pseudoid", how="left")
merged.to_csv("merged_dataset.csv", index=False)
