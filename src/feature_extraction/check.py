import pandas as pd

df = pd.read_csv("extreme_values_with_annotations.csv")
print(df.head(20))

print(df["aami_annotation"].value_counts())

print(df[df["aami_annotation"] == "N"])

summary = df.groupby(["attribute", "aami_annotation"]).size().unstack(fill_value=0)
print(summary)