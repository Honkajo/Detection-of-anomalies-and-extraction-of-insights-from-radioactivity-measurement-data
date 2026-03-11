import pandas as pd

df = pd.read_csv("clustered_samples_improved.csv")

for i in range(4):
    print(i)
    cluster_rows = df[df["cluster"] == i]

    # view
    print(cluster_rows)

    # save
    cluster_rows.to_csv(f"cluster_{i}_samples.csv", index=False)
