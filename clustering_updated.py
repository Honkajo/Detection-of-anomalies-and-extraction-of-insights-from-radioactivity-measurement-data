import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import zscore

df = pd.read_csv("dataset1_preprocessed.csv")

# Save pseudoid for later
pseudoid = None
if "pseudoid" in df.columns:
    pseudoid = df["pseudoid"]
    df_model = df.drop(columns=["pseudoid"])
else:
    df_model = df.copy()

# Presence indicators
presence = df_model.notna().astype(int)
presence.columns = [c + "_present" for c in df_model.columns]

# Separate numeric & categorical
numeric_cols = df_model.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = ["geometry", "quantumunit", "vacuumed"]
# Handle categorical variables
if categorical_cols:
    df_cat = pd.get_dummies(df_model[categorical_cols], dummy_na=True)
else:
    df_cat = pd.DataFrame(index=df_model.index)

# Impute numeric missing values
imputer = SimpleImputer(strategy="median")
df_num = pd.DataFrame(imputer.fit_transform(df_model[numeric_cols]), columns=numeric_cols)

# Remove extreme outliers
z = np.abs(zscore(df_num))
df_num_clean = df_num[(z < 3).all(axis=1)]
df_cat_clean = df_cat.loc[df_num_clean.index]
presence_clean = presence.loc[df_num_clean.index]

# Scale numeric features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df_num_clean)

# Combine all features
X_final = np.hstack([X_num_scaled, df_cat_clean.values, presence_clean.values])

# Dimensionality reduction
pca = PCA(n_components=0.9, svd_solver='full')  # retain 90% variance
X_reduced = pca.fit_transform(X_final)

"""
# Try multiple k to find silhouette scores
scores = []
ks = list(range(2, 11))
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    scores.append(score)
    print(f"k={k}, silhouette={score:.3f}")

# Plot silhouette vs k
plt.figure()
plt.plot(ks, scores, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette scores for different k")
plt.show()

# Final clustering (choose best k from above)
best_k = ks[np.argmax(scores)]
print(f"\nBest k according to silhouette: {best_k}")
"""
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
final_clusters = kmeans.fit_predict(X_reduced)

df_clean = df.loc[df_num_clean.index].copy()
df_clean["cluster"] = final_clusters

# Cluster profiling
print("\nCluster counts:")
print(df_clean["cluster"].value_counts())

print("\nCluster profiles (numeric means):")
print(df_clean.groupby("cluster")[numeric_cols].mean())

# 2D visualization
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_final)

print("Number of PCA components:", pca.n_components_)

print("Explained variance ratio per component:")
print(pca.explained_variance_ratio_)

print("Total variance explained:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8,6))
for cluster_id in np.unique(final_clusters):
    idx = final_clusters == cluster_id
    plt.scatter(X_vis[idx, 0], X_vis[idx, 1], label=f"Cluster {cluster_id}", alpha=0.6)
plt.legend()
plt.title("Clusters visualized in 2D PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Save clustered data
if pseudoid is not None:
    df_clean["pseudoid"] = pseudoid.loc[df_clean.index]

df_clean.to_csv("clustered_samples_improved.csv", index=False)