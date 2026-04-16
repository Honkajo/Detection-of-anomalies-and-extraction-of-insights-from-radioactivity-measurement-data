import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# 1. Load the datasets

normal_df = pd.read_csv("dataset1_final.csv")
outlier_df = pd.read_csv("dataset1_outliers.csv")

print("Normal data shape:", normal_df.shape)
print("Outlier data shape:", outlier_df.shape)


# 2. Drop columns that should not be features

drop_cols = ["pseudoid"]

X_normal = normal_df.drop(columns=drop_cols, errors="ignore")
X_outlier = outlier_df.drop(columns=drop_cols, errors="ignore")

# Keep same columns in both
common_cols = X_normal.columns.intersection(X_outlier.columns)
X_normal = X_normal[common_cols]
X_outlier = X_outlier[common_cols]

print("Feature count before encoding:", len(common_cols))


# 3. Split normal data

X_train, X_test_normal = train_test_split(
    X_normal,
    test_size=0.2,
    random_state=42
)


# 4. Build final test set

X_test = pd.concat([X_test_normal, X_outlier], axis=0).reset_index(drop=True)

y_test = np.concatenate([
    np.zeros(len(X_test_normal), dtype=int),   # normal
    np.ones(len(X_outlier), dtype=int)         # anomaly
])

print("Training normal samples:", len(X_train))
print("Testing normal samples:", len(X_test_normal))
print("Final test set shape:", X_test.shape)

print("Test label distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# 5. Identify numeric and categorical columns

numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)


# 6. Impute missing values

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numeric_cols]),
    columns=numeric_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[numeric_cols]),
    columns=numeric_cols,
    index=X_test.index
)

X_train_cat = pd.DataFrame(index=X_train.index)
X_test_cat = pd.DataFrame(index=X_test.index)

if categorical_cols:
    X_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train[categorical_cols]),
        columns=categorical_cols,
        index=X_train.index
    )

    X_test_cat = pd.DataFrame(
        cat_imputer.transform(X_test[categorical_cols]),
        columns=categorical_cols,
        index=X_test.index
    )

# Combine numeric and categorical data
X_train_imputed = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_imputed = pd.concat([X_test_num, X_test_cat], axis=1)


# 7. One-hot encode categorical columns

X_train_encoded = pd.get_dummies(X_train_imputed, drop_first=False)
X_test_encoded = pd.get_dummies(X_test_imputed, drop_first=False)

# Align test columns to match train columns
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join="left", axis=1, fill_value=0
)

print("Feature count after encoding:", X_train_encoded.shape[1])
print("\nNaNs in X_train_encoded:", X_train_encoded.isna().sum().sum())
print("NaNs in X_test_encoded:", X_test_encoded.isna().sum().sum())


# 8. Scale

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


# 9. Train LOF with best parameters

lof = LocalOutlierFactor(
    n_neighbors=12,
    contamination=0.04,   # not important when using custom threshold
    novelty=True
)

lof.fit(X_train_scaled)


# 10. Compute anomaly scores

# Higher score = more anomalous
anomaly_scores = -lof.decision_function(X_test_scaled)


# 11. Apply best threshold

best_percentile = 91
best_threshold = np.percentile(anomaly_scores, best_percentile)

y_pred = (anomaly_scores >= best_threshold).astype(int)

print(f"\nUsing threshold at {best_percentile}th percentile")
print(f"Threshold value: {best_threshold:.6f}")


# 12. Evaluate

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nPredicted label distribution:")
print(pd.Series(y_pred).value_counts().sort_index())


# 13. Plot confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot()
plt.title("LOF Confusion Matrix (Best Parameters)")
plt.show()


# 14. Save results

results_df = X_test.reset_index(drop=True).copy()
results_df["true_label"] = y_test
results_df["pred_label"] = y_pred
results_df["anomaly_score"] = anomaly_scores
results_df["used_percentile"] = best_percentile
results_df["used_threshold"] = best_threshold

results_df.to_csv("lof_best_results.csv", index=False)
print("\nSaved results to lof_best_results.csv")