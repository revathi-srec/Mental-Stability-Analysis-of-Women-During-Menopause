# %% STEP 1 — Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import joblib

print("STEP 1 — Libraries loaded")


# %% STEP 2 — Load Dataset
file_path = "4th year project data.xlsx"
df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)


# %% 🚀 STEP 3 — CONVERT TO BINARY
# 0 = Healthy, 1 = Stress
df["Overall Mental Well-being"] = df["Overall Mental Well-being"].apply(
    lambda x: 0 if x == 0 else 1
)

print("Binary Target Distribution:\n",
      df["Overall Mental Well-being"].value_counts())


# %% STEP 4 — Features & Target
X = df.drop(columns=["Overall Mental Well-being"])
y = df["Overall Mental Well-being"]

# rename columns
X.columns = [f"q{i}" for i in range(1, 28)]
X = X.apply(pd.to_numeric)


# %% 🚀 STEP 5 — FEATURE ENGINEERING
X['q_sum'] = X.sum(axis=1)
X['q_mean'] = X.mean(axis=1)
X['q_std'] = X.std(axis=1)

print("Feature Engineering Done")


# %% 🚀 STEP 6 — SMOTE (BALANCING)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:", X_res.shape)


# %% 🚀 STEP 7 — SCALING
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)


# %% STEP 8 — Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res_scaled, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)


# %% 🚀 STEP 9 — CATBOOST MODEL (OPTIMIZED)
model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=4,
    loss_function='Logloss',
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

print("Model trained")


# %% 🚀 STEP 10 — Cross Validation
scores = cross_val_score(model, X_res_scaled, y_res, cv=5)

print("\nCross Validation Accuracy:", scores)
print("Average CV Accuracy:", scores.mean())


# %% STEP 11 — Evaluation
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n",
      classification_report(y_test, pred))


# %% STEP 12 — Save Model
joblib.dump(model, "catboost_binary_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully")


# %% STEP 13 — SAMPLE PATIENT REPORT
idx = 2

sample = X.iloc[idx:idx+1]

# apply same feature engineering
sample['q_sum'] = sample.sum(axis=1)
sample['q_mean'] = sample.mean(axis=1)
sample['q_std'] = sample.std(axis=1)

sample_scaled = scaler.transform(sample)

pred_label = model.predict(sample_scaled)[0]


# 🔥 BINARY OUTPUT
status_map = {
    0: "Healthy",
    1: "Stress Detected"
}

score_map = {
    0: 85,
    1: 50
}

recommendation_map = {
    0: "Maintain healthy lifestyle and regular exercise.",
    1: "Practice meditation, reduce stress, and consult a specialist if needed."
}


print("\n------ PATIENT REPORT ------")
print("Health Status:", status_map[int(pred_label)])
print("Health Score:", f"{score_map[int(pred_label)]}/100")
print("Recommendation:", recommendation_map[int(pred_label)])