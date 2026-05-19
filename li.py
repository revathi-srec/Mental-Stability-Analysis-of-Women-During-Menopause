# %% STEP 1 — Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

import joblib

print("STEP 1 — Libraries loaded")


# %% STEP 2 — Load Dataset
file_path = "4th year project data.xlsx"
df = pd.read_excel(file_path)

print("Dataset shape:", df.shape)


# %% STEP 3 — Features & Target
X = df.drop(columns=["Overall Mental Well-being"])
y = df["Overall Mental Well-being"]

print("Target distribution:\n", y.value_counts())


# %% STEP 4 — Column Alignment
X.columns = [f"q{i}" for i in range(1, 28)]
X = X.apply(pd.to_numeric)

print("STEP 4 — Columns aligned")


# %% 🚀 STEP 5 — FEATURE ENGINEERING
X['q_sum'] = X.sum(axis=1)
X['q_mean'] = X.mean(axis=1)
X['q_std'] = X.std(axis=1)

print("STEP 5 — Feature Engineering Done")


# %% 🚀 STEP 6 — SMOTE
smote = SMOTE(k_neighbors=3, random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Before SMOTE:", X.shape)
print("After SMOTE:", X_res.shape)


# %% 🚀 STEP 7 — FEATURE SCALING
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

print("STEP 7 — Scaling Done")


# %% STEP 8 — Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res_scaled, y_res,
    test_size=0.2,
    random_state=42,
    stratify=y_res
)

print("STEP 8 — Data split")


# %% 🚀 STEP 9 — LIGHTGBM MODEL (TUNED)
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("STEP 9 — Model trained")


# %% 🚀 STEP 10 — Cross Validation
scores = cross_val_score(model, X_res_scaled, y_res, cv=5)

print("\nCross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())


# %% STEP 11 — Evaluation
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n",
      classification_report(y_test, pred, zero_division=0))


# %% STEP 12 — Save Model
joblib.dump(model, "lightgbm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("STEP 12 — Model saved successfully")


# %% STEP 13 — SAMPLE PATIENT REPORT
idx = 2

sample = X.iloc[idx:idx+1]

# Apply same feature engineering
sample['q_sum'] = sample.sum(axis=1)
sample['q_mean'] = sample.mean(axis=1)
sample['q_std'] = sample.std(axis=1)

# Scale
sample_scaled = scaler.transform(sample)

pred_label = model.predict(sample_scaled)[0]


# Mapping
wellbeing_map = {
    0: "Good Mental Health",
    1: "Mild Stress",
    2: "Moderate Stress",
    3: "Severe Stress"
}

emotion_map = {
    0: "Emotionally Stable",
    1: "Slight Emotional Disturbance",
    2: "Emotional Imbalance Detected",
    3: "High Emotional Distress"
}

score_map = {
    0: 85,
    1: 65,
    2: 45,
    3: 25
}

recommendation_map = {
    0: "Maintain healthy lifestyle.",
    1: "Practice breathing exercises daily.",
    2: "Do Yoga and Meditation.",
    3: "Consult specialist immediately."
}


print("\n------ PATIENT REPORT ------")
print("Predicted Wellbeing Level:", wellbeing_map[int(pred_label)])
print("Stability Score:", f"{score_map[int(pred_label)]}/100")
print("Detected Emotional State:", emotion_map[int(pred_label)])
print("Recommendation:", recommendation_map[int(pred_label)])