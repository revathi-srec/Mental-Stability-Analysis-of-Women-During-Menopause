# %% STEP 1 — Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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


# %% STEP 3 — Features & Target
X = df.drop(columns=["Overall Mental Well-being"])
y = df["Overall Mental Well-being"]

print("Target distribution:\n", y.value_counts())


# %% STEP 4 — Column Alignment
X.columns = [f"q{i}" for i in range(1, 28)]
X = X.apply(pd.to_numeric)

print("STEP 4 — Columns aligned")


# %% 🚀 STEP 5 — FEATURE ENGINEERING (BOOST ACCURACY)
X['q_sum'] = X.sum(axis=1)
X['q_mean'] = X.mean(axis=1)
X['q_std'] = X.std(axis=1)

print("STEP 5 — Feature Engineering Done")


# %% 🚀 STEP 6 — SMOTE (DATA SCALING)
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


# %% 🚀 STEP 9 — HYPERPARAMETER TUNING (GridSearch)
param_grid = {
    'depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 300, 400]
}

base_model = CatBoostClassifier(
    loss_function='MultiClass',
    verbose=0,
    random_state=42
)

grid = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


# %% 🚀 STEP 10 — FINAL TRAINING WITH EARLY STOPPING
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=20,
    verbose=0
)

print("STEP 10 — Model trained with optimization")


# %% 🚀 STEP 11 — Cross Validation
scores = cross_val_score(model, X_res_scaled, y_res, cv=5)

print("\nCross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())


# %% STEP 12 — Evaluation
pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n",
      classification_report(y_test, pred, zero_division=0))


# %% STEP 13 — Save Model + Scaler
joblib.dump(model, "catboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("STEP 13 — Model & Scaler saved")


# %% STEP 14 — SAMPLE PATIENT REPORT
idx = 2

sample = X.iloc[idx:idx+1]

# Apply same feature engineering
sample['q_sum'] = sample.sum(axis=1)
sample['q_mean'] = sample.mean(axis=1)
sample['q_std'] = sample.std(axis=1)

# Apply scaling
sample_scaled = scaler.transform(sample)

pred_label = model.predict(sample_scaled)[0][0]


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
    0: "Your health is good. Maintain healthy lifestyle.",
    1: "Practice breathing exercises daily.",
    2: "Do Yoga and Meditation regularly.",
    3: "Consult a specialist immediately."
}


print("\n------ PATIENT REPORT ------")
print("Predicted Wellbeing Level:", wellbeing_map[int(pred_label)])
print("Stability Score:", f"{score_map[int(pred_label)]}/100")
print("Detected Emotional State:", emotion_map[int(pred_label)])
print("Recommendation:", recommendation_map[int(pred_label)])