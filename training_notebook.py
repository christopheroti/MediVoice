#!/usr/bin/env python
# coding: utf-8

# """
# MediVoice AI - Triage Model Training
# TF-IDF + Logistic Regression
# 
# Input:
#     medivoice_triage_dataset_augmented.csv
# 
# Output:
#     models/medivoice_triage_model.joblib
# 
# Metrics:
#     Accuracy
#     Precision
#     Recall
#     F1-score
#     Confusion matrix
#     Classification report
# """

# In[1]:


import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


#  1. CONFIGURATION

# In[2]:


DATASET_PATH = "data/medivoice_triage_dataset_augmented.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "medivoice_triage_model.joblib")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

RANDOM_STATE = 42


# 2. LOAD DATA

# In[3]:


print("=" * 70)
print("MEDIVOICE AI - TRIAGE MODEL TRAINING")
print("=" * 70)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Place medivoice_triage_dataset_augmented.csv inside your "
        "project's data/ folder."
    )

df = pd.read_csv(DATASET_PATH)

required_columns = ["text", "chief_complaint", "triage_level"]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["text", "chief_complaint", "triage_level"])

# Combine the patient's description and chief complaint.
df["input_text"] = (
    df["chief_complaint"].astype(str)
    + " "
    + df["text"].astype(str)
)

# Normalize labels.
df["triage_level"] = df["triage_level"].str.lower().str.strip()

allowed_labels = {"low", "moderate", "high"}

invalid_labels = set(df["triage_level"].unique()) - allowed_labels

if invalid_labels:
    raise ValueError(f"Unexpected triage labels: {invalid_labels}")

print(f"\nTotal samples: {len(df)}")

print("\nClass distribution:")
print(df["triage_level"].value_counts())


# 3. TRAIN / TEST SPLIT

# In[4]:


X = df["input_text"]
y = df["triage_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


#  4. BUILD MODEL

# In[5]:


model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# 5. TRAIN

# In[6]:


print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# 6. PREDICTIONS

# In[7]:


y_pred = model.predict(X_test)


# 7. METRICS

# In[8]:


accuracy = accuracy_score(y_test, y_pred)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro Precision:    {precision_macro:.4f}")
print(f"Macro Recall:       {recall_macro:.4f}")
print(f"Macro F1-score:     {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall:    {recall_weighted:.4f}")
print(f"Weighted F1-score:  {f1_weighted:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["low", "moderate", "high"],
        zero_division=0,
    )
)


# 8. CONFUSION MATRIX

# In[9]:


labels = ["low", "moderate", "high"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

print("\nConfusion Matrix:")
print(cm)

os.makedirs(MODEL_DIR, exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["LOW", "MODERATE", "HIGH"],
)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, values_format="d")
ax.set_title("MediVoice AI Triage - Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close()

print(f"\nConfusion matrix saved to: {CM_PATH}")


# 9. HIGH-RISK PERFORMANCE

# In[10]:


#This is particularly important for MediVoice.
# We check how well the model identifies HIGH cases.


high_true = (y_test == "high").astype(int)
high_pred = (y_pred == "high").astype(int)

high_precision, high_recall, high_f1, _ = (
    precision_recall_fscore_support(
        high_true,
        high_pred,
        average="binary",
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("HIGH-RISK CLASS PERFORMANCE")
print("=" * 70)

print(f"HIGH Precision: {high_precision:.4f}")
print(f"HIGH Recall:    {high_recall:.4f}")
print(f"HIGH F1-score:  {high_f1:.4f}")


# 10. SAVE MODEL

# In[11]:


model_package = {
    "model": model,
    "labels": labels,
    "version": "1.0",
    "algorithm": "TF-IDF + Logistic Regression",
    "target": "triage_level",
}

joblib.dump(model_package, MODEL_PATH)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"Model: {MODEL_PATH}")


# 11. TEST SAMPLE PREDICTIONS

# In[12]:


test_cases = [
    "I have sudden crushing chest pain spreading to my left arm with sweating",
    "I have had a mild runny nose and sneezing since yesterday",
    "I have fever, cough and mild shortness of breath for three days",
    "I suddenly cannot speak properly and my right arm feels weak",
    "I have mild stomach discomfort after eating",
]

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for text in test_cases:

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_

    confidence = probabilities.max()

    print("\nInput:")
    print(text)

    print(f"Prediction: {prediction.upper()}")
    print(f"Model probability: {confidence:.4f}")

    print("Probabilities:")

    for class_name, probability in zip(classes, probabilities):
        print(f"  {class_name.upper():8s}: {probability:.4f}")

print("\nTraining finished successfully.")


# In[ ]:




