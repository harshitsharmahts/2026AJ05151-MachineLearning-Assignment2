"""
Train five classifiers on the UCI Optical Recognition of Handwritten Digits
dataset (8x8 pixel images -> 64 numeric features, 10 classes, 1797 instances),
evaluate them on a held-out test split and save everything the Streamlit app
needs (models, test CSV, metrics table).

Run:  python train_models.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_COL = "target"


def load_data() -> pd.DataFrame:
    digits = load_digits(as_frame=True)
    return digits.frame.copy()        # 64 pixel columns + 'target'


def get_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": GaussianNB(var_smoothing=1e-6),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def compute_metrics(model, X, y) -> dict:
    """Same function is mirrored in app.py so the numbers match."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob, multi_class="ovr", average="macro",
                             labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X, y = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features, {y.nunique()} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    # Test data (features + target) used by the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)

    rows = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib"))
        m = compute_metrics(model, X_test, y_test)
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
        print(name, {k: round(v, 4) for k, v in m.items()})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump({"target": TARGET_COL, "features": list(X.columns),
                   "classes": [int(c) for c in sorted(y.unique())]}, f)
    print("\n", results.to_markdown(index=False) if hasattr(results, "to_markdown") else results)


if __name__ == "__main__":
    main()
