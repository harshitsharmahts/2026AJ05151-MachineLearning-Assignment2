import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Digit Classifier Lab", page_icon="🔢", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_data
def load_meta():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        return json.load(f)


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=model.classes_),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


meta = load_meta()
TARGET = meta["target"]

st.title("🔢 Handwritten Digit Classifier Lab")
st.caption("ML Assignment 2 · UCI Optical Recognition of Handwritten Digits · "
           "five pre-trained classifiers evaluated on your test CSV (no retraining).")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Must contain the 64 pixel columns and the 'target' column.")
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    compare_all = st.checkbox("Also compare all 5 models", value=True)
    st.markdown("---")
    st.markdown("Tip: use **test_data.csv** from the GitHub repo.")

if uploaded is None:
    st.info("⬅️ Upload the test CSV in the sidebar to begin.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in meta["features"] + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing[:10]}")
    st.stop()

X, y = df[meta["features"]], df[TARGET]
st.success(f"Loaded {len(df)} rows · {X.shape[1]} features · {y.nunique()} classes present")
with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

# ---- Selected model ----
model = load_model(model_name)
pred, metrics = evaluate(model, X, y)

st.subheader(f"Results — {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = pd.DataFrame(classification_report(y, pred, output_dict=True, zero_division=0)).T
    st.dataframe(report.round(3), use_container_width=True)

# ---- Comparison of all models ----
if compare_all:
    st.subheader("All models on the uploaded data")
    rows = []
    for name in MODEL_FILES:
        _, m = evaluate(load_model(name), X, y)
        rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
