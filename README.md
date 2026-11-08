# Handwritten Digit Classification – ML Assignment 2 (M.Tech AIML, BITS Pilani WILP)

## a. Problem statement
Build and compare five classical classification models on a public UCI dataset, evaluate them with
six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and expose the trained models through an
interactive Streamlit web app deployed on Streamlit Community Cloud.

The task here is to recognise a handwritten digit (0–9) from an 8×8 grayscale pixel image.

## b. Dataset description
- **Source:** UCI Machine Learning Repository – *Optical Recognition of Handwritten Digits* (loaded through `sklearn.datasets.load_digits`, which ships the 1797-image test portion of the full 5620-image UCI dataset).
- **Type:** Multi-class classification (10 classes: digits 0–9).
- **Instances:** 1797
- **Features:** 64 numeric pixel-intensity features (`pixel_0_0` … `pixel_7_7`, values 0–16)
- **Target:** `target` (the digit label). Classes are nearly balanced (~180 samples each).
- **Split:** 75 % train / 25 % test (stratified, `random_state=42`). The 25 % test split (450 rows) is saved as `test_data.csv` and is what the app evaluates on.
- **Metric averaging:** Precision / Recall / F1 use *macro* averaging; AUC uses one-vs-rest macro averaging.

## c. GitHub repository link
[https://github.com/harshitsharmahts/2026AJ05151-MachineLearning-Assignment2](https://github.com/harshitsharmahts/2026AJ05151-MachineLearning-Assignment2)

Repository layout:
```
2026AJ05151-MachineLearning-Assignment2/
├── app.py                 # Streamlit app
├── train_models.py        # training + evaluation script
├── requirements.txt
├── README.md
├── test_data.csv          # held-out test data (features + target)
├── model_comparison.csv   # metrics table produced by train_models.py
└── model/                 # saved models (.joblib) + meta.json
```

## d. Models used

### Comparison table (test set, 450 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9778 | 0.9991 | 0.9778 | 0.9777 | 0.9776 | 0.9753 |
| Decision Tree | 0.8200 | 0.9008 | 0.8197 | 0.8190 | 0.8187 | 0.8001 |
| KNN | 0.9644 | 0.9922 | 0.9651 | 0.9642 | 0.9642 | 0.9606 |
| Naive Bayes | 0.8667 | 0.9661 | 0.8760 | 0.8660 | 0.8669 | 0.8528 |
| Support Vector Machine | 0.9800 | 0.9992 | 0.9804 | 0.9798 | 0.9799 | 0.9778 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong (97.8 % accuracy) even though it is a linear model; scaled pixel features are close to linearly separable per class. Fast to train and nearly tied with SVM. |
| Decision Tree | Weakest model (82 % accuracy, AUC 0.90). Axis-aligned splits on individual pixels generalise poorly and the tree overfits; hard-label probabilities also lower AUC. |
| KNN | Very good (96.4 %); similar digits lie close together in pixel space. Slightly behind LR/SVM and slower at prediction time because it stores all training data. |
| Naive Bayes | Moderate (86.7 %). The independence assumption is violated because neighbouring pixels are strongly correlated, but AUC (0.966) shows its probability ranking is still decent. |
| Support Vector Machine | Best overall (98.0 % accuracy, MCC 0.978). The RBF kernel captures non-linear stroke patterns and gives the highest AUC. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF)** – highest on every metric, with Logistic Regression a very close second. |

## e. Streamlit app features
**Live app:** [https://2026aj05151-machinelearning-assignment2.streamlit.app](https://2026aj05151-machinelearning-assignment2.streamlit.app)

- CSV upload (use `test_data.csv`)
- Model selection dropdown (5 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Optional side-by-side comparison of all models
- Models are loaded from `model/`; **no retraining** happens in the app.
