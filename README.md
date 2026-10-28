# ChurnScope – Telecom Customer Churn Prediction with Five Classifiers

**M.Tech (AIML) – Machine Learning, Assignment 2**
Anurupa Roy · 2026AK05079

| | |
|---|---|
| GitHub repository | https://github.com/anurupa-21/churn-ml-streamlit |
| Live Streamlit app | https://anurupa-churnscope.streamlit.app |

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels their plan ("churns"), and winning a customer back costs far more than keeping one. The goal of this project is to **predict, from a customer's plan details and usage behaviour, whether that customer will churn (1) or stay (0)** – a binary classification problem.

Five classical classifiers – Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and an RBF Support Vector Machine – are trained on the same data and the same preprocessing pipeline, compared on six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and served through an interactive Streamlit app that evaluates the saved models on uploaded test data without retraining.

Because only ~14% of customers churn, accuracy alone is misleading (a model that always says "stays" is 86% accurate). Model tuning therefore optimises **F1**, and **MCC** is used to pick the overall winner.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Telecom Customer Churn ("Churn in Telecom's dataset") |
| Source | Public dataset, originally from the UCI / MLC++ repository, distributed via Kaggle and the Penn Machine Learning Benchmarks (PMLB) collection |
| Instances | 5,000 customers |
| Input features | 20 (≥ 12 required) |
| Target | `churn` – 0 = stays, 1 = churns |
| Class balance | 4,293 stay (85.9%) / 707 churn (14.1%) |
| Missing values | None |

**Features**

| Group | Columns |
|---|---|
| Account | `state` (51 US states, label-coded), `account_length` (days), `area_code` (3 codes), `phone_number` (identifier) |
| Plans | `international_plan` (0/1), `voice_mail_plan` (0/1), `number_vmail_messages` |
| Day usage | `total_day_minutes`, `total_day_calls`, `total_day_charge` |
| Evening usage | `total_eve_minutes`, `total_eve_calls`, `total_eve_charge` |
| Night usage | `total_night_minutes`, `total_night_calls`, `total_night_charge` |
| International usage | `total_intl_minutes`, `total_intl_calls`, `total_intl_charge` |
| Support | `number_customer_service_calls` |

**Preprocessing and feature engineering** (`src/features.py`, applied identically inside every model's pipeline)

1. `phone_number` is dropped – it is a unique identifier with no predictive meaning.
2. Each `*_charge` column is minutes × a fixed tariff, i.e. perfectly collinear with its `*_minutes` column. The four charge columns are replaced by a single **`total_charge`**.
3. New features: **`total_calls`**, **`intl_minutes_per_call`**, and **`high_service_calls`** (1 if ≥ 4 customer-service calls). EDA showed the churn rate is ~11% for 0–3 service calls but jumps to 44% at 4 calls and ~60% at 5–6 calls.
4. `state` and `area_code` are one-hot encoded; all numeric columns are standardised (`StandardScaler`). Final model input: 71 columns.
5. 80 / 20 **stratified** train–test split (`random_state=2026`) → 4,000 training rows and 1,000 test rows. The untouched test split is saved as `test_data.csv` and is the file uploaded to the Streamlit app.

## c. GitHub repository link

https://github.com/anurupa-21/churn-ml-streamlit

```
churn-ml-streamlit/
├── app.py                    # Streamlit app (loads saved models, never retrains)
├── train_models.py           # trains, tunes, evaluates and saves all 5 models
├── requirements.txt
├── README.md
├── test_data.csv             # 1,000-row held-out test split (features + churn)
├── data/
│   └── telecom_churn.csv     # full dataset (5,000 rows)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── metrics.json          # test metrics + tuned hyper-parameters
│   └── model_comparison.csv
├── src/
│   ├── features.py           # feature engineering + ColumnTransformer
│   └── evaluation.py         # the six evaluation metrics
└── .streamlit/config.toml    # app theme
```

Each `.joblib` file is a complete scikit-learn `Pipeline` (feature builder → encoder/scaler → classifier), so the app can feed the raw uploaded CSV straight into it.

## d. Models used

All five models were tuned with `GridSearchCV` (5-fold stratified CV, scoring = F1) on the training split and then evaluated once on the 1,000-row held-out test set.

| Model | Tuned hyper-parameters | CV F1 |
|---|---|---|
| Logistic Regression | C = 0.1, class_weight = balanced | 0.598 |
| Decision Tree | criterion = gini, max_depth = 6, min_samples_leaf = 5 | 0.919 |
| KNN | n_neighbors = 5, weights = uniform | 0.602 |
| Naive Bayes (Gaussian) | var_smoothing = 1e-3 | 0.328 |
| Support Vector Machine | RBF kernel, C = 10, gamma = scale | 0.772 |

### Comparison table (test set, n = 1,000)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8640 | 0.8664 | 0.5111 | 0.8156 | 0.6284 | 0.5730 |
| Decision Tree | **0.9780** | **0.9249** | **1.0000** | **0.8440** | **0.9154** | **0.9071** |
| KNN | 0.9100 | 0.8584 | 0.8592 | 0.4326 | 0.5755 | 0.5705 |
| Naive Bayes | 0.6090 | 0.6977 | 0.2093 | 0.6383 | 0.3152 | 0.1705 |
| Support Vector Machine | 0.9440 | 0.9132 | 0.8899 | 0.6879 | 0.7760 | 0.7527 |

AUC for the SVM is computed from its `decision_function` scores; all other models use predicted probabilities.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | With `class_weight="balanced"` it catches 81.6% of churners (2nd-best recall) but only about half of its churn alerts are correct (precision 0.51). Churn here depends on thresholds and interactions rather than a straight trend – e.g. churn is only ~3% for mid-range total charges but ~31% for the top 20% of bills, and having an international plan lifts churn to 27–59% at every bill level – which a single linear boundary cannot capture, so MCC stays at 0.57. Useful as an interpretable, high-recall baseline. |
| Decision Tree | Best on every metric (MCC 0.907, F1 0.915). Churn behaviour in this data is rule-like: the tree's top splits are `total_charge` (51% of importance), voicemail usage, customer-service calls and the international plan, which matches how telecom churn is driven by bill shock and poor support experiences. Precision is 1.0 on the test set – every flagged customer really churned – while 84% of churners are caught. Limiting depth to 6 and leaf size to 5 kept it from overfitting (CV F1 0.919 ≈ test F1 0.915). |
| KNN | High accuracy (0.91) and good precision (0.86) but the lowest recall of the strong models (0.43): it misses more than half of the churners. Churners are a 14% minority, so a customer's nearest neighbours are mostly non-churners and the majority vote leans towards "stays". The 51 one-hot state columns also dilute the Euclidean distance. |
| Naive Bayes | Weakest model (MCC 0.17, accuracy 0.61). Gaussian NB assumes every feature is independent and normally distributed, which is badly violated: there are 51 binary state columns, binary plan flags, and strongly dependent features (minutes, calls and total charge). It over-predicts churn – recall 0.64 but precision only 0.21. |
| Support Vector Machine | Second best (MCC 0.753, AUC 0.913). The RBF kernel learns the non-linear churn boundary far better than Logistic Regression and gives high precision (0.89), but at a recall of 0.69 it still misses about a third of the churners the Decision Tree finds. It is also the slowest model to train and the least interpretable. |
| **Overall winner for this dataset** | **Decision Tree.** It leads on all six metrics, it is fast, and its rules are easy to explain to a retention team (e.g. "high total charge + 4 or more service calls → high churn risk"). SVM is a solid runner-up; Naive Bayes is not suitable for this feature set. |

## Streamlit app

Features (all required features plus extras):

- **CSV upload** of test data (or a one-click option to use the bundled `test_data.csv`, plus a download button for it)
- **Model selection dropdown** for the five classifiers, showing each model's tuned hyper-parameters
- **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1 and MCC as metric cards
- **Confusion matrix** (counts and row percentages) and **classification report**, plus a ROC curve
- **Compare all models** tab – metric table with best values highlighted, bar chart for any chosen metric, overlaid ROC curves
- **Predictions** tab – row-level predictions with filters (misclassified only, predicted churners) and CSV download

The uploaded CSV must contain the 20 input columns listed above and the `churn` column. The app loads the saved pipelines from `model/` and **never retrains** on uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py        # optional – regenerates model/ and test_data.csv
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud: New app → this repository → branch `main` → main file `app.py` → Deploy.
