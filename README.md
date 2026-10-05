# AI-Powered Telecom Customer Churn Prediction and Retention Analytics

**IBM SkillsBuild Data Analytics with AI Academic Internship Program
2026**\
**Author:** Deval Shah

## 1. Project overview

This project develops an AI/ML-based customer churn prediction and
retention analytics system for a fictional telecommunications company.
The system analyzes customer demographics, services, contract
information, tenure and billing behavior to estimate the probability
that a customer will leave.

The project demonstrates the complete analytics lifecycle: data
collection, cleaning, EDA, feature engineering, preprocessing, model
comparison, evaluation, explainability, risk segmentation and retention
recommendations.

## 2. Problem statement

Customer churn can reduce recurring revenue and increase acquisition
costs. A retention team needs an early-warning system that can identify
high-risk customers before they leave. The project therefore treats
churn as a binary classification problem and emphasizes ROC-AUC, recall,
F1-score and actionable risk segmentation.

## 3. Dataset

**Dataset:** IBM Telco Customer Churn sample dataset\
**Records:** 7,043 customers\
**Original columns:** 21\
**Target:** `Churn` (Yes/No)

-   IBM repository: https://github.com/IBM/telco-customer-churn-on-icp4d
-   Direct CSV:
    https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
-   Kaggle dataset page:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The IBM repository describes the sample as a telecom churn use case and
provides the CSV used by this notebook.

## 4. Technologies used

-   Python
-   Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Logistic Regression
-   Random Forest
-   Gradient Boosting
-   Permutation importance

## 5. Machine-learning workflow

1.  Load the IBM dataset.
2.  Convert `TotalCharges` to numeric and handle invalid blank values.
3.  Remove duplicate records and the customer identifier from modeling.
4.  Perform exploratory data analysis.
5.  Engineer service-count and tenure-group features.
6.  Split data using a stratified 80/20 train-test split.
7.  Impute missing values, scale numeric fields and one-hot encode
    categorical fields.
8.  Train Logistic Regression, Random Forest and Gradient Boosting
    models.
9.  Compare accuracy, precision, recall, F1, ROC-AUC and PR-AUC.
10. Validate model ranking with 5-fold stratified cross-validation.
11. Evaluate the selected model on the untouched test set.
12. Analyze thresholds for retention-team use.
13. Use permutation importance to identify important churn drivers.
14. Assign Low/Medium/High risk segments.
15. Generate practical retention recommendations.

## 6. How to run

### Option A --- Jupyter Notebook

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

macOS/Linux:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Start Jupyter:

``` bash
jupyter notebook
```

Open `DevalShah_TelcoCustomerChurnAI.ipynb` and run all cells.

### Offline/local dataset option

Download `Telco-Customer-Churn.csv` from the dataset source and place it
beside the notebook. If the online IBM URL is unavailable, the notebook
automatically tries the local CSV.

## 7. Expected project outputs

The notebook produces:

-   Dataset quality summary
-   Churn distribution analysis
-   Contract/service/billing visualizations
-   Model comparison table
-   5-fold cross-validation ROC-AUC table
-   Confusion matrix
-   ROC and Precision-Recall curves
-   Threshold analysis
-   Permutation feature-importance table and chart
-   Customer risk segmentation
-   Example retention recommendations

## 8. Interpretation

For a retention use case, a false negative can be more costly than a
false positive because a missed high-risk customer may leave before the
business can intervene. Therefore, the notebook reports recall and
provides threshold analysis instead of relying only on accuracy.

The model should be treated as a decision-support system. It should not
automatically deny service, cancel accounts, or make high-impact
decisions without human review.

## 9. Limitations and future improvements

-   The dataset is a fictional sample and may not represent a real
    telecom operator.
-   Model performance can change with the train/test split, software
    versions and preprocessing choices.
-   The dataset does not contain every real-world signal such as
    customer sentiment, network quality or support-ticket text.
-   Future work could add SHAP explanations, hyperparameter
    optimization, model monitoring, a Streamlit dashboard,
    cost-sensitive threshold optimization and automated retraining.
-   Demographic variables should receive a fairness review before
    operational use.

## 10. Academic integrity note

This repository is intended as an educational internship project. The
numerical evaluation results should be generated by running the supplied
notebook. Do not replace the notebook-generated results with unsupported
claims.