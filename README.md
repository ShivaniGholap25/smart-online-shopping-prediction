# Purchase Intent Predictor

## Problem Statement
Build an end-to-end machine learning system to predict whether an online shopper will generate revenue (`Revenue = 1`) based on session behavior and engagement patterns.

## Dataset Info
- Dataset: Online Shoppers Purchasing Intention Dataset
- File: `data/online_shoppers_intention.csv`
- Target Column: `Revenue` (converted from True/False to 1/0)
- Type: Binary classification

## Folder Structure (tree)
```text
project/
├── app.py
├── evaluate.py
├── preprocess.py
├── run.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── online_shoppers_intention.csv
├── models/
│   ├── ann_model.pkl
│   ├── ensemble_model.pkl
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   ├── results.pkl
│   └── scaler.pkl
├── notebooks/
│   └── analysis.ipynb
└── static/
    ├── bouncerates_distribution.png
    ├── comparison_chart.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── logistic_confusion_matrix.png
    ├── ann_confusion_matrix.png
    ├── randomforest_confusion_matrix.png
    ├── ensemble_confusion_matrix.png
    ├── pagevalues_distribution.png
    ├── revenue_countplot.png
    ├── roc_curve.png
    ├── segment_distribution.png
    └── visitortype_vs_revenue.png
```

## Models Used
- Logistic Regression: Linear baseline model with balanced class weights for binary prediction.
- Random Forest Classifier: Tree ensemble model for non-linear feature interactions and robust performance.
- ANN (MLPClassifier): Neural network model to learn complex behavior patterns in user sessions.
- Voting Classifier (Soft Ensemble): Combines probability outputs of all base models to improve stability and accuracy.

## How To Run
```bash
pip install -r requirements.txt
python run.py
streamlit run app.py
```

## Model Comparison Table (placeholder)
| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Logistic | TBD | TBD | TBD | TBD |
| RandomForest | TBD | TBD | TBD | TBD |
| ANN | TBD | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD | TBD |

## Business Insights Summary
- Browsing users (low probability): prioritize discount nudges to increase engagement.
- Interested users (mid probability): show personalized recommendations to push toward conversion.
- Ready-to-Buy users (high probability): apply urgency-based offers to maximize immediate purchase completion.

## Future Scope
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for each model.
- Add model explainability with SHAP for decision transparency.
- Add real-time monitoring and drift detection for production behavior changes.
- Add A/B testing hooks for segment-specific campaign effectiveness.

## Technologies Used
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib
