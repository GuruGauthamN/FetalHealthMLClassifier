# Fetal Health Classification (CTG data) – 5 ML Models + Streamlit App

## a. Problem Statement
Cardiotocography (CTG) records the fetal heart rate and uterine contractions during pregnancy. Obstetricians read these traces to judge fetal well-being, which is subjective and time-pressured. The goal is to build and compare five machine learning classifiers that predict fetal health from 21 numeric CTG features, as **Normal (1)**, **Suspect (2)** or **Pathological (3)**, and to expose the trained models through an interactive Streamlit app that evaluates them on an uploaded test CSV.

## b. Dataset Description 
- **Source:** Kaggle – *Fetal Health Classification* (derived from the UCI Cardiotocography data).
- **Instances:** 2126 | **Features:** 21 (all numeric) | **Target:** `fetal_health` (multi-class, 3 classes).
- **Features:** baseline_value, accelerations, fetal_movement, uterine_contractions, light/severe/prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability, and 9 histogram features (width, min, max, number_of_peaks, number_of_zeroes, mode, mean, median, variance, tendency).
- **Class distribution:** Normal 1655 (77.8%), Suspect 295 (13.9%), Pathological 176 (8.3%) – clearly imbalanced. No missing values.
- **Split:** 80/20 stratified (`random_state=42`) → 1700 training / 426 test rows. The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside a pipeline for Logistic Regression, KNN and SVM (fit on training data only). Decision Tree and Naive Bayes use raw features. The scaler is saved inside each model file.

## c. GitHub Repository Link  
`https://github.com/GuruGauthamN/FetalHealthMLClassifier`

# Streamlit App link: https://fetalhealthmlclassifier.streamlit.app/

```
ML Assignment 2/
├── app.py                 # Streamlit app
├── train.py               # trains 5 models, saves them + test_data.csv
├── metrics_utils.py       # shared metric computation
├── requirements.txt
├── README.md
├── fetal_health.csv       # full dataset
├── test_data.csv          # held-out test set (features + fetal_health)
└── model/                 # saved pipelines (.joblib) + metrics.csv
```
Run locally: `pip install -r requirements.txt && python train.py && streamlit run app.py`

## d. Models Used 
Metrics are computed on the 426-row test set. Because this is a 3-class problem, **Precision / Recall / F1 are weighted averages** across classes, and **AUC is one-vs-rest** (macro-averaged) from predicted probabilities. MCC is the multi-class Matthews correlation coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8850 | 0.9614 | 0.8893 | 0.8850 | 0.8855 | 0.6834 |
| Decision Tree | 0.9014 | 0.8580 | 0.8983 | 0.9014 | 0.8994 | 0.7252 |
| KNN | 0.8709 | 0.9393 | 0.8615 | 0.8709 | 0.8629 | 0.6191 |
| Naive Bayes (Gaussian) | 0.7981 | 0.8694 | 0.8619 | 0.7981 | 0.8174 | 0.5710 |
| Support Vector Machine (RBF) | 0.8944 | 0.9654 | 0.8884 | 0.8944 | 0.8901 | 0.6968 |

### Observations 
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, stable baseline: 88.5% accuracy and the third-highest AUC (0.961). A linear boundary on standardised features already separates the classes well, and the probabilities are well ranked. It trails the tree and SVM slightly on MCC, so it is weaker on the minority classes. |
| Decision Tree | Highest accuracy (0.901), F1 and MCC (0.725), but a low AUC (0.858). Leaves give near-hard probabilities, so class ranking is poor, and an unpruned tree can overfit. Its lead over SVM is about 3 test samples, so it is within noise. |
| KNN | Mid-table: accuracy 0.871, AUC 0.939, and the lowest MCC after Naive Bayes (0.619). With 78% Normal samples, neighbourhoods are dominated by the majority class, which hurts Suspect/Pathological recall. Sensitive to k and to scaling. |
| Naive Bayes | Weakest on accuracy (0.798), F1 and MCC (0.571). The conditional-independence assumption is a poor fit here, since many features (histogram mean/median/mode, variability measures) are strongly correlated. Its AUC (0.869) is still reasonable, so the ranking is better than its hard predictions. |
| Support Vector Machine | Best AUC (0.965) and second-best accuracy (0.894), F1 (0.890) and MCC (0.697). The RBF kernel captures non-linear structure while scaling keeps features comparable. The most consistent model across all six metrics. |

**Overall winner for this dataset: Support Vector Machine (RBF).** It has the best AUC and is near the top on every other metric. The Decision Tree is marginally ahead on accuracy/MCC, but the gap is only a few samples on one split, and the tree is the weakest model on AUC. Note: all results come from a single 80/20 split, so cross-validation would be needed before claiming a definitive ranking. Class imbalance (e.g. `class_weight='balanced'`) is a natural next step.

## Streamlit App Features
- CSV upload of test data (must contain all 21 feature columns and `fetal_health`)
- Model selection dropdown (5 saved models, loaded from `model/` – **no retraining**)
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison table of all 5 models on the uploaded data
