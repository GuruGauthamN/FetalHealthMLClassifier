"""Train 5 classifiers on one dataset, save models + metrics + test_data.csv."""
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

TARGET = "fetal_health"
# Kaggle "Fetal Health Classification" (CTG data, 2126 rows, 21 features, 3 classes:
# 1 = Normal, 2 = Suspect, 3 = Pathological). Keep fetal_health.csv next to this script.
df = pd.read_csv("fetal_health.csv")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
df[TARGET] = df[TARGET].astype(int)

X, y = df.drop(columns=TARGET), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    joblib.dump(m, f"model/{name.replace(' ', '_').lower()}.joblib")
    rows.append({"ML Model Name": name, **compute_metrics(m, X_te, y_te)})
res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics.csv", index=False)
print(res.to_markdown(index=False))
