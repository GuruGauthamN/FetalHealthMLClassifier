import glob, os, joblib
import pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from metrics_utils import compute_metrics

TARGET = "fetal_health"
st.set_page_config(page_title="ML Classifier Evaluation", layout="wide")
st.title("Classification Model Evaluation")
st.caption("Fetal Health (CTG) - 3 classes: 1 Normal, 2 Suspect, 3 Pathological - upload test data, pick a saved model, view results.")

MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
          "KNN": "knn", "Naive Bayes": "naive_bayes", "Support Vector Machine": "support_vector_machine"}
base = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load(key):
    return joblib.load(os.path.join(base, "model", f"{key}.joblib"))

up = st.file_uploader("Upload test CSV (features + 'fetal_health' column)", type="csv")
choice = st.selectbox("Select model", list(MODELS))

if up is None:
    st.info("Upload a CSV to begin (you can use test_data.csv from the repo).")
    st.stop()

df = pd.read_csv(up)
if TARGET not in df.columns:
    st.error(f"Uploaded file must contain the target column '{TARGET}'."); st.stop()
X, y = df.drop(columns=TARGET), df[TARGET]
st.write(f"Uploaded {len(df)} rows, {X.shape[1]} features.")

try:
    model = load(MODELS[choice])
    X = X[list(model.feature_names_in_)]
    m = compute_metrics(model, X, y)
except Exception as e:
    st.error(f"Could not evaluate: {e}"); st.stop()

st.subheader(f"Metrics - {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y, model.predict(X))).plot(ax=ax, colorbar=False)
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report(y, model.predict(X), output_dict=True)).T.round(3))

st.subheader("All models on uploaded data")
rows = []
for name, key in MODELS.items():
    mod = load(key)
    rows.append({"ML Model Name": name, **compute_metrics(mod, X[list(mod.feature_names_in_)], y)})
st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
