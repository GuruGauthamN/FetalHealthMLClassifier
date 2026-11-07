from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    # binary -> prob of positive class; multi-class -> one-vs-rest
    auc = roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else roc_auc_score(y, proba, multi_class="ovr")
    avg = "binary" if proba.shape[1] == 2 else "weighted"
    return {"Accuracy": accuracy_score(y, pred), "AUC": auc,
            "Precision": precision_score(y, pred, average=avg),
            "Recall": recall_score(y, pred, average=avg),
            "F1": f1_score(y, pred, average=avg),
            "MCC": matthews_corrcoef(y, pred)}
