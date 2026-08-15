import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from eval import evaluate


def logistic_regression(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_all = []
    y_all = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        probs_val = model.predict_proba(X_val)[:, 1]  # Probability of class 1
        probs_all.append(probs_val)
        y_all.append(y_val)

    # Combine results from all folds
    confidences = np.concatenate(probs_all)
    y_true = np.concatenate(y_all)
    predictions = (confidences >= 0.5).astype(int)

    return evaluate(accuracies=y_true, confidences=confidences, predictions=predictions)
