from sklearn.metrics import brier_score_loss, precision_recall_curve, auc, roc_curve
import numpy as np
from .utils import normalize_text

# adapted from https://towardsdatascience.com/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d/
def calculate_ece(scores, confidences, M=10):
    bin_boundaries = np.linspace(0.0, 1.0, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i in range(M):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]

        # Include lower bound and exclude upper bound, except for last bin
        if i == M - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if np.any(in_bin):
            prob_in_bin = np.mean(in_bin)
            accuracy_in_bin = scores[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece

def compute_em(prediction: str, truth: str) -> int:
    """
    Compute Exact Match (EM) score between prediction and truth.
    EM is 1 if the prediction matches the truth exactly, otherwise 0.
    """
    normalized_prediction = normalize_text(prediction)
    normalized_truth = normalize_text(truth)
    return int(normalized_prediction == normalized_truth)


def calculate_macro_ce(accuracies, confidences, give_array=False):
    classes = np.unique(accuracies)
    eces = []

    for c in classes:
        mask = (accuracies == c)
        ece = calculate_ece(accuracies[mask], confidences[mask])
        eces.append(ece)

    eces = np.array(eces)

    if give_array:
        return eces

    return eces.mean()

def calculate_roc_auc(y_true, y_score, pos_label=1):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC)

    Parameters:
        y_true : array-like of shape (n_samples,)
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.

            y_score : array-like of shape (n_samples,)
            Target confidences values.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_rp_auc(y_true, y_scores, pos_label=1):
    """
    Calculate the Area Under the Curve (AUC) for the Precision-Recall curve.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.

            y_score : array-like of shape (n_samples,)
            Target confidences values.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    auc_score = auc(recall, precision)
    return auc_score

def calculate_brier_score(y_true, y_prob, pos_label=1):
    brier_score = brier_score_loss(y_true, y_prob, pos_label=pos_label)
    return brier_score
