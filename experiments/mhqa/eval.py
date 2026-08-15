from utils.metrics import (
    calculate_ece,
    calculate_macro_ce,
    calculate_rp_auc,
    calculate_roc_auc,
)
from sklearn.metrics import brier_score_loss, f1_score, recall_score, precision_score

def evaluate(accuracies, confidences, predictions=None):
    accuracy = accuracies.mean()
    ece = calculate_ece(accuracies, confidences)
    macro_ce = calculate_macro_ce(accuracies, confidences)
    rp_auc = calculate_rp_auc(accuracies, (1-confidences), pos_label=0)
    roc_auc = calculate_roc_auc(accuracies, confidences)
    brier_score = brier_score_loss(accuracies, confidences)

    results = {
        "performance/acc": accuracy,
        "calibration/ece": ece,
        "calibration/macro_ce": macro_ce,
        "calibration/roc_auc":roc_auc,
        "calibration/rp_auc": rp_auc,
        "calibration/brier_score": brier_score,
    }

    if predictions is not None:
        results["classification/f1"] = f1_score(accuracies, predictions)
        results["classification/recall"] = recall_score(accuracies, predictions)
        results["classification/precision"] = precision_score(accuracies, predictions)

    return results
