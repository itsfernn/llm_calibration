import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .utils import compute_bin_stats

_EPS = 1e-12

def histogram_scaling(
    probs_train,
    probs_test,
    y_train,
    bins=None,
):
    """
    Shift/replace test confidences with per-bin training means (histogram binning).

    Parameters
    ----------
    probs_train, probs_test : array-like
        Confidence scores (floats, typically in [0, 1]).
    y_train : array-like
        Target values for training rows (e.g., 0/1 or continuous).
    bins : array-like, optional
        Bin edges. Default: np.linspace(0.0, 1.0, 11) (10 equal-width bins).
    full_output : bool
        If True, returns full test table and train stats

    Returns
    -------
    test_table : pandas.DataFrame
        test rows with columns: conf, bin, adj_conf (bin mean from train), stderr (train stderr), and y (if y_test provided).
    train_stats : pandas.DataFrame
        Per-bin stats from training set: bin_label, mid, mean_y, n, std, stderr.
    """
    # defaults
    if bins is None:
        bins = np.linspace(0.0, 1.0, 11)
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("bins must be a 1D array-like of at least two edges")

    labels = [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(len(bins)-1)]
    stats = compute_bin_stats(y_train, probs_train, bins)

    # fill missing bin means according to the chosen strategy
    known = stats.dropna(subset=["mean_y"])
    if len(known) >= 2:  # need at least two points to interpolate
        interp_vals = np.interp(stats["mid"], known["mid"], known["mean_y"])
        stats["mean_y"] = stats["mean_y"].fillna(pd.Series(interp_vals))
    else:
        # not enough points to interpolate -> fallback to global mean
        stats["mean_y"] = stats["mean_y"].fillna(0)


    # prepare test dataframe and merge
    test_df = pd.DataFrame({"conf": np.asarray(probs_test)})

    test_df["bin"] = pd.cut(
        test_df["conf"], bins=bins, labels=labels, include_lowest=True
    )

    # map the average y in the train set to each bin in the test set
    merged = test_df.merge(
        stats[["bin", "mean_y"]].rename(columns={"mean_y": "adj_conf"}),
        left_on="bin",
        right_on="bin",
        how="left",
    )

    return merged["adj_conf"]


from sklearn.isotonic import IsotonicRegression

def isotonic_scaling(
    probs_train,
    probs_test,
    y_train,
    eps=1e-6,
    clip=True,
    return_model=False
):
    """
    Isotonic Regression scaling: fit a non-decreasing function mapping
    predicted probabilities to calibrated probabilities.

    Parameters
    ----------
    probs_train, probs_test : array-like, shape (n,)
        Predicted probabilities in [0,1].
    y_train : array-like, shape (n,)
        Binary labels {0,1}.
    eps : float
        Small value to avoid extreme values.
    clip : bool
        Whether to clip probabilities to [eps, 1 - eps].
    return_model : bool
        If True, return fitted IsotonicRegression model.

    Returns
    -------
    probs_cal : ndarray
        Calibrated probabilities for probs_test.
    model (optional) : fitted IsotonicRegression
    """
    probs_train = np.asarray(probs_train, dtype=float)
    probs_test = np.asarray(probs_test, dtype=float)

    if clip:
        probs_train = np.clip(probs_train, eps, 1 - eps)
        probs_test = np.clip(probs_test, eps, 1 - eps)

    y_train = np.asarray(y_train, dtype=float)

    # Fit isotonic regression model
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probs_train, y_train)

    # Apply to test set
    probs_cal = model.predict(probs_test)

    return (probs_cal, model) if return_model else probs_cal



def platt_scaling(
    probs_train, probs_test, y_train, eps=1e-6, clip=True, return_model=False
):
    """
    Platt (sigmoid) scaling: fit a logistic regression on train probs (converted to logits)
    and apply it to test probs.

    Parameters
    ----------
    probs_train, probs_test : array-like, shape (n,)
        Predicted probabilities in [0,1].
    y_train : array-like, shape (n,)
        Binary labels {0,1} for training data.
    eps : float
        Clipping epsilon for probabilities before logit.
    clip : bool
        Whether to clip probs to [eps, 1 - eps] (recommended).
    return_model : bool
        If True, return the fitted sklearn LogisticRegression object as second value.

    Returns
    -------
    probs_cal : ndarray, shape (len(probs_test),)
        Calibrated probabilities for probs_test.
    model (optional) : fitted sklearn LogisticRegression
    """
    probs_train = np.asarray(probs_train, dtype=float)
    probs_test = np.asarray(probs_test, dtype=float)

    if clip:
        probs_train = np.clip(probs_train, eps, 1 - eps)
        probs_test = np.clip(probs_test, eps, 1 - eps)

    # Convert to logit (real-valued score)
    logits_train = np.log(probs_train / (1 - probs_train)).reshape(-1, 1)
    logits_test = np.log(probs_test / (1 - probs_test)).reshape(-1, 1)

    # Fit a simple logistic regression (maps logit -> calibrated prob)
    # Use very large C to approximate no regularization (platt scaling is just a logistic fit).
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits_train, y_train)

    probs_cal = model.predict_proba(logits_test)[:, 1]

    return (probs_cal, model) if return_model else probs_cal


def temperature_scaling(probs_train,  probs_test, y_train, eps=1e-6, return_T=False):
    """
    Temperature scaling: find scalar T > 0 that minimizes NLL on the training set,
    then apply scaled logits to the test set.

    Parameters
    ----------
    probs_train, probs_test : array-like, shape (n,)
        Predicted probabilities in [0,1].
    y_train : array-like, shape (n,)
        Binary labels {0,1} for training data.
    eps : float
        Clipping epsilon for probabilities before logit.
    return_T : bool
        If True, also return the optimized temperature T.

    Returns
    -------
    probs_cal : ndarray, shape (len(probs_test),)
        Calibrated probabilities for probs_test.
    T_opt (optional) : float
        Learned temperature (>0)
    """
    probs_train = np.clip(np.asarray(probs_train, dtype=float), eps, 1 - eps)
    probs_test = np.clip(np.asarray(probs_test, dtype=float), eps, 1 - eps)
    y_train = np.asarray(y_train, dtype=int)

    # convert probs -> logits
    logits_train = np.log(probs_train / (1 - probs_train))
    logits_test = np.log(probs_test / (1 - probs_test))

    def nll_given_logT(logT):
        T = np.exp(logT)  # ensure positivity
        scaled = logits_train / T
        probs = 1.0 / (1.0 + np.exp(-scaled))
        return log_loss(y_train, probs, labels=[0, 1])

    # optimize over logT (unconstrained)
    res = minimize(nll_given_logT, x0=0.0, method="L-BFGS-B")
    if not res.success:
        # fallback to T=1 if optimization fails
        T_opt = 1.0
    else:
        T_opt = float(np.exp(res.x[0]))

    # apply to test set
    scaled_test = logits_test / T_opt
    probs_cal = 1.0 / (1.0 + np.exp(-scaled_test))

    if return_T:
        return (probs_cal, T_opt)
    else:
        return probs_cal

