import numpy as np
import pytest

from utils.metrics import calculate_ece, calculate_macro_ce, compute_em
from utils.plotting import plot_confidence_error
from utils.utils import extract_answer, extract_texts_and_confidences, normalize_text


def test_normalize_text():
    assert normalize_text("The quick brown fox.") == "quick brown fox"
    assert normalize_text("A test, with punctuation!") == "test with punctuation"
    assert normalize_text("An example.") == "example"


def test_compute_em():
    assert compute_em("The quick brown fox", "The quick brown fox") == 1
    assert compute_em("The quick brown fox", "The brown fox") == 0
    assert compute_em("The Apple.", "the apple") == 1  # normalization


def test_extract_texts_and_confidences():
    response = "Final Answer: The quick brown fox Probability: 0.95"
    texts, confidences = extract_texts_and_confidences(response)
    assert texts == ["The quick brown fox"]
    assert confidences == [0.95]

    response = (
        "Answer: No. Confidence: 0.9 Answer: Yes. Confidence: 0.1 "
        "Answer: Different nationalities. Confidence: 0.8"
    )
    texts, confidences = extract_texts_and_confidences(response)
    assert len(texts) == 3
    assert confidences == [0.9, 0.1, 0.8]


def test_extract_answer():
    response = "Final Answer: The quick brown fox Confidence: 0.95"
    assert extract_answer(response) == "The quick brown fox"
    assert extract_answer("No answer provided.") is None


def test_calculate_ece():
    scores = np.array([1, 0, 1, 1])
    confidences = np.array([0.9, 0.1, 0.8, 0.7])
    ece = calculate_ece(scores, confidences, M=2)
    assert isinstance(ece, float)


def test_calculate_macro_ce():
    # perfect calibration -> 0 error
    acc = np.array([0.0, 1.0])
    conf = np.array([0.0, 1.0])
    assert calculate_macro_ce(acc, conf) == pytest.approx(0.0)


def test_plot_confidence_error():
    scores = np.array([1, 0, 1, 1])
    confidences = np.array([0.9, 0.1, 0.8, 0.7])
    fig, ax = plot_confidence_error(scores, confidences, M=2)
    assert fig is not None
    assert ax is not None
