"""Dataset loading tests.

Only 2WikiMultihopQA is tested by default because it loads from the local
data/raw/2WikiMultihopQA copy (no network). HotpotQA and MuSiQue load from
the Hugging Face hub and need a network connection.
"""

import pytest

from utils.datasets import get_dataset


@pytest.mark.parametrize("dataset", ["2WikiMultihopQA"])
def test_get_dataset(dataset):
    ds = get_dataset(dataset, num_samples=5)
    assert set(ds.column_names) == {"id", "question", "answer", "type"}
    assert len(ds) == 5
