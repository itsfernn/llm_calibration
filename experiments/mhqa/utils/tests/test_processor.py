import pytest
from utils.processor import (
    DirectProcessor,
    TopKProcessor,
    CotProcessor,
    MultistepProcessor,
    MultistepFewshotProcessor,
    get_processor,
)


class FakeLLM:
    """Minimal stand-in for an LLM client: returns a canned response."""

    def __init__(self, response):
        self._response = response

    def invoke(self, prompt):
        return type("R", (), {"content": self._response})()


@pytest.fixture
def sample_data():
    return {"question": "What is the capital of France?", "answer": "Paris"}


RESPONSE = "Answer: Paris Confidence: 0.9"


def test_get_processor_factory():
    assert isinstance(get_processor("direct"), DirectProcessor)
    assert isinstance(get_processor("top-k"), TopKProcessor)
    assert isinstance(get_processor("top-k-norm"), TopKProcessor)
    assert isinstance(get_processor("cot"), CotProcessor)
    assert isinstance(get_processor("multistep"), MultistepProcessor)
    assert isinstance(get_processor("multistep-fewshot"), MultistepFewshotProcessor)
    with pytest.raises(ValueError):
        get_processor("unknown")


def test_direct_processor(sample_data):
    processor = DirectProcessor()
    result = processor.process_sample(sample_data.copy(), FakeLLM(RESPONSE))
    result = processor.eval_sample(result)
    assert result["prediction"] == "Paris"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["em"] == 1


def test_top_k_processor(sample_data):
    processor = TopKProcessor(k=3)
    resp = (
        "Answer: Paris Confidence: 0.9 "
        "Answer: London Confidence: 0.7 "
        "Answer: Berlin Confidence: 0.2"
    )
    result = processor.process_sample(sample_data.copy(), FakeLLM(resp))
    result = processor.eval_sample(result)
    assert result["prediction"] == "Paris"  # highest confidence first
    assert result["confidence"] == 0.9


def test_cot_processor(sample_data):
    processor = CotProcessor()
    result = processor.process_sample(sample_data.copy(), FakeLLM(RESPONSE))
    result = processor.eval_sample(result)
    assert result["prediction"] == "Paris"
    assert result["confidence"] == 0.9


def test_multistep_processor(sample_data):
    processor = MultistepProcessor()
    resp = (
        "Step 1: France has a capital. Confidence: 0.8 "
        "Step 2: The capital of France is Paris. Confidence: 0.9 "
        "Final Answer: Paris Confidence: 1.0"
    )
    result = processor.process_sample(sample_data.copy(), FakeLLM(resp))
    result = processor.eval_sample(result)
    assert result["prediction"] == "Paris"
    assert result["confidence"] == pytest.approx(0.8 * 0.9 * 1.0)
    assert result["num_steps"] == 2
    assert result["final_confidence"] == 1.0
