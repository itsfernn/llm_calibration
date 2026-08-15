import re
import string


# see https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
def normalize_text(s: str):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_texts_and_confidences(response: str) -> tuple[list[str], list[float]]:
    """
    Extracts the final answer and confidence/probability from the LLM response.
    Expected format at the end:
    "Final Answer: <answer> Probability: <confidence>"
    """
    response = response.strip().replace("\n", " ")

    # Regex to catch variations in punctuation/spacing
    pattern = r"(?:Final Answer|Answer|Guess|Step \d*)[:\-\s]*(.*?)\s*(?:Probability|Confidence)[:\-\s]*([01](?:\.\d+)?)"

    matches = re.findall(pattern, response, flags=re.MULTILINE)
    texts = [text for text, _ in matches]
    confidences = [float(confidence) for _, confidence in matches]

    return texts, confidences


def extract_answer(response: str) -> str | None:
    pattern = r"(?:Final Answer|Answer|Guess)[:\s]*([^\n]+)(?:Confidence|Probability)"
    match = re.search(pattern, response)
    return match.group(1).strip() if match else None
