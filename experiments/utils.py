import json
import re
import torch

def print_metadata(metadata):
    print(f"Running evaluation with model: {metadata['model']}")
    print(f"Batch size: {metadata['batch_size']}")
    print(f"System prompt: {metadata['system_prompt']}")
    print(f"Device: {metadata['device']}")
    if metadata['git_commit']:
        print(f"Git commit: {metadata['git_commit']}")


def find_subsequence(sequence, subsequence):
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            return i
    return None



def parse_response(text: str):
    pattern = r"Final Answer:\s*(?P<answer>.*?)\s*Confidence:\s*(?P<confidence>0(?:\.\d+)?|1(?:\.0+)?)"
    
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None, None
    
    answer = match.group("answer").strip()
    confidence = float(match.group("confidence"))

    return answer, confidence

def parse_output(text):
    """
    Parses model output for:
      - answer (string or number)
      - confidence (float in [0,1])

    Returns:
        dict with keys: {"answer": str or None, "confidence": float or None}
    """

    result = {"answer": None, "confidence": None}

    if not text or not isinstance(text, str):
        return result

    # ----------------------------
    # 1. Try to extract JSON block
    # ----------------------------
    json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)

    for candidate in reversed(json_candidates):  # prefer last JSON block
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                if "answer" in parsed:
                    result["answer"] = str(parsed["answer"]).strip()
                if "confidence" in parsed:
                    result["confidence"] = float(parsed["confidence"])
                if result["answer"] or result["confidence"] is not None:
                    return result
        except:
            pass

    # -----------------------------------
    # 2. Try to repair slightly broken JSON
    # -----------------------------------
    for candidate in reversed(json_candidates):
        fixed = candidate

        # Replace single quotes with double quotes
        fixed = fixed.replace("'", '"')

        # Remove trailing commas
        fixed = re.sub(r",\s*}", "}", fixed)

        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                if "answer" in parsed:
                    result["answer"] = str(parsed["answer"]).strip()
                if "confidence" in parsed:
                    result["confidence"] = float(parsed["confidence"])
                if result["answer"] or result["confidence"] is not None:
                    return result
        except:
            pass

    # ----------------------------
    # 3. Regex fallback extraction
    # ----------------------------

    # answer patterns
    answer_patterns = [
        r'"answer"\s*:\s*"?([^",}]+)"?',
        r'answer\s*[:=]\s*"?([^",}\n]+)"?',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["answer"] = match.group(1).strip()
            break

    # confidence patterns
    confidence_patterns = [
        r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
        r'confidence\s*[:=]\s*([0-9]*\.?[0-9]+)',
    ]

    for pattern in confidence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result["confidence"] = float(match.group(1))
            except:
                pass
            break

    # ----------------------------
    # 5. Normalize confidence
    # ----------------------------
    if result["confidence"] is not None:
        # clamp to [0,1]
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))

    return result
