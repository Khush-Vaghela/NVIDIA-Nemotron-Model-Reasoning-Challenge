"""Official grader functions — copied VERBATIM from the winning submission's
reasoning.py (attached corpus), whose docstrings state they match the official
metric notebook's metric_reference.extract_final_answer / verify.

PROVENANCE FLAG (do not silently trust):
- The official Kaggle metric notebook (kaggle.com/code/metric/nvidia-nemotron-metric)
  is not retrievable offline, so this is the closest verbatim source available.
- These functions implement exactly the officially documented quirks:
  boxed-priority extraction (last non-empty \\boxed{}), binary strings strict-exact,
  numeric relative tolerance 1e-2 (abs 1e-5 near zero), case-insensitive otherwise.
- A third-party reimplementation found online (yunior123/nvidia-nemotron-reasoning)
  OMITS the binary-strict and case-insensitive rules and was therefore rejected.

The only changes here: function names aliased to the official names
(extract_final_answer / verify); bodies are byte-identical to reasoning.py.
"""

from __future__ import annotations

import math
import re


def extract_answer(reasoning_text: str) -> str:
    """Extract the answer from \\boxed{...}, matching metric_reference.extract_final_answer."""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", reasoning_text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


# Official metric notebook names
extract_final_answer = extract_answer
verify = compare_answer
