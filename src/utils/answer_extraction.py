import re
from typing import Optional

# Robust extraction of numeric answer after #### marker
# Handles integers, decimals, negatives, and thousands separators
_FINAL_PATTERN = re.compile(r"####\s*([-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)")


def extract_final_answer(answer_text: str) -> Optional[float]:
    if not answer_text:
        return None
    m = _FINAL_PATTERN.search(answer_text)
    if not m:
        return None
    number_str = m.group(1).replace(",", "")
    try:
        return float(number_str)
    except ValueError:
        return None

