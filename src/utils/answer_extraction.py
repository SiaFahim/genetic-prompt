"""
Answer extraction utilities for GSM8K dataset.
Handles parsing of GSM8K answer format with #### markers.
"""

import re
from typing import Optional, Union


def extract_final_answer(answer_text: str) -> Optional[float]:
    """
    Extract numeric answer after #### marker from GSM8K answer format.
    
    The GSM8K dataset uses the format:
    "Step-by-step solution...
    #### 42"
    
    Args:
        answer_text: The answer text containing the solution and final answer
        
    Returns:
        float: The extracted numeric answer, or None if no answer found
        
    Examples:
        >>> extract_final_answer("Solution steps... #### 72")
        72.0
        >>> extract_final_answer("Steps... #### 3.5")
        3.5
        >>> extract_final_answer("Steps... #### -10")
        -10.0
        >>> extract_final_answer("Steps... #### 1,234")
        1234.0
        >>> extract_final_answer("No answer here")
        None
    """
    if not answer_text:
        return None
    
    # Pattern to match #### followed by a number (with optional comma separators)
    pattern = r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, answer_text)
    
    if match:
        number_str = match.group(1).replace(',', '')
        try:
            return float(number_str)
        except ValueError:
            return None
    
    return None


def extract_answer_from_response(response_text: str) -> Optional[float]:
    """
    Extract numeric answer from LLM response text.
    
    This function tries multiple strategies to extract a numeric answer:
    1. Look for #### marker (GSM8K format)
    2. Look for "Answer:" followed by a number
    3. Look for the last number in the text
    
    Args:
        response_text: The LLM response text
        
    Returns:
        float: The extracted numeric answer, or None if no answer found
    """
    if not response_text:
        return None
    
    # Strategy 1: Look for #### marker
    answer = extract_final_answer(response_text)
    if answer is not None:
        return answer
    
    # Strategy 2: Look for "Answer:" followed by a number
    answer_pattern = r'(?:Answer|answer):\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(answer_pattern, response_text, re.IGNORECASE)
    if match:
        number_str = match.group(1).replace(',', '')
        try:
            return float(number_str)
        except ValueError:
            pass
    
    # Strategy 3: Look for the last number in the text
    number_pattern = r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    matches = re.findall(number_pattern, response_text)
    if matches:
        # Take the last number found
        number_str = matches[-1].replace(',', '')
        try:
            return float(number_str)
        except ValueError:
            pass
    
    return None


def compare_answers(predicted: Optional[float], ground_truth: Optional[float], 
                   tolerance: float = 0.001) -> bool:
    """
    Compare predicted answer with ground truth allowing for small numerical tolerance.
    
    Args:
        predicted: The predicted numeric answer
        ground_truth: The ground truth numeric answer
        tolerance: Tolerance for floating point comparison
        
    Returns:
        bool: True if answers match within tolerance, False otherwise
    """
    if predicted is None or ground_truth is None:
        return predicted == ground_truth
    
    return abs(predicted - ground_truth) < tolerance


def validate_answer_extraction():
    """Test the answer extraction functions with sample data."""
    test_cases = [
        ("Solution steps... #### 72", 72.0),
        ("Steps... #### 3.5", 3.5),
        ("Steps... #### -10", -10.0),
        ("Steps... #### 1,234", 1234.0),
        ("No answer here", None),
        ("Answer: 42", 42.0),
        ("The answer is 123.45", 123.45),
        ("Multiple numbers 10, 20, final answer is 30", 30.0),
    ]
    
    print("Testing answer extraction...")
    all_passed = True
    
    for text, expected in test_cases:
        result = extract_answer_from_response(text)
        passed = (result == expected)
        status = "✅" if passed else "❌"
        print(f"{status} '{text}' -> {result} (expected {expected})")
        if not passed:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    # Run validation tests
    success = validate_answer_extraction()
    if success:
        print("\n✅ All answer extraction tests passed!")
    else:
        print("\n❌ Some answer extraction tests failed!")
