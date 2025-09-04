"""
Answer extraction utilities for GSM8K problems.
Handles parsing of various numeric formats from model outputs.
"""

import re
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_final_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from GSM8K format text.
    
    This function handles the GSM8K format where final answers appear after
    four hash symbols (####). It can extract:
    - Integers: #### 42
    - Decimals: #### 3.14
    - Negative numbers: #### -17
    - Numbers with commas: #### 1,234
    
    Args:
        text: Text containing the answer (either ground truth or model output)
        
    Returns:
        Float value of the extracted answer, or None if no answer found
    """
    if not text:
        return None
    
    # Primary pattern: Look for #### followed by a number
    pattern = r'####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, text)
    
    if match:
        number_str = match.group(1).replace(',', '')
        try:
            return float(number_str)
        except ValueError:
            logger.warning(f"Could not convert extracted answer to float: {number_str}")
            return None
    
    return None


def extract_answer_from_response(response: str) -> Optional[float]:
    """
    Extract numeric answer from model response text.
    
    This function tries multiple strategies to extract answers from model outputs:
    1. Look for #### format (GSM8K standard)
    2. Look for "The answer is X" patterns
    3. Look for "Answer: X" patterns
    4. Look for numbers at the end of the response
    5. Look for numbers in parentheses or brackets
    
    Args:
        response: Model's response text
        
    Returns:
        Float value of the extracted answer, or None if no answer found
    """
    if not response:
        return None
    
    # Strategy 1: GSM8K format with ####
    answer = extract_final_answer(response)
    if answer is not None:
        return answer
    
    # Strategy 2: "The answer is X" patterns
    patterns = [
        r'[Tt]he answer is\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Aa]nswer:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Aa]nswer\s*=\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ff]inal answer:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Ss]o the answer is\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'[Tt]herefore,?\s*(?:the answer is\s*)?([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                continue
    
    # Strategy 3: Numbers in parentheses or brackets at the end
    end_patterns = [
        r'\((\d+(?:,\d{3})*(?:\.\d+)?)\)\s*$',
        r'\[(\d+(?:,\d{3})*(?:\.\d+)?)\]\s*$',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*$'
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, response.strip())
        if match:
            number_str = match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                continue
    
    # Strategy 4: Last number in the text (fallback)
    numbers = re.findall(r'([-+]?\d+(?:,\d{3})*(?:\.\d+)?)', response)
    if numbers:
        # Try the last number found
        last_number = numbers[-1].replace(',', '')
        try:
            return float(last_number)
        except ValueError:
            pass
    
    logger.debug(f"Could not extract answer from response: {response[:100]}...")
    return None


def compare_answers(predicted: Optional[float], ground_truth: Optional[float], 
                   tolerance: float = 0.001) -> bool:
    """
    Compare predicted answer with ground truth allowing for small numerical differences.
    
    Args:
        predicted: Predicted answer from model
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if answers match within tolerance, False otherwise
    """
    if predicted is None or ground_truth is None:
        return False
    
    return abs(predicted - ground_truth) < tolerance


def validate_answer_format(answer: Union[str, float, int]) -> Optional[float]:
    """
    Validate and normalize answer format.
    
    Args:
        answer: Answer in various formats
        
    Returns:
        Normalized float answer or None if invalid
    """
    if answer is None:
        return None
    
    if isinstance(answer, (int, float)):
        return float(answer)
    
    if isinstance(answer, str):
        # Remove common formatting
        cleaned = answer.strip().replace(',', '').replace('$', '')
        
        # Try to convert to float
        try:
            return float(cleaned)
        except ValueError:
            # Try to extract number from string
            return extract_answer_from_response(answer)
    
    return None


def calculate_accuracy(predictions: list, ground_truths: list, 
                      tolerance: float = 0.001) -> float:
    """
    Calculate accuracy between predictions and ground truths.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    if not predictions:
        return 0.0
    
    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if compare_answers(pred, truth, tolerance):
            correct += 1
    
    return correct / len(predictions)


def extract_and_validate_answer(response: str, ground_truth: Optional[float] = None,
                               tolerance: float = 0.001) -> dict:
    """
    Extract answer from response and optionally validate against ground truth.
    
    Args:
        response: Model response text
        ground_truth: Optional ground truth for validation
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with extraction results and validation info
    """
    extracted = extract_answer_from_response(response)
    
    result = {
        'extracted_answer': extracted,
        'extraction_successful': extracted is not None,
        'response_length': len(response) if response else 0
    }
    
    if ground_truth is not None:
        result['ground_truth'] = ground_truth
        result['correct'] = compare_answers(extracted, ground_truth, tolerance)
        if extracted is not None and ground_truth is not None:
            result['absolute_error'] = abs(extracted - ground_truth)
    
    return result
