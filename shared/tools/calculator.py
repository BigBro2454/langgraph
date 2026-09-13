"""
Standard calculator tools for arithmetic reasoning in LangGraph agents.
"""

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Adds two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtracts the second number from the first.

    Args:
        a: The starting number.
        b: The number to subtract.

    Returns:
        The difference between a and b.
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of a and b.
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divides the first number by the second.

    Args:
        a: The numerator.
        b: The denominator (must not be zero).

    Returns:
        The quotient of a divided by b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
