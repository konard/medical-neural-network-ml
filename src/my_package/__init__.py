"""Example module entry point.

Replace this with your actual implementation.
"""

from __future__ import annotations

import asyncio

__version__ = "0.1.0"

__all__ = ["add", "delay", "multiply"]


def add(a: int | float, b: int | float) -> int | float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def multiply(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


async def delay(seconds: float) -> None:
    """Async delay function.

    Args:
        seconds: Seconds to wait
    """
    await asyncio.sleep(seconds)
