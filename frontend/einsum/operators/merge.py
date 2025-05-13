import inspect
from typing import List, Optional, Union, Callable, Dict, Any
from enum import Enum, auto
import operator
from abc import ABC, abstractmethod


# --- Operators ---
# Using simple strings for now, could be classes or callables later


class MergeOperator:
    """Represents a merge operator (e.g., intersection, union)."""

    def __init__(self, name: str, symbol: str):
        self.name = name
        self.symbol = symbol  # e.g., '∩', '∪'

    def __repr__(self):
        return f"{self.name}({self.symbol})"


# Predefined merge operators 16
INTERSECT = MergeOperator("Intersection", "∩")
UNION = MergeOperator("Union", "∪")
# ... add others as needed
