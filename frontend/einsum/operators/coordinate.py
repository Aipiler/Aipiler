import inspect
from typing import List, Optional, Union, Callable, Dict, Any
from enum import Enum, auto
import operator
from abc import ABC, abstractmethod


# --- Operators ---
# Using simple strings for now, could be classes or callables later


class CoordinateOperator:
    """Represents a coordinate operator used in Populate actions."""

    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        func: Optional[Callable] = None,
    ):
        self.name = name
        self.params = params or {}  # e.g., {'K': 3} for TopK
        self.func = func  # Optional: Store the actual function

    def __repr__(self):
        param_str = (
            f"({','.join(f'{k}={v}' for k, v in self.params.items())})"
            if self.params
            else ""
        )
        return f"{self.name}{param_str}"


# Example coordinate operators
TOP_K = lambda k: CoordinateOperator("TopK", params={"K": k})
SELECT_ALL_DIMS = CoordinateOperator("SelectAllDims")
