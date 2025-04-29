from .operators import ComputeOperator, MergeOperator, CoordinateOperator
from .rankVariable import RankVariable
from typing import List


class Action:
    """Base class for EDGE actions."""

    pass


class MapAction(Action):
    """Represents a Map action (Ó)."""

    def __init__(
        self,
        target_ranks: List[RankVariable],
    ):
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


class ReduceAction(Action):
    """Represents a Reduce action (Ô)."""

    def __init__(self, target_ranks: List[RankVariable]):
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


class PopulateAction(Action):
    """Represents a specialized Populate action (≪)."""

    def __init__(
        self,
        target_ranks: List[RankVariable],
    ):
        # Ensure mutable_ranks correspond to RankVariables marked mutable in LHS
        self.target_ranks = target_ranks

    def __repr__(self):
        pass
