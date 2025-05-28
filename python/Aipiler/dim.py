from typing import List, Union, Sequence, Optional
import dataclasses


class AffineExpr:
    pass


class AffineConstantExpr(AffineExpr):
    def __init__(self, cst) -> None:
        assert isinstance(cst, int)
        self.cst = cst


class AffineDimExpr(AffineExpr):
    def __init__(self, dim: "Dim") -> None:
        super().__init__()
        self.dim = dim


class AffineAddExpr(AffineExpr):
    def __init__(self, lhs: AffineExpr, rhs: AffineExpr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs


class AffineMulExpr(AffineExpr):
    def __init__(self, lhs: AffineExpr, rhs: AffineExpr) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs


@dataclasses.dataclass
class SymIntArgument:
    name: str


@dataclasses.dataclass
class SymFloatArgument:
    name: str


@dataclasses.dataclass
class SymBoolArgument:
    name: str


# TODO: 抄pytorch的实现
# 过去的方法只为动态shape设定symbolic值，现在为每一个dim都设置symbolic值。
# 维护一个shapeEnv，全局管理dim之间symbolic值的关系，并且维护symbolic和size的关系。
class Dim:
    def __init__(
        self, affine_expr: Union[AffineExpr, Sequence[AffineExpr]] = [], size: int = 0
    ):
        self.affine_exprs = (
            [affine_expr] if isinstance(affine_expr, AffineExpr) else list(affine_expr)
        )

        self.size = size

    def set_size(self, size: int):
        if size < 0:
            raise ValueError("Size must be non-negative")
        self.size = size

    def get_size(self) -> int:
        return self.size
