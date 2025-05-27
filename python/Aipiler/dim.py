from typing import List, Union, Sequence, Optional


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


class Dim:
    def __init__(self, affine_expr: Union[AffineExpr, Sequence[AffineExpr]] = []):
        self.affine_exprs = (
            [affine_expr] if isinstance(affine_expr, AffineExpr) else list(affine_expr)
        )
