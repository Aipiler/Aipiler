class AffineExpr:
    pass

class AffineConstantExpr(AffineExpr):
    def __init__(self, cst) -> None:
        assert isinstance(cst, int)
        self.cst = cst

class AffineDimExpr(AffineExpr):
    def __init__(self, rank: "Dim") -> None:
        super().__init__()
        self.rank = rank


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
    def __init__(self, affine_expr: AffineExpr = None):
        self.affine_expr = affine_expr
