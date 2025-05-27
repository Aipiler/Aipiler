import Aipiler
from Aipiler.ops import add, matmul
from Aipiler import Tensor
from Aipiler.dim import Dim


def test_add():
    a = Tensor([Dim(), Dim()], dtype=Aipiler.i32)
    b = Tensor([Dim(), Dim()], dtype=Aipiler.i32)

    # ab, ab -> ab
    c = add(a, b)
    assert len(c.symbolic_shape) == 2

    exprs1 = [expr.dim for expr in c.symbolic_shape[0].affine_exprs]
    exprs2 = [expr.dim for expr in c.symbolic_shape[1].affine_exprs]

    assert a.symbolic_shape[0] in exprs1
    assert a.symbolic_shape[1] in exprs2
    assert b.symbolic_shape[0] in exprs1
    assert b.symbolic_shape[1] in exprs2


def test_mm():
    a = Tensor([Dim(), Dim()], dtype=Aipiler.i32)
    b = Tensor([Dim(), Dim()], dtype=Aipiler.i32)

    # ik, kj -> ij
    c = matmul(a, b)
    assert len(c.symbolic_shape) == 2

    exprs1 = [expr.dim for expr in c.symbolic_shape[0].affine_exprs]
    exprs2 = [expr.dim for expr in c.symbolic_shape[1].affine_exprs]

    assert a.symbolic_shape[0] in exprs1
    assert b.symbolic_shape[1] in exprs2
