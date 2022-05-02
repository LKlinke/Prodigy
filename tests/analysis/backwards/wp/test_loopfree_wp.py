from probably.pgcl.ast import VarExpr
from probably.pgcl.compiler import compile_pgcl
from probably.pgcl.substitute import substitute_expr
from probably.analysis.backward.wp import loopfree_wp
from probably.util.ref import Mut


def test_uniform_sequence():
    program = compile_pgcl("""
        nat jump; nat t;
        jump := unif(0,1); t := t + 1
    """)
    wp_expr = loopfree_wp(program.instructions, VarExpr("X"))
    wp_expr_ref = Mut.alloc(wp_expr)
    substitute_expr(wp_expr_ref, symbolic=set(["X"]))
    assert str(wp_expr_ref.val) == '(1/2 * ((X)[jump/0, t/t + 1])) + (1/2 * ((X)[jump/1, t/t + 1]))'
