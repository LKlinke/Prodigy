from prodigy.pgcl.compiler import parse_pgcl
from prodigy.analysis.backward.wp import one_loop_wp_transformer
from prodigy.analysis.backward.simplify import normalize_expectation_transformer


def test_ber_ert():
    program = parse_pgcl("""
        nat x;
        nat n;
        nat r;
        while (x < n) {
            r := 1 : 1/2 + 0 : 1/2;
            x := x + r;
            tick(1);
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == 'λ𝐹. lfp 𝑋. [x < n] * ((1/2 * (((𝑋)[r/1, x/x + 1]) + (tick(1)))) + (1/2 * (((𝑋)[r/0, x/x + 0]) + (tick(1))))) + [not (x < n)] * 𝐹'
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [x < n] * 1/2 * ((𝑋)[r/1, x/x + 1] + tick(1)) + [x < n] * 1/2 * ((𝑋)[r/0, x/x + 0] + tick(1))'


def test_linear01():
    program = parse_pgcl("""
        nat x;
        while (2 <= x) {
            { x := x - 1; } [1/3] {
                x := x - 2;
            }
            tick(1);
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == 'λ𝐹. lfp 𝑋. [2 <= x] * (((((𝑋)[x/x - 1]) + (tick(1))) * 1/3) + ((((𝑋)[x/x - 2]) + (tick(1))) * (1.0 - 1/3))) + [not (2 <= x)] * 𝐹'
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [2 <= x] * 1/3 * ((𝑋)[x/x - 1] + tick(1)) + [2 <= x] * (1.0 - 1/3) * ((𝑋)[x/x - 2] + tick(1))'


def test_prspeed():
    program = parse_pgcl("""
        nat x;
        nat y;
        nat m;
        nat n;
        while ((x + 3 <= n)) {
            if (y < m) {
                { y := y + 1; } [1/2] {
                    y := y + 0;
                }
            } else {
                { x := x + 0; } [1/4] {
                    { x := x + 1; } [1/3] {
                        { x := x + 2; } [1/2] {
                            x := x + 3;
                        }
                    }
                }
            }
            tick(1);
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == 'λ𝐹. lfp 𝑋. [(x + 3) <= n] * (([y < m] * (((((𝑋)[y/y + 1]) + (tick(1))) * 1/2) + ((((𝑋)[y/y + 0]) + (tick(1))) * (1.0 - 1/2)))) + ([not (y < m)] * (((((𝑋)[x/x + 0]) + (tick(1))) * 1/4) + ((((((𝑋)[x/x + 1]) + (tick(1))) * 1/3) + ((((((𝑋)[x/x + 2]) + (tick(1))) * 1/2) + ((((𝑋)[x/x + 3]) + (tick(1))) * (1.0 - 1/2))) * (1.0 - 1/3))) * (1.0 - 1/4))))) + [not ((x + 3) <= n)] * 𝐹'
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [((x + 3) <= n) & (y < m)] * 1/2 * ((𝑋)[y/y + 1] + tick(1)) + [((x + 3) <= n) & (y < m)] * (1.0 - 1/2) * ((𝑋)[y/y + 0] + tick(1)) + [((x + 3) <= n) & not (y < m)] * 1/4 * ((𝑋)[x/x + 0] + tick(1)) + [((x + 3) <= n) & not (y < m)] * (1/3 * (1.0 - 1/4)) * ((𝑋)[x/x + 1] + tick(1)) + [((x + 3) <= n) & not (y < m)] * ((1/2 * (1.0 - 1/3)) * (1.0 - 1/4)) * ((𝑋)[x/x + 2] + tick(1)) + [((x + 3) <= n) & not (y < m)] * (((1.0 - 1/2) * (1.0 - 1/3)) * (1.0 - 1/4)) * ((𝑋)[x/x + 3] + tick(1))'
