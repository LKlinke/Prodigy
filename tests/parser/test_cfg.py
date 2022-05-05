from textwrap import dedent

import pytest
from prodigy.pgcl.cfg import program_one_big_loop
from prodigy.pgcl.compiler import parse_pgcl


def test_one_big_loop1():
    program = parse_pgcl("""
        bool b;
        nat x;
        x := 1
        while (b) {
            x := 2
            while (b) {
                { x := 3 } [0.5] { x:= 4 }
            }
            x := 5
        }
        x := 6
    """)
    program = program_one_big_loop(program, 'pc')

    assert 'pc' in program.variables
    expected = dedent("""
        bool b;
        nat x;
        nat pc;
        pc := 1;
        while (not (pc = 0)) {
            if (pc = 1) {
                x := 1;
                pc := 2;
            } else {
                if (pc = 2) {
                    if (b) {
                        x := 2;
                        pc := 3;
                    } else {
                        x := 6;
                        pc := 0;
                    }
                } else {
                    if (b) {
                        {
                            x := 3;
                        } [0.5] {
                            x := 4;
                        }
                        pc := 3;
                    } else {
                        x := 5;
                        pc := 2;
                    }
                }
            }
        }
    """).strip()
    assert str(program) == expected


@pytest.mark.xfail(reason="known bug", run=False)
def test_one_big_loop2():
    program = parse_pgcl("")
    # the call below crashes
    _program = program_one_big_loop(program, 'pc')
