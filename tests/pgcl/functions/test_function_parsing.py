from probably.pgcl import parse_pgcl


def test_basic_function():
    prog = parse_pgcl("""
        fun f := {
            nat x;
            x := 10;
            return x;
        }
        nat x;
        x := f();
    """)
