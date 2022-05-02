"""
Here we do a bunch of tests on a larger collection of linear programs, with focus on weakest pre-expectation computation.
"""

from probably.pgcl.compiler import parse_pgcl
from probably.analysis.backward.wp import one_loop_wp_transformer
from probably.analysis.backward.simplify import normalize_expectation_transformer


def test_branchy():
    code = """
        while (c < 6) {
            {c := 3} [0.8] {c := 7}
            if (c=2) { d:=10 } {d := 20}
            if (f=1) { f:=0 } {f := 1}
        }
    """
    program = parse_pgcl(code)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == 'λ𝐹. lfp 𝑋. [c < 6] * (((([3 = 2] * (([f = 1] * ((𝑋)[c/3, d/10, f/0])) + ([not (f = 1)] * ((𝑋)[c/3, d/10, f/1])))) + ([not (3 = 2)] * (([f = 1] * ((𝑋)[c/3, d/20, f/0])) + ([not (f = 1)] * ((𝑋)[c/3, d/20, f/1]))))) * 0.8) + ((([7 = 2] * (([f = 1] * ((𝑋)[c/7, d/10, f/0])) + ([not (f = 1)] * ((𝑋)[c/7, d/10, f/1])))) + ([not (7 = 2)] * (([f = 1] * ((𝑋)[c/7, d/20, f/0])) + ([not (f = 1)] * ((𝑋)[c/7, d/20, f/1]))))) * (1.0 - 0.8))) + [not (c < 6)] * 𝐹'
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(c < 6) & ((3 = 2) & (f = 1))] * 0.8 * (𝑋)[c/3, d/10, f/0] + [(c < 6) & ((3 = 2) & not (f = 1))] * 0.8 * (𝑋)[c/3, d/10, f/1] + [(c < 6) & (not (3 = 2) & (f = 1))] * 0.8 * (𝑋)[c/3, d/20, f/0] + [(c < 6) & (not (3 = 2) & not (f = 1))] * 0.8 * (𝑋)[c/3, d/20, f/1] + [(c < 6) & ((7 = 2) & (f = 1))] * (1.0 - 0.8) * (𝑋)[c/7, d/10, f/0] + [(c < 6) & ((7 = 2) & not (f = 1))] * (1.0 - 0.8) * (𝑋)[c/7, d/10, f/1] + [(c < 6) & (not (7 = 2) & (f = 1))] * (1.0 - 0.8) * (𝑋)[c/7, d/20, f/0] + [(c < 6) & (not (7 = 2) & not (f = 1))] * (1.0 - 0.8) * (𝑋)[c/7, d/20, f/1]'


def test_geometric():
    program = parse_pgcl("""
        nat c;
        nat f;

        while(f=1){
            {f := 0}[0.5]{ c := c +1 }
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [f = 1] * ((((𝑋)[f/0]) * 0.5) + (((𝑋)[c/c + 1]) * (1.0 - 0.5))) + [not (f = 1)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [f = 1] * 0.5 * (𝑋)[f/0] + [f = 1] * (1.0 - 0.5) * (𝑋)[c/c + 1]'


def test_geometric_flipping():
    program = parse_pgcl("""
        nat c;
        nat f;
        nat k;

        while(f=1){
                if(k=0){
                    {f := 0}[0.5]{ c := c +1 };
                    k := 1
                }{
                k :=0
                }
        }
        """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [f = 1] * (([k = 0] * ((((𝑋)[f/0, k/1]) * 0.5) + (((𝑋)[c/c + 1, k/1]) * (1.0 - 0.5)))) + ([not (k = 0)] * ((𝑋)[k/0]))) + [not (f = 1)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(f = 1) & (k = 0)] * 0.5 * (𝑋)[f/0, k/1] + [(f = 1) & (k = 0)] * (1.0 - 0.5) * (𝑋)[c/c + 1, k/1] + [(f = 1) & not (k = 0)] * 1.0 * (𝑋)[k/0]'


def test_geometric_monus():
    program = parse_pgcl("""
        nat c;
        nat f;
        while(f=1){
            {f := 0}[0.5]{ c := c - 1}
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [f = 1] * ((((𝑋)[f/0]) * 0.5) + (((𝑋)[c/c - 1]) * (1.0 - 0.5))) + [not (f = 1)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [f = 1] * 0.5 * (𝑋)[f/0] + [f = 1] * (1.0 - 0.5) * (𝑋)[c/c - 1]'


def test_geometric_monus_2():
    program = parse_pgcl("""
        nat c;
        nat f;
        while(f=1){
            {f := 0}[0.5]{ c := c - 1 + 2}
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [f = 1] * ((((𝑋)[f/0]) * 0.5) + (((𝑋)[c/(c - 1) + 2]) * (1.0 - 0.5))) + [not (f = 1)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [f = 1] * 0.5 * (𝑋)[f/0] + [f = 1] * (1.0 - 0.5) * (𝑋)[c/(c - 1) + 2]'


def test_program_dueling_cowboys_no_count():
    program = parse_pgcl("""
        # we have a variable player: player=i (i=0 or i=1) means it's player i's turn
        nat player;

        # and a variable shot: shot=0 means the current player did not shoot the opponent. shot=1 means he did
        nat shot;

        # In this model, player 0 wins with a higher probability

        while(shot=0){
            if(player = 0){
                {shot := 1}[0.6]{player := 1}
            }{
                {shot := 0}[0.4]{player := 0}
            }
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [shot = 0] * (([player = 0] * ((((𝑋)[shot/1]) * 0.6) + (((𝑋)[player/1]) * (1.0 - 0.6)))) + ([not (player = 0)] * ((((𝑋)[shot/0]) * 0.4) + (((𝑋)[player/0]) * (1.0 - 0.4))))) + [not (shot = 0)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(shot = 0) & (player = 0)] * 0.6 * (𝑋)[shot/1] + [(shot = 0) & (player = 0)] * (1.0 - 0.6) * (𝑋)[player/1] + [(shot = 0) & not (player = 0)] * 0.4 * (𝑋)[shot/0] + [(shot = 0) & not (player = 0)] * (1.0 - 0.4) * (𝑋)[player/0]'


def test_dueling_cowboys_count():
    program = parse_pgcl("""
        # we have a variable player: player=i (i=0 or i=1) means it's player i's turn
        nat player;
        # and a variable shot: shot=0 means the current player did not shoot the opponent. shot=1 means he did
        nat shot;
        nat c;
        # In this model, player 0 wins with a higher probability
        while(shot=0){
            if(player = 0){
                {shot := 1}[0.6]{player := 1}
            }{
                {shot := 1}[0.4]{player := 0}
            }

            c := c + 1
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [shot = 0] * (([player = 0] * ((((𝑋)[shot/1, c/c + 1]) * 0.6) + (((𝑋)[player/1, c/c + 1]) * (1.0 - 0.6)))) + ([not (player = 0)] * ((((𝑋)[shot/1, c/c + 1]) * 0.4) + (((𝑋)[player/0, c/c + 1]) * (1.0 - 0.4))))) + [not (shot = 0)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(shot = 0) & (player = 0)] * 0.6 * (𝑋)[shot/1, c/c + 1] + [(shot = 0) & (player = 0)] * (1.0 - 0.6) * (𝑋)[player/1, c/c + 1] + [(shot = 0) & not (player = 0)] * 0.4 * (𝑋)[shot/1, c/c + 1] + [(shot = 0) & not (player = 0)] * (1.0 - 0.4) * (𝑋)[player/0, c/c + 1]'


def test_loop_forever():
    program = parse_pgcl("""
        while(True){skip}
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(tf) == "λ𝐹. lfp 𝑋. [True] * ((𝑋)[]) + [not True] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(snf) == 'λ𝑋. [True] * 1.0 * (𝑋)[]'


def test_fair_random_walk():
    program = parse_pgcl("""
        nat x;

        while((not x<=0)){
            {x := x+1 }[0.5]{ x := x-1}
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [not x <= 0] * ((((𝑋)[x/x + 1]) * 0.5) + (((𝑋)[x/x - 1]) * (1.0 - 0.5))) + [not (not x <= 0)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [not x <= 0] * 0.5 * (𝑋)[x/x + 1] + [not x <= 0] * (1.0 - 0.5) * (𝑋)[x/x - 1]'


def test_unfair_random_walk():
    program = parse_pgcl("""
        nat x;
        while((not x<=0)){
            {x := x+1 }[0.1]{ x := x-1}
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [not x <= 0] * ((((𝑋)[x/x + 1]) * 0.1) + (((𝑋)[x/x - 1]) * (1.0 - 0.1))) + [not (not x <= 0)] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [not x <= 0] * 0.1 * (𝑋)[x/x + 1] + [not x <= 0] * (1.0 - 0.1) * (𝑋)[x/x - 1]'


def test_brp_simple_parameterized():
    program = parse_pgcl("""
        # The number of total packages to send
        nat toSend;

        # Number of packages sent
        nat sent;

        # The maximal number of retransmission tries
        nat maxFailed;

        # The number of failed retransmission tries
        nat failed;

        while(failed < maxFailed & sent < toSend){
            {
                # Transmission of current packages successful
                failed := 0;
                sent := sent + 1;

            }
            [0.9]
            {
                # Transmission not successful
                failed := failed +1;
            }
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [(failed < maxFailed) & (sent < toSend)] * ((((𝑋)[failed/0, sent/sent + 1]) * 0.9) + (((𝑋)[failed/failed + 1]) * (1.0 - 0.9))) + [not ((failed < maxFailed) & (sent < toSend))] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(failed < maxFailed) & (sent < toSend)] * 0.9 * (𝑋)[failed/0, sent/sent + 1] + [(failed < maxFailed) & (sent < toSend)] * (1.0 - 0.9) * (𝑋)[failed/failed + 1]'


def test_complete_binary_tree():
    program = parse_pgcl("""
        nat a;
        nat b;
        nat c;
        nat maxA;
        nat maxB;
        while (a < maxA & b < maxB) {
            {a:=a+1} [0.5] {b := b+1}
            c:=c+1
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [(a < maxA) & (b < maxB)] * ((((𝑋)[a/a + 1, c/c + 1]) * 0.5) + (((𝑋)[b/b + 1, c/c + 1]) * (1.0 - 0.5))) + [not ((a < maxA) & (b < maxB))] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [(a < maxA) & (b < maxB)] * 0.5 * (𝑋)[a/a + 1, c/c + 1] + [(a < maxA) & (b < maxB)] * (1.0 - 0.5) * (𝑋)[b/b + 1, c/c + 1]'


def test_zero_conf_parameterized():
    program = parse_pgcl("""
        # Variable free is either 0 or 1. Free=1 means hosts received answer address free. Free=1 means a collision occurred.
        # If free=0 holds on termination, then the host erroneously assumes that address is free.
        # (inital state constraint: free=0).
        nat free;

        # answerReceived = 1 (= 0) if host does (not) receive an answer.
        nat answerReceived;

        # count keeps track of the number of times the host requested an answer.
        # (initial state constraint: count = 0)
        nat count;

        nat maxCount;


        while(count < maxCount & free = 0){
            {answerReceived := 0}[0.8]{answerReceived := 1}
            if(answerReceived=1){
                {free:=1}[0.5]{free:=0}
                count := 0
            }{
                count:=count+1
            }
        }
    """)
    tf = one_loop_wp_transformer(program, program.instructions)
    assert str(
        tf
    ) == "λ𝐹. lfp 𝑋. [(count < maxCount) & (free = 0)] * (((([0 = 1] * ((((𝑋)[answerReceived/0, free/1, count/0]) * 0.5) + (((𝑋)[answerReceived/0, free/0, count/0]) * (1.0 - 0.5)))) + ([not (0 = 1)] * ((𝑋)[answerReceived/0, count/count + 1]))) * 0.8) + ((([1 = 1] * ((((𝑋)[answerReceived/1, free/1, count/0]) * 0.5) + (((𝑋)[answerReceived/1, free/0, count/0]) * (1.0 - 0.5)))) + ([not (1 = 1)] * ((𝑋)[answerReceived/1, count/count + 1]))) * (1.0 - 0.8))) + [not ((count < maxCount) & (free = 0))] * 𝐹"
    snf = normalize_expectation_transformer(program, tf.body)
    assert str(
        snf
    ) == 'λ𝑋. [((count < maxCount) & (free = 0)) & (0 = 1)] * (0.5 * 0.8) * (𝑋)[answerReceived/0, free/1, count/0] + [((count < maxCount) & (free = 0)) & (0 = 1)] * ((1.0 - 0.5) * 0.8) * (𝑋)[answerReceived/0, free/0, count/0] + [((count < maxCount) & (free = 0)) & not (0 = 1)] * 0.8 * (𝑋)[answerReceived/0, count/count + 1] + [((count < maxCount) & (free = 0)) & (1 = 1)] * (0.5 * (1.0 - 0.8)) * (𝑋)[answerReceived/1, free/1, count/0] + [((count < maxCount) & (free = 0)) & (1 = 1)] * ((1.0 - 0.5) * (1.0 - 0.8)) * (𝑋)[answerReceived/1, free/0, count/0] + [((count < maxCount) & (free = 0)) & not (1 = 1)] * (1.0 - 0.8) * (𝑋)[answerReceived/1, count/count + 1]'
