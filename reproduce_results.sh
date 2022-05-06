#!/bin/bash
echo
echo
echo -e '\033[92m----------------------------- REPLICATING RESULTS FROM THE CAV 2022 PAPER -----------------------------\033[0m'
echo
echo
echo -e '\033[96m--------------------------- Equivalence Checks (using the GiNaC engine) --------------------------\033[0m'
echo

echo -e '\033[93mEXAMPLE #1 (Photorealistic Rendering):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/rendering_inner.pgcl pgfexamples/paper_examples/rendering_inner_inv.pgcl
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/rendering_modified_outer.pgcl pgfexamples/paper_examples/rendering_modified_outer_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #2 & #7 (n-Geometric Distribution Generator):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/n_geometric.pgcl pgfexamples/paper_examples/n_geometric_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #3 (Complementary Binomial Distributions):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/comp_binom_dist.pgcl pgfexamples/paper_examples/comp_binom_dist_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #4 (Dueling Cowboys):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/dueling_cowboys.pgcl pgfexamples/paper_examples/dueling_cowboys_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #5 (Nested Loops):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/nested_loop_inner.pgcl pgfexamples/paper_examples/nested_loop_inner_inv.pgcl
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/nested_loop_modified_outer.pgcl pgfexamples/paper_examples/nested_loop_modified_outer_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #6 (Geometric Distribution Generator):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/geometric.pgcl pgfexamples/paper_examples/geometric_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #8 (IID-Sampling Statement):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/trivial_iid.pgcl pgfexamples/paper_examples/trivial_iid_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #9 (Random Walk):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/random_walk.pgcl pgfexamples/paper_examples/random_walk_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #10 (Knuth-Yao Die):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/ky_die.pgcl pgfexamples/paper_examples/ky_die_inv.pgcl
echo
echo

echo -e '\033[93mEXAMPLE #11 (Sequential Loops):\n\033[0m'
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/trivial_iid.pgcl pgfexamples/paper_examples/trivial_iid_inv.pgcl
python prodigy/cli.py --engine ginac check_equality pgfexamples/paper_examples/random_walk.pgcl pgfexamples/paper_examples/random_walk_inv.pgcl
echo

echo -e '\033[96m----------------------------- Query Examples (using the SymPy engine) -----------------------------\033[0m'
echo
echo

echo -e '\033[93mEXAMPLE #1 (Photorealistic Rendering -- with input distribution \033[35mn^5\033[93m):\n\033[0m'
echo -e "1\\npgfexamples/paper_examples/rendering_modified_outer_inv.pgcl" | python prodigy/cli.py main pgfexamples/paper_examples/rendering_modified_outer.pgcl "n^5"
echo

echo -e '\033[93mEXAMPLE #3 (Complementary Binomial Distributions -- with input distribution \033[35mc^10\033[93m):\n\033[0m'
echo -e "1\\npgfexamples/paper_examples/comp_binom_dist_inv.pgcl" | python prodigy/cli.py main pgfexamples/paper_examples/comp_binom_dist.pgcl "c^10"
echo
echo

echo -e '\033[93mEXAMPLE #9 (Random Walk -- with input distribution \033[35ms^1\033[93m):\n\033[0m'
echo -e "1\\npgfexamples/paper_examples/random_walk_inv.pgcl" | python prodigy/cli.py main pgfexamples/paper_examples/random_walk.pgcl "s"
echo
echo
