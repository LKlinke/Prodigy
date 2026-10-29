# Loops with conditioning (not supported yet)

The lower-bound modes (sub-invariant, sandwich, Newton) do not support `observe`: a faithful treatment of
conditioning has to distinguish violated observations from non-termination (Olmedo et al., TOPLAS 2018;
Klinkenberg et al., OOPSLA 2024), whereas the occupation-measure rules of these modes treat all lost mass alike.
The programs in this folder are kept for future work; the modes reject them with an `UnsupportedProgram` error.
