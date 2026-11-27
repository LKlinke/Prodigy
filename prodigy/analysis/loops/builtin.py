"""
-----------------------------
The built-in loop strategies
-----------------------------

Importing this module registers Prodigy's own loop strategies, in the order of the historic menu: the modules
below register theirs at import time. The registry (:mod:`prodigy.analysis.loops.registry`) and the vocabulary
(:mod:`prodigy.analysis.loops.api`) import no strategy, so they stay below the handlers; the session and the
front-end module :mod:`prodigy.analysis.loop_strategies` import this module, so every entry point sees the same
strategies in the same order.
"""
# pylint: disable=unused-import
import prodigy.analysis.loops.invariants.park  # noqa: F401  invariant
import prodigy.analysis.loops.unrolling  # noqa: F401  unroll, mass_threshold
import prodigy.analysis.loops.exploration  # noqa: F401  evt_iterate
import prodigy.analysis.loops.invariants.exact.strategies  # noqa: F401  evt_invariant, evt_synthesis
import prodigy.analysis.loops.invariants.bounds.strategies  # noqa: F401  evt_subinvariant, evt_sandwich
import prodigy.analysis.loops.ai  # noqa: F401  ai_invariant
import prodigy.analysis.loops.auto  # noqa: F401  auto
