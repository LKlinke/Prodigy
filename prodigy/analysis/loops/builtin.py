"""
-----------------------------
The built-in loop strategies
-----------------------------

Importing this module registers Prodigy's own loop strategies (they are defined with the while handler,
:mod:`prodigy.analysis.instructionhandler.while_handler`). The registry itself
(:mod:`prodigy.analysis.loops.api`) imports no strategy, so the vocabulary layer stays below the handlers;
the session and the front-end module :mod:`prodigy.analysis.loop_strategies` import this module, so every
entry point sees the same strategies in the same order.
"""
# pylint: disable=unused-import
import prodigy.analysis.instructionhandler.while_handler  # noqa: F401
