"""``State`` is used as a dictionary key (explicit exploration, #359): equal states hash equally."""
from prodigy.distribution.distribution import State


def test_empty_state_is_hashable():
    assert isinstance(hash(State()), int)
    assert hash(State()) == hash(State({}))


def test_hash_ignores_insertion_order():
    a = State({"x": 1, "y": 2})
    b = State({"y": 2, "x": 1})
    assert a == b
    assert hash(a) == hash(b)


def test_hash_uses_all_entries():
    hashes = {hash(State({"x": 1, "y": v})) for v in range(50)}
    assert len(hashes) > 1


def test_state_as_dictionary_key():
    table = {State({"x": 1, "y": 2}): "a"}
    assert table[State({"y": 2, "x": 1})] == "a"
    assert State({"x": 1}) not in table
    assert State() not in table
