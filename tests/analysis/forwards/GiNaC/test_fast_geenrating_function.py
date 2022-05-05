from prodigy.analysis.forward.pgfs import ProdigyPGF


def test_geometric():
    dist = ProdigyPGF.geometric("x", "1/2")
    print(dist)
    assert True