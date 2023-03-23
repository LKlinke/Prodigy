import itertools
import random as rng

import pytest

from prodigy.distribution.generating_function import (GeneratingFunction,
                                                      SympyPGF)


@pytest.mark.parametrize('probability',
                         [str(rng.random()) for _ in range(100)])
def test_bernoulli(probability):
    assert GeneratingFunction(
        f"1-{probability} + {probability}* variable") == SympyPGF.bernoulli(
            "variable", probability)


@pytest.mark.parametrize('probability',
                         [str(rng.random()) for _ in range(100)])
def test_geometric(probability):
    assert GeneratingFunction(f"{probability}/(1- (1-{probability})*variable)"
                              ) == SympyPGF.geometric("variable", probability)


@pytest.mark.parametrize('start,end',
                         [(i, i + j)
                          for i, j in itertools.product(range(10), range(10))])
def test_uniform(start, end):
    assert GeneratingFunction(f"variable**{start} / ({end} - {start}+1) *"
                              f"(variable ** ({end} - {start} + 1) - 1) / (variable - 1)") == \
           SympyPGF.uniform("variable", str(start), str(end))


@pytest.mark.parametrize('probability',
                         [str(rng.random()) for _ in range(100)])
def test_log(probability):
    assert GeneratingFunction(
        f"log(1-{probability}*variable)/log(1-{probability})") == SympyPGF.log(
            "variable", probability)


@pytest.mark.parametrize('rate', range(10))
def test_poisson(rate):
    hand = GeneratingFunction(f"exp({rate} * (variable -1))", "variable")
    factory = SympyPGF.poisson("variable", str(rate))
    assert hand == factory


@pytest.mark.parametrize('n,p', [(i, rng.random()) for i in range(20)])
def test_binomial(n, p):
    assert GeneratingFunction(f"(1-{p}+{p}*variable)**{n}",
                              "variable") == SympyPGF.binomial(
                                  "variable", str(n), str(p))
