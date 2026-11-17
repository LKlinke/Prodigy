"""An unverified template certificate has no masses; ``to_json`` and the accessors must say so."""
from fractions import Fraction

import pytest

from prodigy.analysis.lowerbounds.egcalc import Denominator, GFCalculus
from prodigy.analysis.lowerbounds.synthesis import Certificate


def _certificate() -> Certificate:
    calc = GFCalculus(["x"], Denominator([Fraction(1, 2)], [1]))
    return Certificate(kind="lower", calc=calc, tail_start=[0], numerator={(0,): Fraction(1)},
                       exit_numerator={(0,): Fraction(1)}, template_size=1, margin=None, time_build=0.0,
                       time_lp=0.0)


def test_unverified_certificate_has_no_masses() -> None:
    cert = _certificate()
    data = cert.to_json()
    assert data["exit_mass"] is None and data["total_mass"] is None
    with pytest.raises(ValueError, match="not verified"):
        cert.verified_exit_mass()


def test_verified_certificate_reports_masses() -> None:
    cert = _certificate()
    cert.exit_mass, cert.total_mass = Fraction(1, 2), Fraction(3, 4)
    data = cert.to_json()
    assert (data["exit_mass"], data["total_mass"]) == ("1/2", "3/4")
    assert cert.verified_exit_mass() == Fraction(1, 2)
