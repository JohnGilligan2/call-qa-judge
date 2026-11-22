"""Quadratic-weighted kappa, checked against a hand-worked example."""
from __future__ import annotations

import pytest

from callqa.kappa import format_kappa, landis_koch, paired_scores, quadratic_weighted_kappa


def test_hand_worked_three_category_example():
    """Categories {0,1,2}, w_ij = (i-j)^2 / 4.

    Pairs (judge, human): (0,0) (1,1) (2,2) (0,1) (1,0) (2,1). N = 6.
    Observed weighted disagreement: 0.25 + 0.25 + 0.25 = 0.75.
    Marginals: judge 2/2/2, human 2/3/1. Expected E_ij = r_i * c_j / 6:
        0.25*1.000 + 1.0*0.333 + 0.25*0.667 + 0.25*0.333
        + 1.0*0.667 + 0.25*1.000 = 1.750
    kappa_w = 1 - 0.75 / 1.75 = 4/7
    """
    k = quadratic_weighted_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 0, 1], categories=(0, 1, 2))
    assert k == pytest.approx(4 / 7, abs=1e-9)
    assert landis_koch(k) == "moderate"


def test_perfect_agreement_is_one():
    s = [0, 3, 5, 7, 10, 2, 8]
    assert quadratic_weighted_kappa(s, s) == pytest.approx(1.0)


def test_reversed_scores_are_worse_than_chance():
    k = quadratic_weighted_kappa([0, 2, 4, 6, 8, 10], [10, 8, 6, 4, 2, 0])
    assert k is not None and k < 0


def test_the_offset_ladder_is_the_documented_one():
    """Pins the numbers the module docstring quotes. Equally fixable offsets,
    very different kappas: why the gap, not kappa, is the primary signal."""
    human = [3, 4, 5, 6, 7, 8, 9, 5, 6, 7]
    k1 = quadratic_weighted_kappa([h - 1 for h in human], human)
    k2 = quadratic_weighted_kappa([h - 2 for h in human], human)
    k3 = quadratic_weighted_kappa([h - 3 for h in human], human)
    assert k1 == pytest.approx(0.857, abs=0.005)
    assert k2 == pytest.approx(0.600, abs=0.005)
    assert k3 == pytest.approx(0.400, abs=0.005)


def test_a_constant_judge_against_a_varying_human_is_exactly_zero():
    assert quadratic_weighted_kappa([5] * 6, [5, 6, 7, 5, 6, 7]) == 0.0


def test_scatter_loses_to_a_small_offset():
    human = [3, 4, 5, 6, 7, 8, 5, 4, 6, 7]
    judge = [h + d for h, d in zip(human, [3, -3] * 5)]
    assert quadratic_weighted_kappa(judge, human) < quadratic_weighted_kappa([h - 1 for h in human], human)


def test_constant_raters_are_undefined_not_zero():
    assert quadratic_weighted_kappa([5, 5, 5], [5, 5, 5]) is None
    assert quadratic_weighted_kappa([5], [5]) is None
    assert quadratic_weighted_kappa([], []) is None


def test_out_of_range_values_are_clamped_not_dropped():
    assert quadratic_weighted_kappa([0, 5, 12], [0, 5, 10]) == pytest.approx(
        quadratic_weighted_kappa([0, 5, 10], [0, 5, 10]))


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([1, 2], [1])


def test_paired_scores_drops_declines():
    js, hs = paired_scores({"a": 5, "b": 6, "c": 7, "d": None}, {"a": 5, "b": "ungradable", "c": 9, "d": 4})
    assert js == [5, 7] and hs == [5, 9]


@pytest.mark.parametrize("k,band", [
    (None, "n/a"), (-0.1, "worse than chance"), (0.1, "slight"), (0.3, "fair"),
    (0.5, "moderate"), (0.7, "substantial"), (0.9, "almost perfect"),
])
def test_landis_koch_bands(k, band):
    assert landis_koch(k) == band


def test_format_kappa():
    assert format_kappa(0.57) == "0.57 (moderate)"
    assert format_kappa(None) == "n/a"
