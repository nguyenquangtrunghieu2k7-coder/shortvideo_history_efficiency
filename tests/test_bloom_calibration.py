
import pytest

from src.bloom_pipeline.calibration import (
    calibrate_from_observation,
    estimate_cardinality,
    optimal_bit_count,
    optimal_hash_count,
    theoretical_fpr,
)


def test_optimal_sizing_returns_valid_parameters():
    m = optimal_bit_count(n=1000, target_fpr=0.01)
    k = optimal_hash_count(m=m, n=1000)

    assert m > 0
    assert k >= 1
    assert theoretical_fpr(m=m, n=1000, k=k) <= 0.02


def test_cardinality_estimate_is_close_to_observed_count():
    n = 100
    m = 8192
    k = 4

    expected_fill = 1 - (1 - 1 / m) ** (k * n)
    estimated = estimate_cardinality(m, k, expected_fill)

    assert estimated == pytest.approx(n, rel=0.02)


def test_calibration_recommends_future_configuration():
    result = calibrate_from_observation(
        m=8192,
        k=4,
        fill_ratio=0.1,
        target_fpr=0.01,
    )

    assert result["estimated_cardinality"] > 0
    assert result["recommended_m"] > 0
    assert result["recommended_k"] >= 1
    assert result["predicted_fpr"] <= 0.02


def test_calibration_rejects_saturated_filter():
    with pytest.raises(ValueError):
        calibrate_from_observation(
            m=1024,
            k=3,
            fill_ratio=1.0,
        )
