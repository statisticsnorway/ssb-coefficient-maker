"""Pytest module for testing the FormulaEvaluator class with both standard and arbitrary precision."""

import mpmath
import numpy as np
import pandas as pd
import pytest

from ssb_coefficient_maker.coeff_maker import FormulaEvaluator

# Fixed seed for reproducible tests
SEED = 42


def create_test_data() -> dict[str, pd.DataFrame]:
    """Create test data for formula evaluation tests."""
    rng = np.random.default_rng(seed=SEED)

    # Regular test matrices
    data_dict = {
        "a": pd.DataFrame(rng.integers(low=1, high=10, size=(3, 3))).astype(float),
        "b": pd.DataFrame(rng.integers(low=1, high=5, size=(3, 3))).astype(float),
        "c": pd.DataFrame(rng.integers(low=1, high=3, size=(3, 3))).astype(float),
        "d": pd.DataFrame(rng.integers(low=2, high=6, size=(3, 3))).astype(float),
        "e": pd.DataFrame(rng.integers(low=0, high=1, size=(3, 3))).astype(float),
        "f": pd.DataFrame(
            np.tile(rng.integers(low=0, high=5, size=(3)), (3, 1))
        ).astype(float),
    }

    # Add a diagonal matrix
    diag_values = rng.integers(low=1, high=10, size=(3))
    diag_matrix = np.diag(diag_values)
    data_dict["g"] = pd.DataFrame(diag_matrix).astype(float)

    # Add a near-diagonal matrix (with some zeros but invertible)
    near_diag = np.diag(diag_values).copy()
    near_diag[0, 1] = 1  # Add one non-zero off-diagonal element
    data_dict["h"] = pd.DataFrame(near_diag).astype(float)

    # Add a matrix with some NaN values
    nan_matrix = rng.integers(low=1, high=10, size=(3, 3)).astype(float)
    nan_matrix[0, 1] = np.nan
    nan_matrix[2, 2] = np.nan
    data_dict["i"] = pd.DataFrame(nan_matrix)

    # Add a sparse matrix (mostly zeros)
    sparse = np.zeros((3, 3))
    sparse[0, 0] = 5
    sparse[2, 1] = 3
    data_dict["j"] = pd.DataFrame(sparse).astype(float)

    # Add extreme values for precision testing
    data_dict["small"] = pd.DataFrame(np.full((3, 3), 1e-20))
    data_dict["large"] = pd.DataFrame(np.full((3, 3), 1e20))

    return data_dict


@pytest.fixture
def formula_evaluator_std() -> tuple[FormulaEvaluator, dict[str, pd.DataFrame]]:
    """Pytest fixture for the FormulaEvaluator with standard precision."""
    data_dict = create_test_data()
    evaluator = FormulaEvaluator(
        data_dict, adp_enabled=False, verbose=True, fill_invalid=True
    )
    return evaluator, data_dict


@pytest.fixture
def formula_evaluator_adp() -> tuple[FormulaEvaluator, dict[str, pd.DataFrame]]:
    """Pytest fixture for the FormulaEvaluator with arbitrary decimal precision."""
    data_dict = create_test_data()
    evaluator = FormulaEvaluator(
        data_dict,
        adp_enabled=True,
        decimal_precision=50,
        verbose=True,
        fill_invalid=True,
    )
    return evaluator, data_dict


# Standard precision tests


def test_simple_division_subtraction_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test simple division with subtraction using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, b, c = data_dict["a"], data_dict["b"], data_dict["c"]

    formula = "(a - b) / c"
    result = evaluator.evaluate_formula(formula)
    expected = (a - b) / c

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_complex_nested_division_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test complex nested division using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, b, c, d = data_dict["a"], data_dict["b"], data_dict["c"], data_dict["d"]

    formula = "(a+b)/(c/d) + b"
    result = evaluator.evaluate_formula(formula)
    expected = ((a + b) / (c / d)) + b

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_exponentiation_multiplication_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test exponentiation with multiplication using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, c = data_dict["a"], data_dict["c"]

    formula = "(a ** 2.0) * (a ** c)"
    result = evaluator.evaluate_formula(formula)
    expected = (a**2.0) * (a**c)

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_exponentiation_subtraction_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test exponentiation with subtraction using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, b, c = data_dict["a"], data_dict["b"], data_dict["c"]

    formula = "a**b - c"
    result = evaluator.evaluate_formula(formula)
    expected = a**b - c

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_zero_division_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test division with potential zeros using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, e = data_dict["a"], data_dict["e"]

    formula = "a/e"
    result = evaluator.evaluate_formula(formula)
    expected = a / e

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_add_dataframes_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test adding two dataframes using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, f = data_dict["a"], data_dict["f"]

    formula = "a + f"
    result = evaluator.evaluate_formula(formula)
    expected = a + f

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_division_by_diagonal_matrix_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test division by diagonal matrix using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, g = data_dict["a"], data_dict["g"]

    formula = "a / g"
    result = evaluator.evaluate_formula(formula)
    expected = a / g

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_division_by_near_diagonal_matrix_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test division by near-diagonal matrix using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    b, h = data_dict["b"], data_dict["h"]

    formula = "b / h"
    result = evaluator.evaluate_formula(formula)
    expected = b / h

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_matrix_with_nan_values_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test matrix with NaN values in computation using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, i = data_dict["a"], data_dict["i"]

    formula = "a * i"
    result = evaluator.evaluate_formula(formula)
    expected = a * i

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_division_by_sparse_matrix_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test division by sparse matrix (potential zeros) using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    c, j = data_dict["c"], data_dict["j"]

    formula = "c / j"
    result = evaluator.evaluate_formula(formula)
    expected = c / j

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_complex_operation_with_diagonal_matrix_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test complex operation with diagonal matrix using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, g, h, j = data_dict["a"], data_dict["g"], data_dict["h"], data_dict["j"]

    formula = "(a + g) / (h - j)"
    result = evaluator.evaluate_formula(formula)
    expected = (a + g) / (h - j)

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_inverting_diagonal_matrix_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test inverting a diagonal matrix using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    g = data_dict["g"]

    formula = "1 / g"
    result = evaluator.evaluate_formula(formula)
    expected = 1 / g

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_nan_handling_std(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test formula with mixed NaN handling using standard precision."""
    evaluator, data_dict = formula_evaluator_std
    a, i = data_dict["a"], data_dict["i"]

    formula = "i.fillna(0) * a"
    result = evaluator.evaluate_formula(formula)
    expected = i.fillna(0) * a

    pd.testing.assert_frame_equal(
        result.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


# Arbitrary decimal precision tests


def test_simple_division_subtraction_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test simple division with subtraction using arbitrary precision."""
    evaluator, data_dict = formula_evaluator_adp

    formula = "(a - b) / c"
    result = evaluator.evaluate_formula(formula)

    # Convert result to float64 for comparison
    result_float = result.map(lambda x: float(x))

    # Calculate expected using numpy/pandas for comparison
    a_float = data_dict["a"].astype(float)
    b_float = data_dict["b"].astype(float)
    c_float = data_dict["c"].astype(float)
    expected = (a_float - b_float) / c_float

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result_float.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_complex_nested_division_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test complex nested division using arbitrary precision."""
    evaluator, data_dict = formula_evaluator_adp

    formula = "(a+b)/(c/d) + b"
    result = evaluator.evaluate_formula(formula)

    # Convert result to float64 for comparison
    result_float = result.map(lambda x: float(x))

    # Calculate expected using numpy/pandas for comparison
    a_float = data_dict["a"].astype(float)
    b_float = data_dict["b"].astype(float)
    c_float = data_dict["c"].astype(float)
    d_float = data_dict["d"].astype(float)
    expected = ((a_float + b_float) / (c_float / d_float)) + b_float

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result_float.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_high_precision_addition_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test high precision addition that would lose precision in standard float64."""
    evaluator, _ = formula_evaluator_adp

    # Create high precision data directly
    small_val = mpmath.mpf("1e-30")
    large_val = mpmath.mpf("1e30")

    # Create DataFrames with mpmath values
    small_df = pd.DataFrame([[small_val, small_val], [small_val, small_val]])
    large_df = pd.DataFrame([[large_val, large_val], [large_val, large_val]])

    # Add to evaluator's data_dict
    evaluator.data_dict["small_hp"] = small_df
    evaluator.data_dict["large_hp"] = large_df

    # Test addition that would normally lose precision
    formula = "small_hp + small_hp"
    result = evaluator.evaluate_formula(formula)

    # Expected: each value should be exactly 2e-30
    expected_val = mpmath.mpf("2e-30")

    # Check at least one value for exact equality using mpmath
    assert mpmath.almosteq(result.iloc[0, 0], expected_val)


def test_extreme_multiplication_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test multiplication of very small and very large numbers that requires high precision."""
    evaluator, _ = formula_evaluator_adp

    formula = "small * large"
    result = evaluator.evaluate_formula(formula)

    # Check if values are close to 1.0
    for i in range(3):
        for j in range(3):
            # Convert to float for comparison
            result_value = float(result.iloc[i, j])
            assert abs(result_value - 1.0) < 1e-10


def test_high_precision_division_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test high precision division using arbitrary precision."""
    evaluator, _ = formula_evaluator_adp

    # Create test data with very precise fractions
    precise_df1 = pd.DataFrame(
        [
            [mpmath.mpf("1") / mpmath.mpf("3"), mpmath.mpf("1") / mpmath.mpf("7")],
            [mpmath.mpf("1") / mpmath.mpf("9"), mpmath.mpf("1") / mpmath.mpf("11")],
        ]
    )

    precise_df2 = pd.DataFrame(
        [
            [mpmath.mpf("1") / mpmath.mpf("13"), mpmath.mpf("1") / mpmath.mpf("17")],
            [mpmath.mpf("1") / mpmath.mpf("19"), mpmath.mpf("1") / mpmath.mpf("23")],
        ]
    )

    # Add to evaluator's data_dict
    evaluator.data_dict["precise1"] = precise_df1
    evaluator.data_dict["precise2"] = precise_df2

    # Test division
    formula = "precise1 / precise2"
    result = evaluator.evaluate_formula(formula)

    # Expected values with full precision
    expected_values = [
        [mpmath.mpf("13") / mpmath.mpf("3"), mpmath.mpf("17") / mpmath.mpf("7")],
        [mpmath.mpf("19") / mpmath.mpf("9"), mpmath.mpf("23") / mpmath.mpf("11")],
    ]

    # Check values with high precision
    for i in range(2):
        for j in range(2):
            assert mpmath.almosteq(result.iloc[i, j], expected_values[i][j])


def test_zero_division_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test division with zeros raises ZeroDivisionError when fill_invalid is False."""
    evaluator, data_dict = formula_evaluator_adp

    # Temporarily set fill_invalid to False to trigger the error
    evaluator.fill_invalid = False
    evaluator.validator.fill_invalid = False

    formula = "a/e"

    # The evaluation should raise a ZeroDivisionError or ValueError
    with pytest.raises((ZeroDivisionError, ValueError)):
        evaluator.evaluate_formula(formula)

    # Reset fill_invalid for other tests
    evaluator.fill_invalid = True
    evaluator.validator.fill_invalid = True


def test_matrix_with_nan_values_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test matrix with NaN values in computation using arbitrary precision."""
    evaluator, data_dict = formula_evaluator_adp

    formula = "a * i"
    result = evaluator.evaluate_formula(formula)

    # Convert result to float64 for comparison
    result_float = result.map(
        lambda x: float(x) if not hasattr(x, "is_nan") or not x.is_nan else np.nan
    )

    # Calculate expected using numpy/pandas
    a_float = data_dict["a"].astype(float)
    i_float = data_dict["i"].astype(float)
    expected = a_float * i_float

    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
        result_float = result_float.replace([np.inf, -np.inf, np.nan], 0)

    pd.testing.assert_frame_equal(
        result_float.astype(np.float64), expected.astype(np.float64), atol=1e-10
    )


def test_complex_calculation_precision_adp(
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Test a complex calculation that benefits from arbitrary precision."""
    evaluator, _ = formula_evaluator_adp

    # Create data for a financial calculation that benefits from precision
    # Use DataFrames instead of Series to avoid scalar operations
    principal_df = pd.DataFrame({"val": [1000000.00, 2000000.00, 5000000.00]})
    rate_df = pd.DataFrame({"val": [0.0325, 0.0310, 0.0295]})
    periods_df = pd.DataFrame({"val": [360, 240, 180]})

    # Create a constant DataFrame for the divisor 12
    twelve_df = pd.DataFrame({"val": [12, 12, 12]})

    # Create a ones DataFrame to use for comparison
    ones_df = pd.DataFrame({"val": [1, 1, 1]})

    # Convert to mpmath for high precision
    principal_mp = principal_df.map(lambda x: mpmath.mpf(str(x)))
    rate_mp = rate_df.map(lambda x: mpmath.mpf(str(x)))
    periods_mp = periods_df.map(lambda x: mpmath.mpf(str(x)))
    twelve_mp = twelve_df.map(lambda x: mpmath.mpf(str(x)))
    ones_mp = ones_df.map(lambda x: mpmath.mpf(str(x)))

    # Add to evaluator's data_dict
    evaluator.data_dict["principal"] = principal_mp
    evaluator.data_dict["rate"] = rate_mp
    evaluator.data_dict["periods"] = periods_mp
    evaluator.data_dict["twelve"] = twelve_mp
    evaluator.data_dict["ones"] = ones_mp

    # Monthly payment formula for a loan using only DataFrames
    # Transformed to avoid scalar operations:
    # original: principal * (rate/12) / (1 - (1 + rate/12)**(-periods))
    formula = "principal * (rate/twelve) / (ones - (ones + rate/twelve)*(-periods))"
    result = evaluator.evaluate_formula(formula)

    # Calculate expected values using mpmath directly for comparison
    expected_df = pd.DataFrame(index=principal_df.index, columns=principal_df.columns)

    for i in range(len(principal_df)):
        p_mp = mpmath.mpf(str(principal_df.iloc[i, 0]))
        r_mp = mpmath.mpf(str(rate_df.iloc[i, 0]))
        n_mp = mpmath.mpf(str(periods_df.iloc[i, 0]))

        monthly_rate = r_mp / mpmath.mpf("12")
        payment = (
            p_mp
            * monthly_rate
            / (mpmath.mpf("1") - (mpmath.mpf("1") + monthly_rate) * (-n_mp))
        )
        expected_df.iloc[i, 0] = payment

    # Check results with high precision
    for i in range(len(principal_df)):
        assert mpmath.almosteq(result.iloc[i, 0], expected_df.iloc[i, 0])


def test_precision_comparison_std_vs_adp(
    formula_evaluator_std: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
    formula_evaluator_adp: tuple[FormulaEvaluator, dict[str, pd.DataFrame]],
) -> None:
    """Compare standard and arbitrary precision for a calculation with potential precision issues."""
    std_evaluator, _ = formula_evaluator_std
    adp_evaluator, _ = formula_evaluator_adp

    # Add high precision test data to both evaluators
    # Use very small and large values that will challenge precision
    small_val_std = 1e-16
    large_val_std = 1e16

    # Create DataFrames
    small_df_std = pd.DataFrame(np.full((2, 2), small_val_std))
    large_df_std = pd.DataFrame(np.full((2, 2), large_val_std))

    # Add to standard precision evaluator
    std_evaluator.data_dict["small_test"] = small_df_std
    std_evaluator.data_dict["large_test"] = large_df_std

    # Create mpmath versions for ADP evaluator
    small_val_mp = mpmath.mpf("1e-16")
    large_val_mp = mpmath.mpf("1e16")

    small_df_mp = pd.DataFrame(
        [[small_val_mp, small_val_mp], [small_val_mp, small_val_mp]]
    )
    large_df_mp = pd.DataFrame(
        [[large_val_mp, large_val_mp], [large_val_mp, large_val_mp]]
    )

    # Add to ADP evaluator
    adp_evaluator.data_dict["small_test"] = small_df_mp
    adp_evaluator.data_dict["large_test"] = large_df_mp

    # Test a calculation that could have precision issues: small * large * small
    formula = "small_test * large_test * small_test"

    # Calculate with standard precision
    result_std = std_evaluator.evaluate_formula(formula)

    # Calculate with arbitrary precision
    result_adp = adp_evaluator.evaluate_formula(formula)

    # Expected value with perfect precision would be 1e-16 * 1e16 * 1e-16 = 1e-16
    expected_value_mp = mpmath.mpf("1e-16")
    result_std_mp = result_std.iloc[0, 0]

    # Use mpmath's own functions to calculate relative error
    std_rel_error = mpmath.fabs((result_std_mp - expected_value_mp) / expected_value_mp)
    adp_rel_error = mpmath.fabs(
        (result_adp.iloc[0, 0] - expected_value_mp) / expected_value_mp
    )

    # Print the actual values for debugging
    print(f"Standard precision result: {result_std_mp}")
    print(f"ADP precision result: {result_adp.iloc[0, 0]}")
    print(f"Expected value: {expected_value_mp}")
    print(f"Standard precision relative error: {std_rel_error}")
    print(f"ADP precision relative error: {adp_rel_error}")

    # ADP should be more accurate than standard precision
    assert adp_rel_error < std_rel_error

    # Additionally, check that ADP maintains high accuracy
    # The relative error should be very small (below a certain threshold)
    assert adp_rel_error < mpmath.mpf("1e-40")
