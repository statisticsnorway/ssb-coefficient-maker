"""Pytest module for testing the CoefficientCalculator class."""

import numpy as np
import pandas as pd
import pytest

from ssb_coefficient_maker.coeff_maker import CoefficientCalculator, FormulaEvaluator

# Fixed seed for reproducible tests
SEED = 42


def create_test_data() -> dict[str, pd.DataFrame | pd.Series]:
    """Create test data for CoefficientCalculator tests."""
    rng = np.random.default_rng(seed=SEED)

    # Regular test matrices
    data_dict = {
        "a": pd.DataFrame(rng.integers(low=1, high=10, size=(3, 3))).astype(float),
        "b": pd.DataFrame(rng.integers(low=1, high=5, size=(3, 3))).astype(float),
        "c": pd.Series(rng.integers(low=1, high=10, size=3)).astype(float),
    }

    return data_dict


def create_coefficient_map() -> pd.DataFrame:
    """Create a coefficient map for testing."""
    # Define coefficients with different formula types
    return pd.DataFrame({
        "result_name": [
            "sum_ab",
            "diff_ab",
            "a_times_c",
            "a_divided_by_b",
            "empty_formula"
        ],
        "formula": [
            "a + b",
            "a - b",
            "a * c",
            "a / b",
            ""  # Empty formula to test skipping
        ],
        "description": [
            "Sum of matrices a and b",
            "Difference of matrices a and b",
            "Matrix a multiplied by vector c",
            "Matrix a divided by matrix b (element-wise)",
            "Empty formula to test skipping"
        ]
    })


@pytest.fixture
def coefficient_calculator() -> tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]:
    """Pytest fixture for the CoefficientCalculator with test data."""
    data_dict = create_test_data()
    coef_map = create_coefficient_map()

    calculator = CoefficientCalculator(
        data_dict=data_dict,
        coefficient_map=coef_map,
        result_name_col="result_name",
        formula_name_col="formula",
        adp_enabled=False,  # Use standard precision for faster tests
        fill_invalid=True,  # Replace invalid values with zeros
        verbose=False
    )

    return calculator, data_dict


def test_validate_coefficient_map_headers_valid(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with valid column names."""
    calculator, _ = coefficient_calculator

    # Create coefficient map with required columns
    coef_map = pd.DataFrame({
        "result_col": ["test_result"],
        "formula_col": ["a * 2"]
    })

    # Validate headers explicitly (should not raise any exceptions)
    calculator._validate_coefficient_map_headers(
        coefficient_map=coef_map,
        mandatory_cols=["result_col", "formula_col"]
    )

    # Test passes if no exception is raised


def test_validate_coefficient_map_headers_missing_one(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with one missing column."""
    calculator, _ = coefficient_calculator

    # Create coefficient map missing one required column
    coef_map = pd.DataFrame({
        "result_col": ["test_result"],
        # Missing "formula_col"
    })

    # Validate headers for columns that include a missing one
    with pytest.raises(KeyError) as excinfo:
        calculator._validate_coefficient_map_headers(
            coefficient_map=coef_map,
            mandatory_cols=["result_col", "formula_col"]
        )

    # Verify error message
    assert "formula_col" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


def test_validate_coefficient_map_headers_missing_multiple(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with multiple missing columns."""
    calculator, _ = coefficient_calculator

    # Create coefficient map with different column names
    coef_map = pd.DataFrame({
        "some_other_col": ["test_result"],
    })

    # Validate headers for completely different column names
    with pytest.raises(KeyError) as excinfo:
        calculator._validate_coefficient_map_headers(
            coefficient_map=coef_map,
            mandatory_cols=["result_col", "formula_col"]
        )

    # Verify error message contains both missing columns
    assert "result_col" in str(excinfo.value)
    assert "formula_col" in str(excinfo.value)
    assert "not found" in str(excinfo.value)


def test_validate_coefficient_map_headers_empty_df(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with empty DataFrame."""
    calculator, _ = coefficient_calculator

    # Create empty coefficient map
    coef_map = pd.DataFrame()

    # Validate headers on empty DataFrame
    with pytest.raises(KeyError) as excinfo:
        calculator._validate_coefficient_map_headers(
            coefficient_map=coef_map,
            mandatory_cols=["result_col", "formula_col"]
        )

    # Verify error message
    assert "result_col" in str(excinfo.value)
    assert "formula_col" in str(excinfo.value)


def test_validate_coefficient_map_headers_additional_columns(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with additional non-mandatory columns."""
    calculator, _ = coefficient_calculator

    # Create coefficient map with required columns plus extra ones
    coef_map = pd.DataFrame({
        "result_col": ["test_result"],
        "formula_col": ["a * 2"],
        "description": ["Test coefficient"],
        "author": ["Test User"],
        "date_added": ["2025-03-16"]
    })

    # Validate headers (should not raise any exceptions)
    calculator._validate_coefficient_map_headers(
        coefficient_map=coef_map,
        mandatory_cols=["result_col", "formula_col"]
    )

    # Test passes if no exception is raised


def test_validate_coefficient_map_headers_empty_mandatory_list(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test _validate_coefficient_map_headers with empty mandatory columns list."""
    calculator, _ = coefficient_calculator

    # Create coefficient map
    coef_map = pd.DataFrame({
        "result_col": ["test_result"],
        "formula_col": ["a * 2"]
    })

    # Validate headers with empty mandatory list (should not raise any exceptions)
    calculator._validate_coefficient_map_headers(
        coefficient_map=coef_map,
        mandatory_cols=[]
    )

    # Test passes if no exception is raised


def test_compute_coefficients_basic(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test compute_coefficients with basic operations."""
    calculator, data_dict = coefficient_calculator

    # Compute all coefficients
    results = calculator.compute_coefficients()

    # Check that we got the expected coefficients
    assert "sum_ab" in results
    assert "diff_ab" in results
    assert "a_times_c" in results
    assert "a_divided_by_b" in results
    assert "empty_formula" not in results  # Empty formula should be skipped

    # Check that the results are the expected type
    assert isinstance(results["sum_ab"], pd.DataFrame)
    assert isinstance(results["diff_ab"], pd.DataFrame)
    assert isinstance(results["a_times_c"], pd.DataFrame)
    assert isinstance(results["a_divided_by_b"], pd.DataFrame)

    # Manually calculate expected results for comparison
    a = data_dict["a"]
    b = data_dict["b"]
    c = data_dict["c"]

    expected_sum = a + b
    expected_diff = a - b
    expected_mul = a * c
    expected_div = a / b

    # Replace NaN/Inf with 0 in expected results
    expected_div = expected_div.replace([np.inf, -np.inf, np.nan], 0)

    # Compare with calculated results
    pd.testing.assert_frame_equal(results["sum_ab"].astype(np.float64), expected_sum.astype(np.float64))
    pd.testing.assert_frame_equal(results["diff_ab"].astype(np.float64), expected_diff.astype(np.float64))
    pd.testing.assert_frame_equal(results["a_times_c"].astype(np.float64), expected_mul.astype(np.float64))
    pd.testing.assert_frame_equal(results["a_divided_by_b"].astype(np.float64), expected_div.astype(np.float64))


def test_compute_coefficients_missing_variable(coefficient_calculator: tuple[CoefficientCalculator, dict[str, pd.DataFrame | pd.Series]]) -> None:
    """Test compute_coefficients with a formula that uses a non-existent variable."""
    calculator, _ = coefficient_calculator

    # Add a formula with a missing variable
    new_coef_map = create_coefficient_map()
    new_coef_map = pd.concat([
        new_coef_map,
        pd.DataFrame({
            "result_name": ["missing_var"],
            "formula": ["a + nonexistent_var"],
            "description": ["Formula with missing variable"]
        })
    ])

    # Replace the coefficient map
    calculator.coefficient_map = new_coef_map
