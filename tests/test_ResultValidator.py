"""Pytest module for testing the _ResultValidator class."""

import warnings
from typing import Any

import mpmath
import numpy as np
import pandas as pd
import pytest
import sympy as sp

from ssb_coefficient_maker.coeff_maker import _ResultValidator

# Fixed seed for reproducible tests
SEED = 42


def create_test_data() -> dict[str, pd.DataFrame | pd.Series]:
    """Create test data for validator tests."""
    rng = np.random.default_rng(seed=SEED)

    # Regular test matrices
    data_dict = {
        "normal_df": pd.DataFrame(rng.integers(low=1, high=10, size=(3, 3))).astype(
            float
        ),
        "small_df": pd.DataFrame(rng.integers(low=1, high=5, size=(3, 3))).astype(
            float
        ),
        "normal_series": pd.Series(rng.integers(low=1, high=10, size=3)).astype(float),
    }

    # Add dataframe with zeros (potential division by zero)
    zero_df = pd.DataFrame(np.zeros((3, 3)))
    zero_df.iloc[1, 1] = 1.0  # One non-zero value
    data_dict["zero_df"] = zero_df

    # Add series with zeros
    zero_series = pd.Series([0.0, 1.0, 0.0])
    data_dict["zero_series"] = zero_series

    # Add a matrix with NaN values
    nan_df = pd.DataFrame(rng.integers(low=1, high=10, size=(3, 3))).astype(float)
    nan_df.iloc[0, 1] = np.nan
    nan_df.iloc[2, 2] = np.nan
    data_dict["nan_df"] = nan_df

    # Add a series with NaN values
    nan_series = pd.Series([1.0, np.nan, 3.0])
    data_dict["nan_series"] = nan_series

    # Add a dataframe with both zeros and NaNs
    mixed_df = pd.DataFrame(rng.integers(low=0, high=10, size=(3, 3))).astype(float)
    mixed_df.iloc[0, 0] = 0.0
    mixed_df.iloc[1, 1] = np.nan
    data_dict["mixed_df"] = mixed_df

    return data_dict


@pytest.fixture
def result_validator() -> tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]]:
    """Pytest fixture for the _ResultValidator with test data."""
    data_dict = create_test_data()
    validator = _ResultValidator(fill_invalid=False, verbose=False, adp_enabled=False)
    return validator, data_dict


@pytest.fixture
def result_validator_fill_invalid() -> (
    tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]]
):
    """Pytest fixture for the _ResultValidator with fill_invalid=True."""
    data_dict = create_test_data()
    validator = _ResultValidator(fill_invalid=True, verbose=False, adp_enabled=False)
    return validator, data_dict


@pytest.fixture
def result_validator_adp() -> (
    tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]]
):
    """Pytest fixture for the _ResultValidator with arbitrary decimal precision enabled."""
    data_dict = create_test_data()

    # Convert data to mpmath float objects
    mpmath.mp.dps = 35  # Set precision
    adp_data = {}
    for key, val in data_dict.items():
        if isinstance(val, pd.DataFrame):
            adp_data[key] = val.map(lambda x: mpmath.mpf(x) if not pd.isna(x) else x)
        else:  # Series
            adp_data[key] = val.apply(lambda x: mpmath.mpf(x) if not pd.isna(x) else x)

    validator = _ResultValidator(fill_invalid=False, verbose=False, adp_enabled=True)
    return validator, adp_data


def test_get_invalid_mask_dataframe(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _get_invalid_mask method with a DataFrame containing invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with NaN and Inf values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, -np.inf]})

    # Get the invalid mask
    invalid_mask = validator._get_invalid_mask(test_df)

    # Expected mask
    expected = pd.DataFrame({"A": [False, True, False], "B": [True, False, True]})

    # Verify correct mask was created
    pd.testing.assert_frame_equal(invalid_mask, expected)


def test_get_invalid_mask_series(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _get_invalid_mask method with a Series containing invalid values."""
    validator, data_dict = result_validator

    # Create a test series with NaN and Inf values
    test_series = pd.Series([1.0, np.nan, np.inf, 4.0, -np.inf])

    # Get the invalid mask
    invalid_mask = validator._get_invalid_mask(test_series)

    # Expected mask
    expected = pd.Series([False, True, True, False, True])

    # Verify correct mask was created
    pd.testing.assert_series_equal(invalid_mask, expected)


def test_count_invalid_values(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _count_invalid_values method with known invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with 3 invalid values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, -np.inf]})

    # Count invalid values
    invalid_count = validator._count_invalid_values(test_df)

    # Verify count
    assert invalid_count == 3, f"Expected 3 invalid values, got {invalid_count}"


def test_fill_invalid_values_dataframe(
    result_validator_fill_invalid: tuple[
        _ResultValidator, dict[str, pd.DataFrame | pd.Series]
    ],
) -> None:
    """Test _fill_invalid_values method with a DataFrame."""
    validator, data_dict = result_validator_fill_invalid

    # Create a test dataframe with NaN and Inf values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, -np.inf]})

    # Fill invalid values
    filled_df = validator._fill_invalid_values(test_df)

    # Expected result
    expected = pd.DataFrame({"A": [1.0, 0.0, 3.0], "B": [0.0, 5.0, 0.0]})

    # Verify invalid values were replaced with zeros
    pd.testing.assert_frame_equal(filled_df, expected)


def test_fill_invalid_values_series(
    result_validator_fill_invalid: tuple[
        _ResultValidator, dict[str, pd.DataFrame | pd.Series]
    ],
) -> None:
    """Test _fill_invalid_values method with a Series."""
    validator, data_dict = result_validator_fill_invalid

    # Create a test series with NaN and Inf values
    test_series = pd.Series([1.0, np.nan, np.inf, 4.0, -np.inf])

    # Fill invalid values
    filled_series = validator._fill_invalid_values(test_series)

    # Expected result
    expected = pd.Series([1.0, 0.0, 0.0, 4.0, 0.0])

    # Verify invalid values were replaced with zeros
    pd.testing.assert_series_equal(filled_series, expected)


def test_check_invalid_status_all_invalid(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _check_invalid_status method with all invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with all invalid values
    test_df = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [np.inf, np.inf, np.inf]}
    )

    # Check invalid status
    all_invalid, some_invalid, has_nan, has_inf = validator._check_invalid_status(
        test_df
    )

    # Verify status
    assert all_invalid, "Expected all_invalid to be True"
    assert not some_invalid, "Expected some_invalid to be False"
    assert has_nan, "Expected has_nan to be True"
    assert has_inf, "Expected has_inf to be True"


def test_check_invalid_status_some_invalid(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _check_invalid_status method with some invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with some invalid values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, 6.0]})

    # Check invalid status
    all_invalid, some_invalid, has_nan, has_inf = validator._check_invalid_status(
        test_df
    )

    # Verify status
    assert not all_invalid, "Expected all_invalid to be False"
    assert some_invalid, "Expected some_invalid to be True"
    assert has_nan, "Expected has_nan to be True"
    assert has_inf, "Expected has_inf to be True"


def test_check_invalid_status_no_invalid(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _check_invalid_status method with no invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with no invalid values
    test_df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

    # Check invalid status
    all_invalid, some_invalid, has_nan, has_inf = validator._check_invalid_status(
        test_df
    )

    # Verify status
    assert not all_invalid, "Expected all_invalid to be False"
    assert not some_invalid, "Expected some_invalid to be False"
    assert not has_nan, "Expected has_nan to be False"
    assert not has_inf, "Expected has_inf to be False"


def test_parse_formula(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _parse_formula method with a simple formula."""
    validator, data_dict = result_validator

    # Formula string
    formula_str = "normal_df + small_df"

    # Parse formula
    expr = validator._parse_formula(formula_str, data_dict)

    # Verify the expression has the expected structure
    assert isinstance(expr, sp.Add), f"Expected sp.Add, got {type(expr)}"
    assert len(expr.args) == 2, f"Expected 2 arguments, got {len(expr.args)}"

    # Verify variable names in expression
    variables = [str(s) for s in expr.free_symbols]
    assert "normal_df" in variables, "Expected 'normal_df' in variables"
    assert "small_df" in variables, "Expected 'small_df' in variables"


def test_extract_variables(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _extract_variables method with a sympy expression."""
    validator, data_dict = result_validator

    # Create sympy symbols
    a = sp.Symbol("normal_df")
    b = sp.Symbol("small_df")
    c = sp.Symbol("normal_series")

    # Create expression
    expr = a + b * c

    # Extract variables
    variables = validator._extract_variables(expr)

    # Verify extracted variables
    assert set(variables) == {
        "normal_df",
        "small_df",
        "normal_series",
    }, f"Expected {{'normal_df', 'small_df', 'normal_series'}}, got {set(variables)}"


def test_check_variable_mixture_mixed(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _check_variable_mixture method with mixed variable types."""
    validator, data_dict = result_validator

    # Formula with mixed variable types
    formula_str = "normal_df * normal_series"

    # Check variable mixture
    variables, series_vars, df_vars, mixture_issue = validator._check_variable_mixture(
        formula_str, data_dict
    )

    # Verify results
    assert set(variables) == {
        "normal_df",
        "normal_series",
    }, f"Expected {{'normal_df', 'normal_series'}}, got {set(variables)}"
    assert series_vars == [
        "normal_series"
    ], f"Expected ['normal_series'], got {series_vars}"
    assert df_vars == ["normal_df"], f"Expected ['normal_df'], got {df_vars}"
    assert mixture_issue, "Expected mixture_issue to be True"


def test_check_variable_mixture_only_df(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _check_variable_mixture method with only DataFrame variables."""
    validator, data_dict = result_validator

    # Formula with only DataFrame variables
    formula_str = "normal_df + small_df"

    # Check variable mixture
    variables, series_vars, df_vars, mixture_issue = validator._check_variable_mixture(
        formula_str, data_dict
    )

    # Verify results
    assert set(variables) == {
        "normal_df",
        "small_df",
    }, f"Expected {{'normal_df', 'small_df'}}, got {set(variables)}"
    assert series_vars == [], f"Expected [], got {series_vars}"
    assert set(df_vars) == {
        "normal_df",
        "small_df",
    }, f"Expected {{'normal_df', 'small_df'}}, got {set(df_vars)}"
    assert not mixture_issue, "Expected mixture_issue to be False"


def test_handle_all_invalid_mixture(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _handle_all_invalid method with mixed variable types."""
    validator, data_dict = result_validator

    # Setup test parameters
    formula_str = "a * b"
    variables = ["a", "b"]
    series_vars = ["a"]
    df_vars = ["b"]
    mixture_issue = True

    # Test that ValueError is raised with appropriate message
    with pytest.raises(ValueError) as excinfo:
        validator._handle_all_invalid(
            formula_str, variables, series_vars, df_vars, mixture_issue
        )

    # Verify error message
    assert "Operation between Series" in str(
        excinfo.value
    ), f"Expected 'Operation between Series' in error message, got '{excinfo.value!s}'"
    assert "misaligned indices or columns" in str(
        excinfo.value
    ), f"Expected 'misaligned indices or columns' in error message, got '{excinfo.value!s}'"


def test_handle_all_invalid_no_mixture(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _handle_all_invalid method without mixed variable types."""
    validator, data_dict = result_validator

    # Setup test parameters
    formula_str = "a / b"
    variables: list[str] = ["a", "b"]
    series_vars: list[str] = []
    df_vars: list[str] = ["a", "b"]
    mixture_issue = False

    # Test that ValueError is raised with appropriate message
    with pytest.raises(ValueError) as excinfo:
        validator._handle_all_invalid(
            formula_str, variables, series_vars, df_vars, mixture_issue
        )

    # Verify error message
    assert "division by zero" in str(
        excinfo.value
    ), f"Expected 'division by zero' in error message, got '{excinfo.value!s}'"
    assert "completely misaligned data" in str(
        excinfo.value
    ), f"Expected 'completely misaligned data' in error message, got '{excinfo.value!s}'"


def test_create_warning_message(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test _create_warning_message method for partially invalid results."""
    validator, data_dict = result_validator

    # Setup test parameters
    formula_str = "a / b"
    result = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, 6.0]})
    variables: list[str] = ["a", "b"]
    series_vars: list[str] = []
    df_vars: list[str] = ["a", "b"]
    mixture_issue = False
    has_nan = True
    has_inf = True
    invalid_count = 2

    # Create warning message
    warning_msg = validator._create_warning_message(
        formula_str,
        result,
        variables,
        series_vars,
        df_vars,
        mixture_issue,
        has_nan,
        has_inf,
        invalid_count,
    )

    # Verify warning message contains expected elements
    assert (
        "Warning: Formula 'a / b'" in warning_msg
    ), f"Expected 'Warning: Formula 'a / b'' in message, got '{warning_msg}'"
    assert (
        "Some values are infinite due to division by zero" in warning_msg
    ), f"Expected 'Some values are infinite due to division by zero' in message, got '{warning_msg}'"
    assert (
        "Some values are NaN" in warning_msg
    ), f"Expected 'Some values are NaN' in message, got '{warning_msg}'"


def test_validate_no_invalid(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test validate method with no invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with no invalid values
    test_df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

    # Call validate
    result, invalid_count = validator.validate(test_df, "a + b", data_dict)

    # Verify result is unchanged and invalid count is 0
    pd.testing.assert_frame_equal(result, test_df)
    assert invalid_count == 0, f"Expected invalid count to be 0, got {invalid_count}"


def test_validate_some_invalid_no_fill(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test validate method with some invalid values, no filling."""
    validator, data_dict = result_validator

    # Create a test dataframe with some invalid values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, 6.0]})

    # Create a temporary formula for testing that uses existing variables
    formula = "normal_df + small_df"

    # Use monkeypatch pattern for mocking methods
    from unittest.mock import patch

    # Catch warnings
    with warnings.catch_warnings(record=True) as w, patch.object(
        validator,
        "_check_variable_mixture",
        return_value=(["normal_df", "small_df"], [], ["normal_df", "small_df"], False),
    ):

        # Call validate
        result, invalid_count = validator.validate(test_df, formula, data_dict)

        # Verify warning was issued
        assert len(w) > 0, "Expected a warning, but none was issued"
        assert "Warning: Formula" in str(
            w[0].message
        ), f"Expected 'Warning: Formula' in warning, got '{w[0].message!s}'"

    # Verify result is unchanged and invalid count is correct
    pd.testing.assert_frame_equal(result, test_df)
    assert invalid_count == 2, f"Expected invalid count to be 2, got {invalid_count}"


def test_validate_some_invalid_with_fill(
    result_validator_fill_invalid: tuple[
        _ResultValidator, dict[str, pd.DataFrame | pd.Series]
    ],
) -> None:
    """Test validate method with some invalid values, with filling."""
    validator, data_dict = result_validator_fill_invalid

    # Create a test dataframe with some invalid values
    test_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.inf, 5.0, 6.0]})

    # Expected result after filling
    expected = pd.DataFrame({"A": [1.0, 0.0, 3.0], "B": [0.0, 5.0, 6.0]})

    # Call validate
    result, invalid_count = validator.validate(test_df, "a / b", data_dict)

    # Verify result has invalid values filled with zeros
    pd.testing.assert_frame_equal(result, expected)
    assert invalid_count == 2, f"Expected invalid count to be 2, got {invalid_count}"


def test_validate_all_invalid(
    result_validator: tuple[_ResultValidator, dict[str, pd.DataFrame | pd.Series]],
) -> None:
    """Test validate method with all invalid values."""
    validator, data_dict = result_validator

    # Create a test dataframe with all invalid values
    test_df = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [np.inf, np.inf, np.inf]}
    )

    # Use a formula with variables that exist in the data dict
    formula = "normal_df + small_df"

    # Use monkeypatch pattern with unittest.mock for mocking methods
    from unittest.mock import patch

    # Create a mock function for _handle_all_invalid that raises the expected error
    def mock_handle_all_invalid(*args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "Operation using variables resulted in all invalid values. This suggests a fundamental problem"
        )

    # Use patch to mock both methods
    with patch.object(
        validator,
        "_check_variable_mixture",
        return_value=(["normal_df", "small_df"], [], ["normal_df", "small_df"], False),
    ), patch.object(
        validator, "_handle_all_invalid", side_effect=mock_handle_all_invalid
    ):

        # Test that ValueError is raised
        with pytest.raises(ValueError) as excinfo:
            validator.validate(test_df, formula, data_dict)

        # Verify error message
        assert "resulted in all invalid values" in str(
            excinfo.value
        ), f"Expected 'resulted in all invalid values' in error message, got '{excinfo.value!s}'"


def test_validate_all_invalid_with_fill(
    result_validator_fill_invalid: tuple[
        _ResultValidator, dict[str, pd.DataFrame | pd.Series]
    ],
) -> None:
    """Test validate method with all invalid values, with filling."""
    validator, data_dict = result_validator_fill_invalid

    # Create a test dataframe with all invalid values
    test_df = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan], "B": [np.inf, np.inf, np.inf]}
    )

    # Expected result after filling
    expected = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})

    # Call validate
    result, invalid_count = validator.validate(test_df, "a / 0", data_dict)

    # Verify result has all invalid values filled with zeros
    pd.testing.assert_frame_equal(result, expected)
    assert invalid_count == 6, f"Expected invalid count to be 6, got {invalid_count}"
