"""
Pytest module for testing the CoefficientCalculator class.
"""

import pytest
import pandas as pd
import numpy as np

from ssb_coefficient_maker.coeff_maker import CoefficientCalculator

# Fixed seed for reproducible tests
SEED = 42


def create_test_data() -> dict[str, pd.DataFrame]:
    """Create test data for coefficient calculator tests."""
    rng = np.random.default_rng(seed=SEED)
    
    # Regular test matrices
    data_dict = {
        "a": pd.DataFrame(rng.integers(low=1, high=10, size=(3, 3))).astype(float),
        "b": pd.DataFrame(rng.integers(low=1, high=5, size=(3, 3))).astype(float),
        "c": pd.DataFrame(rng.integers(low=1, high=3, size=(3, 3))).astype(float),
        "d": pd.DataFrame(rng.integers(low=2, high=6, size=(3, 3))).astype(float),
        "e": pd.DataFrame(rng.integers(low=0, high=1, size=(3, 3))).astype(float),
        "f": pd.DataFrame(np.tile(rng.integers(low=0, high=5, size=(3)), (3, 1))).astype(float),
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
    
    return data_dict


def create_coefficient_map() -> pd.DataFrame:
    """Create a coefficient map for testing."""
    return pd.DataFrame({
        'navn': [
            'coef_simple', 
            'coef_complex', 
            'coef_nan', 
            'coef_diagonal',
            'coef_invalid_var',
            'coef_empty',
            'coef_division_by_sparse'
        ],
        'formel': [
            'a + b',                   # Simple addition
            '(a + b) / (c * d)',       # Complex nested operations
            'a * i',                   # Contains NaN values
            'a / g',                   # Division by diagonal matrix
            'nonexistent * a',         # Invalid variable
            '',                        # Empty formula
            'b / j'                    # Division by sparse (has zeros)
        ]
    })


@pytest.fixture
def coefficient_calculator() -> tuple(CoefficientCalculator, dict[str, pd.DataFrame | pd.Series], pd.DataFrame):
    """Pytest fixture for CoefficientCalculator with test data."""
    data_dict = create_test_data()
    coef_map = create_coefficient_map()
    calculator = CoefficientCalculator(data_dict, coef_map, adp_enabled=False, fill_invalid=True)
    return calculator, data_dict, coef_map


def test_compute_coefficients(coefficient_calculator: tuple):
    """Test that coefficients are computed correctly."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Expected coefficients that should be computed
    expected_coefficients = [
        'coef_simple',
        'coef_complex',
        'coef_nan',
        'coef_diagonal',
        'coef_division_by_sparse'
    ]
    
    # Check that all expected coefficients were computed
    for coef_name in expected_coefficients:
        assert coef_name in computed_coefficients, f"{coef_name} was not computed"
    
    # Check the total number of computed coefficients
    assert len(computed_coefficients) == len(expected_coefficients), \
        f"Expected {len(expected_coefficients)} coefficients, got {len(computed_coefficients)}"


def test_skipped_coefficients(coefficient_calculator: tuple):
    """Test that invalid coefficients are skipped."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Coefficients that should be skipped
    skipped_coefficients = [
        'coef_invalid_var',
        'coef_empty'
    ]
    
    # Check that all expected coefficients were skipped
    for coef_name in skipped_coefficients:
        assert coef_name not in computed_coefficients, f"{coef_name} was computed but should be skipped"


def test_coefficient_simple_addition(coefficient_calculator: tuple):
    """Test simple addition coefficient."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Get the computed coefficient
    coef_name = 'coef_simple'
    coef_value = computed_coefficients[coef_name]
    
    # Calculate expected result directly
    a = data_dict["a"]
    b = data_dict["b"]
    expected = a + b
    
    # Compare results
    pd.testing.assert_frame_equal(
        coef_value.astype(np.float64),
        expected.astype(np.float64),
        atol=1e-10
    )


def test_coefficient_complex_operations(coefficient_calculator: tuple):
    """Test complex nested operations coefficient."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Get the computed coefficient
    coef_name = 'coef_complex'
    coef_value = computed_coefficients[coef_name]
    
    # Calculate expected result directly
    a = data_dict["a"]
    b = data_dict["b"]
    c = data_dict["c"]
    d = data_dict["d"]
    expected = (a + b) / (c * d)
    
    # Handle NaN and Infinity values if fill_invalid is True
    if calculator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    # Compare results
    pd.testing.assert_frame_equal(
        coef_value.astype(np.float64),
        expected.astype(np.float64),
        atol=1e-10
    )


def test_coefficient_nan_handling(coefficient_calculator: tuple):
    """Test NaN handling in coefficients."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Get the computed coefficient
    coef_name = 'coef_nan'
    coef_value = computed_coefficients[coef_name]
    
    # Calculate expected result directly
    a = data_dict["a"]
    i = data_dict["i"]
    expected = a * i
    
    # Handle NaN and Infinity values if fill_invalid is True
    if calculator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    # Compare results
    pd.testing.assert_frame_equal(
        coef_value.astype(np.float64),
        expected.astype(np.float64),
        atol=1e-10
    )


def test_coefficient_division_by_diagonal(coefficient_calculator: tuple):
    """Test division by diagonal matrix coefficient."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Get the computed coefficient
    coef_name = 'coef_diagonal'
    coef_value = computed_coefficients[coef_name]
    
    # Calculate expected result directly
    a = data_dict["a"]
    g = data_dict["g"]
    expected = a / g
    
    # Handle NaN and Infinity values if fill_invalid is True
    if calculator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    # Compare results
    pd.testing.assert_frame_equal(
        coef_value.astype(np.float64),
        expected.astype(np.float64),
        atol=1e-10
    )


def test_coefficient_division_by_sparse(coefficient_calculator: tuple):
    """Test division by sparse matrix coefficient."""
    calculator, data_dict, coef_map = coefficient_calculator
    
    # Compute coefficients
    computed_coefficients = calculator.compute_coefficients()
    
    # Get the computed coefficient
    coef_name = 'coef_division_by_sparse'
    coef_value = computed_coefficients[coef_name]
    
    # Calculate expected result directly
    b = data_dict["b"]
    j = data_dict["j"]
    expected = b / j
    
    # Handle NaN and Infinity values if fill_invalid is True
    if calculator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    # Compare results
    pd.testing.assert_frame_equal(
        coef_value.astype(np.float64),
        expected.astype(np.float64),
        atol=1e-10
    )