"""
Pytest module for testing the FormulaEvaluator class.
"""

import pytest
import pandas as pd
import numpy as np

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


@pytest.fixture
def formula_evaluator() -> tuple(FormulaEvaluator, dict[str, pd.DataFrame]):
    """Pytest fixture for the FormulaEvaluator with test data."""
    data_dict = create_test_data()
    evaluator = FormulaEvaluator(data_dict, adp_enabled=False, verbose=True, fill_invalid=True)
    return evaluator, data_dict


def test_simple_division_subtraction(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test simple division with subtraction."""
    evaluator, data_dict = formula_evaluator
    a, b, c = data_dict["a"], data_dict["b"], data_dict["c"]
    
    formula = "(a - b) / c"
    result = evaluator.evaluate_formula(formula)
    expected = (a - b) / c
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_complex_nested_division(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test complex nested division."""
    evaluator, data_dict = formula_evaluator
    a, b, c, d = data_dict["a"], data_dict["b"], data_dict["c"], data_dict["d"]
    
    formula = "(a+b)/(c/d) + b"
    result = evaluator.evaluate_formula(formula)
    expected = ((a+b)/(c/d)) + b
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_exponentiation_multiplication(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test exponentiation with multiplication."""
    evaluator, data_dict = formula_evaluator
    a, c = data_dict["a"], data_dict["c"]
    
    formula = "(a ** 2.0) * (a ** c)"
    result = evaluator.evaluate_formula(formula)
    expected = (a ** 2.0) * (a ** c)
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_exponentiation_subtraction(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test exponentiation with subtraction."""
    evaluator, data_dict = formula_evaluator
    a, b, c = data_dict["a"], data_dict["b"], data_dict["c"]
    
    formula = "a**b - c"
    result = evaluator.evaluate_formula(formula)
    expected = a**b - c
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_zero_division(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test division with potential zeros."""
    evaluator, data_dict = formula_evaluator
    a, e = data_dict["a"], data_dict["e"]
    
    formula = "a/e"
    result = evaluator.evaluate_formula(formula)
    expected = a / e
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_add_dataframes(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test adding two dataframes."""
    evaluator, data_dict = formula_evaluator
    a, f = data_dict["a"], data_dict["f"]
    
    formula = "a + f"
    result = evaluator.evaluate_formula(formula)
    expected = a + f
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_division_by_diagonal_matrix(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test division by diagonal matrix."""
    evaluator, data_dict = formula_evaluator
    a, g = data_dict["a"], data_dict["g"]
    
    formula = "a / g"
    result = evaluator.evaluate_formula(formula)
    expected = a / g
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_division_by_near_diagonal_matrix(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test division by near-diagonal matrix."""
    evaluator, data_dict = formula_evaluator
    b, h = data_dict["b"], data_dict["h"]
    
    formula = "b / h"
    result = evaluator.evaluate_formula(formula)
    expected = b / h
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_matrix_with_nan_values(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test matrix with NaN values in computation."""
    evaluator, data_dict = formula_evaluator
    a, i = data_dict["a"], data_dict["i"]
    
    formula = "a * i"
    result = evaluator.evaluate_formula(formula)
    expected = a * i
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_division_by_sparse_matrix(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test division by sparse matrix (potential zeros)."""
    evaluator, data_dict = formula_evaluator
    c, j = data_dict["c"], data_dict["j"]
    
    formula = "c / j"
    result = evaluator.evaluate_formula(formula)
    expected = c / j
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_complex_operation_with_diagonal_matrix(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test complex operation with diagonal matrix."""
    evaluator, data_dict = formula_evaluator
    a, g, h, j = data_dict["a"], data_dict["g"], data_dict["h"], data_dict["j"]
    
    formula = "(a + g) / (h - j)"
    result = evaluator.evaluate_formula(formula)
    expected = (a + g) / (h - j)
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_inverting_diagonal_matrix(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test inverting a diagonal matrix."""
    evaluator, data_dict = formula_evaluator
    g = data_dict["g"]
    
    formula = "1 / g"
    result = evaluator.evaluate_formula(formula)
    expected = 1 / g
    
    if evaluator.fill_invalid:
        expected = expected.replace([np.inf, -np.inf, np.nan], 0)
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )


def test_nan_handling(formula_evaluator: tuple(FormulaEvaluator, dict[str, pd.DataFrame])):
    """Test formula with mixed NaN handling."""
    evaluator, data_dict = formula_evaluator
    a, i = data_dict["a"], data_dict["i"]
    
    formula = "i.fillna(0) * a"
    result = evaluator.evaluate_formula(formula)
    expected = i.fillna(0) * a
    
    pd.testing.assert_frame_equal(
        result.astype(np.float64), 
        expected.astype(np.float64), 
        atol=1e-10
    )