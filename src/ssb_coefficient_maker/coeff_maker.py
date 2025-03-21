"""Formula evaluation and coefficient calculation objects for national accounts.
This module provides classes for evaluating mathematical formulas that operate on pandas
objects (DataFrames and Series) and for calculating input-output coefficients used in
the hovedbok (main book) system in national accounts.

The module consists of two primary classes:
- FormulaEvaluator: Parses and evaluates mathematical expressions (like 'a/b' or 'a+b')
  where the variables represent pandas DataFrames or Series.
- CoefficientCalculator: Uses FormulaEvaluator to compute coefficient matrices based on
  formula definitions in a mapping table.

The coefficients are used as a basis for forming the input-output model in the quarterly national accounts.

The formula evaluation supports basic operations including addition, multiplication, division,
and powers, with special handling for pandas alignment issues when performing operations
between different data types (DataFrame-DataFrame, DataFrame-Series, etc.).

Author: Benedikt Goodman
Helper: Claude Sonnet 3.7

"""

from __future__ import annotations

from _resultvalidator import _ResultValidator

import warnings
from typing import Annotated
from typing import Any

import mpmath
import numpy as np
import pandas as pd
import sympy as sp
from pydantic import ConfigDict
from pydantic import Field
from pydantic import validate_call

class FormulaEvaluator:
    """Class for evaluating mathematical formulas with pandas objects.

    This class provides methods for parsing and evaluating mathematical
    expressions involving pandas DataFrames and Series. It supports arbitrary
    decimal precision using mpmath for high-precision numerical operations.

    Attributes:
        data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary mapping variable
            names to pandas objects used in formula evaluation.
        adp_enabled (bool): Whether arbitrary decimal precision is enabled.
        decimal_precision (bool): Number of decimal digits for precision when arbitrary precision
            is enabled. Must be positive. Defaults to 35 (roughly equivalent to 128-bit).
        fill_invalid (bool): Whether to replace Inf and NaN values with zeros after computation.
        verbose (bool): Whether to print verbose information during evaluation.
        validator (ResultValidator): Validates and potentially transforms results.

    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame | pd.Series],
        adp_enabled: bool = False,
        decimal_precision: Annotated[int, Field(gt=0)] = 35,
        fill_invalid: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the FormulaEvaluator with data and precision settings.

        Args:
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary mapping variable names
                to pandas objects (DataFrames or Series).
            adp_enabled (bool): Whether to enable arbitrary decimal precision
                using mpmath. If True, all numeric values will be converted to mpmath.mpf.
                Defaults to False.
            decimal_precision (int): Number of decimal digits for precision when arbitrary precision
                is enabled. Must be positive. Defaults to 35 (roughly equivalent to 128-bit).
            fill_invalid (bool): Whether to replace Inf and NaN values with zeros in the computation
                results. Useful when handling division by zero in diagonal matrices. Defaults to False.
            verbose (bool): Whether to print verbose information during evaluation. Defaults to False.

        Raises:
            AttributeError: If adp_enabled is True but decimal_precision is not a positive integer.

        """
        self.adp_enabled = adp_enabled
        self.fill_invalid = fill_invalid
        self.verbose = verbose
        # Set up the validator - defaults to ResultValidator
        self.validator = _ResultValidator(
            fill_invalid=fill_invalid, verbose=verbose, adp_enabled=adp_enabled
        )

        # If arbitrary precision is enabled, use mpf
        if adp_enabled:
            if not decimal_precision:
                raise AttributeError(
                    "decimal_precision must be a positive integer when adp_enabled is True."
                )
            mpmath.mp.dps = decimal_precision # Digits of precision
            result = {}
            for key, val in data_dict.copy.items():
                if isinstance(val, pd.DataFrame):
                    result[key] = val.map(lambda x: mpmath.mpf(x))
                else: # Series
                    result[key] = val.apply(lambda x: mpmath.mpf(x))
            data_dict = result

        # Else use np.float64 datatype
        else:
            self.data_dict = {
                key: val.astype(np.float64, copy=True, errors="raise")
                for key, val in data_dict.copy().items()
            }

        if self.verbose:
            print(f"FormulaEvaluator initialized with {len(data_dict)} variables")
            print(
                f"Settings: precision_mode={'mpmath' if adp_enabled else 'numpy'}, "
                f"fill_invalid={fill_invalid}"
            ) 

    def _perform_evaluation(
        self, formula_str: str | sp.Expr
    ) -> pd.DataFrame | pd.Series:
        """Perform the actual formula evaluation.

        This method evaluates mathematical formulas using pandas objects. It handles
        special cases like Series-DataFrame broadcasting by converting Series to
        transposed DataFrames to ensure correct alignment during operations.

        Args:
            formula_str (Union[str, sp.Expr]): Formula string or sympy expression to evaluate.

        Returns:
            Union[pd.DataFrame, pd.Series]: The result of evaluating the formula.

        Raises:
            ValueError: If power operations ('**') are found in the formula when arbitrary
                precision is enabled, which is not supported by pandas eval.
            KeyError: If the formula references a variable not present in the data dictionary.
            SyntaxError: If the formula contains invalid syntax for pandas eval.
            TypeError: If an operation in the formula is not supported between the given types.
            ZeroDivisionError: If division by zero occurs when arbitrary decimal precision is enabled.
        """
        # Check for power operations in ADP mode
        if self.adp_enabled and "**" in formula_str:
            raise ValueError(
                f"Power operation '**' found in formula '{formula_str}'. "
                "Pandas eval doesn't support power operations between DataFrames and scalars. "
                "Consider using DataFrame.pow() method instead or pre-compute this operation."
            )

        try:
            # Create a copy of the data dictionary with Series transposed for evaluation
            eval_dict = {}
            series_indices = []

            # Transform Series objects and collect their indices
            for key, val in self.data_dict.items():
                if isinstance(val, pd.Series):
                    series_indices.append(val.index)
                    # Convert Series to transposed NumPy array for correct broadcasting
                    eval_dict[key] = val.T.to_numpy()
                else:
                    eval_dict[key] = val

            # Evaluates formula expression and calculates result
            result = pd.eval(formula_str, local_dict=eval_dict)

            # Apply common index to Series if all Series have the same index
            if series_indices and all(
                index.equals(series_indices[0]) for index in series_indices
            ):
                common_index = series_indices[0]
                if isinstance(result, pd.DataFrame):
                    result.index = common_index
                elif isinstance(result, pd.Series):
                    result.index = common_index

        except KeyError as e:
            # Variable not found in data dictionary
            raise KeyError(
                f"Variable '{e.args[0]}' not found in data dictionary"
            ) from e
        except SyntaxError as e:
            # Invalid syntax in formula
            raise SyntaxError(f"Invalid syntax in formula '{formula_str}': {e}") from e
        except TypeError as e:
            # Type error in operation
            raise TypeError(f"Type error in formula '{formula_str}': {e}") from e
        except ZeroDivisionError as e:
            if self.adp_enabled:
                raise ZeroDivisionError(
                    f"Zero division error in formula '{formula_str}': "
                    "Division by zero is not supported when using arbitrary decimal precision (adp_enabled=True). "
                    "Options: 1) Set adp_enabled=False to use numpy's handling of infinity, "
                    "2) Modify your input data to avoid division by zero."
                ) from e

        return result

    def evaluate_formula(self, formula_str: str | sp.Expr) -> pd.DataFrame | pd.Series:
        """Evaluate a formula string using pandas objects in data_dict.

        Computes the result of the formula using pandas eval() with the
        variables defined in data_dict.

        Args:
            formula_str (str): Formula string (e.g., "var1 / var2") to evaluate.

        Returns:
            Union[pd.DataFrame, pd.Series]: Result of the evaluation.
        """
        if self.verbose:
            print(f"Evaluating formula: {formula_str}")

            # Check for division operations
            if "/" in str(formula_str):
                print(
                    "Note: Formula contains division. Invalid values will "
                    + (
                        "be replaced with zeros."
                        if self.fill_invalid
                        else "trigger warnings or errors."
                    )
                )

        # Evaluate the formula
        result = self._perform_evaluation(formula_str)

        # Validate and potentially transform the result
        if self.validator:
            result, _ = self.validator.validate(
                result=result, formula_str=formula_str, data_dict=self.data_dict
            )

        if self.verbose:
            print(
                f"Formula evaluation complete. Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}"
            )

        return result

class CoefficientCalculator(FormulaEvaluator):
    """Class for calculating coefficients based on formulas and input data.

    This class evaluates a set of coefficient formulas using data from pandas objects
    and produces a dictionary of calculated coefficients. It leverages the FormulaEvaluator
    for parsing and evaluating mathematical expressions, supporting arbitrary decimal
    precision for high-precision numerical operations.

    The CoefficientCalculator is designed for batch processing of multiple formula-based
    calculations, and is made to make the process of doing arithmetic operations on vectors
    and matrices as streamlined as possible.It reads formula definitions from a DataFrame and
    applies them toinput data stored in pandas structures.

    Key features:
    - Processes multiple coefficient calculations in a single operation
    - Supports configurable column names in the formula definition table
    - Handles arbitrary precision calculations for high accuracy requirements
    - Provides error handling for missing variables and invalid formulas
    - Can automatically replace invalid calculation results (NaN, Inf, pd.NA) with zeros
    - Preserves input data integrity by working with a copy of the input dictionary

    Calculation workflow:
    1. Initialize with input data and a formula definition table
    2. Validate the formula table has the required columns
    3. For each formula definition:
       a. Parse the formula expression
       b. Check if all required variables exist
       c. Evaluate the formula using the FormulaEvaluator
       d. Store the result in the output dictionary
    4. Return a dictionary of all calculated coefficients
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        coefficient_map: pd.DataFrame,
        result_name_col: str,
        formula_name_col: str,
    ) -> None:
        """Initialize the CoefficientCalculator with data and precision settings.

        Args:
            coefficient_map (pd.DataFrame): DataFrame with coefficient definitions including
                columns for result names and formula strings as specified by result_name_col
                and formula_name_col parameters.
            result_name_col (str): Name of the column in coefficient_map containing result names.
            formula_name_col (str): Name of the column in coefficient_map containing formulas.
        """
        #  check for mandatory columns
        missing = set(mandatory_cols).difference(coefficient_map.columns)
        if missing:
            raise KeyError(f"{missing} not found among coefficient map columns. ")

        # store data
        self.coefficient_map = coefficient_map
        self.result_name_col = result_name_col
        self.formula_name_col = formula_name_col

    def compute_coefficients(self) -> dict[str, Any]:
        """Compute coefficient matrices based on formulas in the coefficient map.

        Processes each row in the coefficient_map, evaluating the formula and
        storing the result in a dictionary. Skips entries with missing formulas
        or missing variables, and handles exceptions gracefully.

        If invalid values (NaN, Inf) occur during calculation, they are either preserved
        or replaced with zeros depending on the fill_invalid setting. This is particularly
        important when dividing by matrices that may contain zeros.

        Returns:
            Dict[str, Union[pd.DataFrame, pd.Series]]: Dictionary mapping coefficient
                names to their computed values (either DataFrames or Series).

        Note:
            This method logs information about skipped coefficients and errors
            using print statements. For production use, consider implementing
            proper logging.
        """
        result = {}

        for _, row in self.coefficient_map.iterrows():
            name = row[self.result_name_col]
            formula = row[self.formula_name_col]

            if pd.isna(formula) or formula.strip() == "":
                print(f"Skipping coefficient {name}: No formula provided")
                continue

            # Parse the formula into a sympy expression
            if self.verbose:
                print(f"Parsing formula: {formula}")

            # Create local dictionary of symbols for sympy
            symbols = {name: sp.Symbol(name) for name in self.data_dict.keys()}
            expr = sp.sympify(formula, locals=symbols)

            if self.verbose:
                print(f"Parsed formula: {parsed}")

            # Check if all variables in the formula exist in the data dictionary
            variables = [str(symbol) for symbol in expr.free_symbols]
            if self.verbose:
                print(f"Variables in expression: {variables}")
            
            missing_vars = [var for var in variables if var not in self.data_dict]

            if missing_vars:
                print(f"Skipping coefficient {name}: Missing variables {missing_vars}")
                continue

            # Evaluate the expression
            computed_value = self.evaluate_formula(expr)

            # Store the coefficient
            result[name] = computed_value

            print(f"Successfully computed coefficient: {name}")

        return result