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


class _ResultValidator:
    """Validator for formula evaluation results.

    This class handles detection and validation of invalid values (NaN and Inf)
    in formula evaluation results, providing appropriate warnings and errors.

    Attributes:
        fill_invalid (bool): Whether to replace invalid values with zeros.
        verbose (bool): Whether to print verbose information during validation.
        adp_enabled (bool): Whether arbitrary decimal precision is enabled.
    """

    def __init__(
        self,
        fill_invalid: bool = False,
        verbose: bool = False,
        adp_enabled: bool = False,
    ) -> None:
        """Initialize the result validator.

        Args:
            fill_invalid (bool): Whether to replace invalid values with zeros.
            verbose (bool): Whether to print verbose information during validation.
            adp_enabled (bool): Whether arbitrary decimal precision is enabled.
        """
        self.fill_invalid = fill_invalid
        self.verbose = verbose
        self.adp_enabled = adp_enabled

    def validate(
        self,
        result: pd.DataFrame | pd.Series,
        formula_str: str | sp.Expr,
        data_dict: dict[str, pd.DataFrame | pd.Series],
    ) -> tuple[pd.DataFrame | pd.Series, int]:
        """Validate the result of formula evaluation.

        Detects invalid values (NaN and Inf) in computation results and provides
        appropriate warnings or errors depending on the severity.

        Args:
            result (Union[pd.DataFrame, pd.Series]): The result of the formula evaluation.
            formula_str (str): The formula string used for computation.
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary of variable names to pandas objects.

        Warns:
            UserWarning: If some (but not all) values in the result are invalid,
                suggesting potential issues with the computation and fill_invalid is False.

        Returns:
            Tuple[Union[pd.DataFrame, pd.Series], int]: A tuple containing the processed result
                and the number of invalid values found.
        """
        # Check invalid status
        all_invalid, some_invalid, has_nan, has_inf = self._check_invalid_status(result)

        # Log details if verbose
        self._log_invalid_details(result, all_invalid, some_invalid, has_nan, has_inf)

        # Get invalid count if needed
        invalid_count = 0
        if all_invalid or some_invalid:
            invalid_count = self._count_invalid_values(result)

        # If we're filling invalid values, do that and return
        if self.fill_invalid:
            if all_invalid or some_invalid:
                processed_result = self._fill_invalid_values(result)
                if self.verbose and invalid_count > 0:
                    print(
                        f"Replaced {invalid_count} invalid values (NaN/Inf) with zeros"
                    )
                return processed_result, invalid_count
            return result, 0

        # If we have invalid values, continue with validation
        if all_invalid or some_invalid:
            # Get variable information
            variables, series_vars, df_vars, mixture_issue = (
                self._check_variable_mixture(formula_str, data_dict)
            )

            if all_invalid:
                # Handle case where all values are invalid
                self._handle_all_invalid(
                    formula_str, variables, series_vars, df_vars, mixture_issue
                )
            elif some_invalid:
                # Create and issue warning for partial invalid values
                warning_msg = self._create_warning_message(
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
                warnings.warn(warning_msg, UserWarning, stacklevel=0)

        return result, invalid_count

    def _get_invalid_mask(
        self, result: pd.DataFrame | pd.Series
    ) -> pd.DataFrame | pd.Series:
        """Create a mask identifying where all invalid values (NaN and Inf) are in result.

        Args:
            result (Union[pd.DataFrame, pd.Series]): DataFrame or Series to check for invalid values.

        Returns:
            Union[pd.DataFrame, pd.Series]: DataFrame or Series of booleans with True for invalid values.
        """
        if isinstance(result, pd.DataFrame):
            if self.adp_enabled:
                # For mpmath objects
                return result.map(
                    lambda x: getattr(x, "is_nan", False)
                    or getattr(x, "is_infinite", False)
                )
            else:
                # For numpy floats, returns mask with true where either below are the case.
                return pd.DataFrame(
                    data=np.logical_or(result.isna(), np.isinf(result.values)),
                    index=result.index,
                    columns=result.columns,
                )

        elif isinstance(result, pd.Series):
            if self.adp_enabled:
                # For mpmath objects
                return result.apply(
                    lambda x: getattr(x, "is_nan", False)
                    or getattr(x, "is_infinite", False)
                )
            else:
                # For numpy floats, returns mask with true where either below are the case.
                return pd.Series(
                    data=np.logical_or(result.isna(), np.isinf(result.values)),
                    index=result.index,
                )

        else:
            # For other types, return empty mask
            return pd.Series([False])

    def _count_invalid_values(self, result: pd.DataFrame | pd.Series) -> int:
        """Count the number of invalid values (NaN and Inf) in the result.

        Args:
            result (Union[pd.DataFrame, pd.Series]): DataFrame or Series to check for invalid values.

        Returns:
            int: Count of invalid values found.
        """
        invalid_mask = self._get_invalid_mask(result)

        if isinstance(invalid_mask, pd.DataFrame):
            return int(invalid_mask.sum().sum())
        elif isinstance(invalid_mask, pd.Series):
            return int(invalid_mask.sum())
        else:
            return 0

    def _fill_invalid_values(
        self, result: pd.DataFrame | pd.Series
    ) -> pd.DataFrame | pd.Series:
        """Replace Inf and NaN values with zeros.

        Args:
            result (Union[pd.DataFrame, pd.Series]): DataFrame or Series with potential Inf or NaN values.

        Returns:
            Union[pd.DataFrame, pd.Series]: DataFrame or Series with Inf and NaN values replaced by zeros.
        """
        # Get mask of invalid values
        invalid_mask = self._get_invalid_mask(result)

        # Check if we have any invalid values
        if not self._has_invalid_values(invalid_mask):
            return result

        # Handle replacement based on type and ADP setting
        if isinstance(result, pd.DataFrame):
            return self._fill_invalid_dataframe(result, invalid_mask)
        elif isinstance(result, pd.Series):
            return self._fill_invalid_series(result, invalid_mask)

        return result

    def _has_invalid_values(self, invalid_mask: pd.DataFrame | pd.Series) -> bool | Any:
        """Check if there are any invalid values in the data.

        Args:
            invalid_mask (Union[pd.DataFrame, pd.Series]): Mask of invalid values.

        Returns:
            bool: True if there are any invalid values, False otherwise.
        """
        if isinstance(invalid_mask, pd.DataFrame):
            return invalid_mask.any().any()
        elif isinstance(invalid_mask, pd.Series):
            return invalid_mask.any()
        return False

    def _fill_invalid_dataframe(
        self, df: pd.DataFrame, invalid_mask: pd.DataFrame
    ) -> pd.DataFrame:
        """Replace invalid values in a DataFrame with zeros.

        Args:
            df (pd.DataFrame): DataFrame with potential invalid values.
            invalid_mask (pd.DataFrame): Boolean mask of invalid values.

        Returns:
            pd.DataFrame: DataFrame with invalid values replaced by zeros.
        """
        if self.adp_enabled:
            return self._fill_invalid_dataframe_adp(df, invalid_mask)
        return df.replace([np.inf, -np.inf, np.nan, pd.NA], 0)

    def _fill_invalid_dataframe_adp(
        self, df: pd.DataFrame, invalid_mask: pd.DataFrame
    ) -> pd.DataFrame:
        """Replace invalid values in a DataFrame with zeros using ADP precision.

        Args:
            df (pd.DataFrame): DataFrame with potential invalid values.
            invalid_mask (pd.DataFrame): Boolean mask of invalid values.

        Returns:
            pd.DataFrame: DataFrame with invalid values replaced by zeros using mpmath.
        """
        result_copy = df.copy()
        for col in df.columns:
            for idx in df.index:
                if invalid_mask.loc[idx, col]:
                    result_copy.loc[idx, col] = mpmath.mpf("0")
        return result_copy

    def _fill_invalid_series(
        self, series: pd.Series, invalid_mask: pd.Series
    ) -> pd.Series:
        """Replace invalid values in a Series with zeros.

        Args:
            series (pd.Series): Series with potential invalid values.
            invalid_mask (pd.Series): Boolean mask of invalid values.

        Returns:
            pd.Series: Series with invalid values replaced by zeros.
        """
        if self.adp_enabled:
            return self._fill_invalid_series_adp(series, invalid_mask)
        return series.replace([np.inf, -np.inf, np.nan, pd.NA], 0)

    def _fill_invalid_series_adp(
        self, series: pd.Series, invalid_mask: pd.Series
    ) -> pd.Series:
        """Replace invalid values in a Series with zeros using ADP precision.

        Args:
            series (pd.Series): Series with potential invalid values.
            invalid_mask (pd.Series): Boolean mask of invalid values.

        Returns:
            pd.Series: Series with invalid values replaced by zeros using mpmath.
        """
        result_copy = series.copy()
        for idx in series.index:
            if invalid_mask.loc[idx]:
                result_copy.loc[idx] = mpmath.mpf("0")
        return result_copy

    def _check_invalid_status(
        self, result: pd.DataFrame | pd.Series
    ) -> tuple[bool, bool, bool, bool]:
        """Check the status of invalid values in the result.

        Determines whether the result contains all invalid values, some invalid values,
        or specifically NaN or Inf values.

        Args:
            result (Union[pd.DataFrame, pd.Series]): DataFrame or Series to check.

        Returns:
            Tuple[bool, bool, bool, bool]: Tuple containing (all_invalid, some_invalid, has_nan, has_inf)
        """
        # Get invalid mask
        invalid_mask = self._get_invalid_mask(result)

        # Check for all vs some invalid values
        if isinstance(invalid_mask, pd.DataFrame):
            all_invalid = bool(invalid_mask.all().all())
            some_invalid = bool(invalid_mask.any().any()) and not all_invalid

        elif isinstance(invalid_mask, pd.Series):
            all_invalid = bool(invalid_mask.all())
            some_invalid = bool(invalid_mask.any()) and not all_invalid
        else:
            all_invalid = False
            some_invalid = False

        # Check specifically for NaNs and Infs
        if isinstance(result, pd.DataFrame):
            # We need to check the attributes of each number in the series or dataframe in the case of using mpf floats
            if self.adp_enabled:
                has_nan = bool(
                    result.map(lambda x: getattr(x, "is_nan", False)).any().any()
                )
                has_inf = bool(
                    result.map(lambda x: getattr(x, "is_infinite", False)).any().any()
                )
            # if using numpy we just check if there are any inf or nan values
            else:
                has_nan = bool(result.isna().any().any())
                has_inf = bool(np.isinf(result.values).any())

        # Same operation as above, but for series
        elif isinstance(result, pd.Series):
            if self.adp_enabled:
                has_nan = bool(
                    result.apply(lambda x: getattr(x, "is_nan", False)).any()
                )
                has_inf = bool(
                    result.apply(lambda x: getattr(x, "is_infinite", False)).any()
                )
            else:
                has_nan = bool(result.isna().any())
                has_inf = bool(np.isinf(result.values).any())
        else:
            has_nan = False
            has_inf = False

        return all_invalid, some_invalid, has_nan, has_inf

    def _log_invalid_details(
        self,
        result: pd.DataFrame | pd.Series,
        all_invalid: bool,
        some_invalid: bool,
        has_nan: bool,
        has_inf: bool,
    ) -> None:
        """Log details about invalid values if verbose mode is enabled.

        Args:
            result (Union[pd.DataFrame, pd.Series]): The DataFrame or Series being checked.
            all_invalid (bool): Whether all values are invalid.
            some_invalid (bool): Whether some (but not all) values are invalid.
            has_nan (bool): Whether there are NaN values.
            has_inf (bool): Whether there are Inf values.
        """
        if not self.verbose:
            return

        if all_invalid:
            print("WARNING: Result contains all invalid values")
        elif some_invalid:
            invalid_count = self._count_invalid_values(result)
            total = result.size if hasattr(result, "size") else len(result)
            percentage = (invalid_count / total) * 100 if total > 0 else 0
            print(
                f"WARNING: Result contains {invalid_count}/{total} ({percentage:.2f}%) invalid values"
            )

            if has_nan and has_inf:
                print(" - Result contains both NaN and Inf values")
            elif has_nan:
                print(" - Result contains NaN values")
            elif has_inf:
                print(" - Result contains Inf values (division by zero)")

        if self.fill_invalid and (all_invalid or some_invalid):
            print("Invalid values will be replaced with zeros")

    def _parse_formula(
        self, formula_str: str | sp.Expr, data_dict: dict[str, pd.DataFrame | pd.Series]
    ) -> sp.Expr:
        """Parse a formula string into a sympy expression.

        Args:
            formula_str (str): Formula string to parse.
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary of variable names to pandas objects.

        Returns:
            sp.Expr: Parsed sympy expression.
        """
        # Create local dictionary of symbols for sympy
        symbols = {name: sp.Symbol(name) for name in data_dict.keys()}

        # Parse the formula
        return sp.sympify(formula_str, locals=symbols)

    def _extract_variables(self, expr: sp.Expr) -> list[str]:
        """Extract variable names from a sympy expression.

        Args:
            expr (sp.Expr): Sympy expression to analyze.

        Returns:
            List[str]: List of variable names used in the expression.
        """
        return [str(symbol) for symbol in expr.free_symbols]

    def _check_variable_mixture(
        self, formula_str: str | sp.Expr, data_dict: dict[str, pd.DataFrame | pd.Series]
    ) -> tuple[list[str], list[str], list[str], bool]:
        """Check if the formula mixes Series and DataFrame variables.

        Args:
            formula_str (Union[str, sp.Expr]): The formula string to check.
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary of variable names to pandas objects.

        Returns:
            Tuple[List[str], List[str], List[str], bool]: Tuple containing
                (variables, series_vars, df_vars, mixture_issue)
        """
        # Parse the formula to get variables
        expr = self._parse_formula(formula_str, data_dict)
        variables = self._extract_variables(expr)

        # Check if we mixed Series and DataFrames
        series_vars = [
            var for var in variables if isinstance(data_dict[var], pd.Series)
        ]
        df_vars = [var for var in variables if isinstance(data_dict[var], pd.DataFrame)]

        mixture_issue = bool(series_vars and df_vars)

        return variables, series_vars, df_vars, mixture_issue

    def _handle_all_invalid(
        self,
        formula_str: str | sp.Expr,
        variables: list[str],
        series_vars: list[str],
        df_vars: list[str],
        mixture_issue: bool,
    ) -> None:
        """Handle the case where all values in the result are invalid.

        Args:
            formula_str (str): The formula string that was evaluated.
            variables (List[str]): List of variables used in the formula.
            series_vars (List[str]): List of Series variables used.
            df_vars (List[str]): List of DataFrame variables used.
            mixture_issue (bool): Whether there's a mixture of Series and DataFrame variables.

        Raises:
            ValueError: With an appropriate error message.
        """
        if mixture_issue:
            pass
            raise ValueError(
                f"Operation between Series {series_vars} and "
                f"DataFrames {df_vars} in formula '{formula_str}' resulted in all invalid values. "
                f"This is likely due to misaligned indices or columns. "
                f"Consider aligning your data or restructuring your formula."
            )
        else:
            raise ValueError(
                f"Operation using variables {variables} in formula '{formula_str}' "
                f"resulted in all invalid values. This suggests a fundamental problem "
                f"with the computation, such as division by zero, invalid operations, "
                f"or completely misaligned data."
            )

    def _create_warning_message(
        self,
        formula_str: str | sp.Expr,
        result: pd.DataFrame | pd.Series,
        variables: list[str],
        series_vars: list[str],
        df_vars: list[str],
        mixture_issue: bool,
        has_nan: bool,
        has_inf: bool,
        invalid_count: int,
    ) -> str:
        """Create an appropriate warning message for partially invalid results.

        Args:
            formula_str (str): The formula string that was evaluated.
            result (Union[pd.DataFrame, pd.Series]): The result containing some invalid values.
            variables (List[str]): List of variables used in the formula.
            series_vars (List[str]): List of Series variables used.
            df_vars (List[str]): List of DataFrame variables used.
            mixture_issue (bool): Whether there's a mixture of Series and DataFrame variables.
            has_nan (bool): Whether there are NaN values.
            has_inf (bool): Whether there are Inf values.
            invalid_count (int): Number of invalid values found.

        Returns:
            str: The formatted warning message.
        """
        # Calculate the percentage of invalid values
        if isinstance(result, pd.DataFrame):
            total_cells = result.size
        else:  # Series
            total_cells = len(result)

        invalid_percent = (invalid_count / total_cells) * 100 if total_cells > 0 else 0

        warning_msg = (
            f"Warning: Formula '{formula_str}' using variables {variables} "
            f"produced a result with {invalid_percent:.1f}% invalid values. "
        )

        if has_inf:
            warning_msg += "Some values are infinite due to division by zero. "

        if has_nan:
            if mixture_issue:
                warning_msg += (
                    f"Some values are NaN, possibly due to operations between Series {series_vars} and "
                    f"DataFrames {df_vars} with partially misaligned indices or columns. "
                )
            else:
                warning_msg += (
                    "Some values are NaN, which could indicate partial data misalignment, missing values "
                    "in the original data, or operations that produced undefined results for some elements. "
                )

        warning_msg += (
            "Consider checking your input data and formula for potential issues.\n"
        )

        return warning_msg


class FormulaEvaluator:
    """Class for evaluating mathematical formulas with pandas objects.

    This class provides methods for parsing and evaluating mathematical
    expressions involving pandas DataFrames and Series. It supports arbitrary
    decimal precision using mpmath for high-precision numerical operations.

    Attributes:
        data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary mapping variable
            names to pandas objects used in formula evaluation.
        adp_enabled (bool): Whether arbitrary decimal precision is enabled.
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
            self.data_dict = self._cast_dtypes_to_mpfloat(
                data_dict.copy(), decimal_precision
            )
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

    @staticmethod
    def _cast_dtypes_to_mpfloat(
        data_dict: dict[str, pd.DataFrame | pd.Series], decimal_precision: int
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """Convert all numeric data to mpmath float types with specified precision.

        Args:
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary of pandas objects to convert.
            decimal_precision (int): Number of decimal digits for precision.

        Returns:
            Dict[str, Union[pd.DataFrame, pd.Series]]: Dictionary with the same structure
                but with all numeric values converted to mpmath.mpf type.
        """
        # Set precision
        mpmath.mp.dps = decimal_precision  # Digits of precision

        # Convert to mpmath float
        result = {}
        for key, val in data_dict.items():
            if isinstance(val, pd.DataFrame):
                result[key] = val.map(lambda x: mpmath.mpf(x))
            else:  # Series
                result[key] = val.apply(lambda x: mpmath.mpf(x))
        return result

    def parse_formula(self, formula: str) -> sp.Expr:
        """Parse a formula string into a sympy expression.

        Converts a string formula into a symbolic math expression using sympy,
        with variables corresponding to keys in the data_dict.

        Args:
            formula (str): Formula string (e.g., "wspluss/xsvek" or "var1 + var2 * 3").

        Returns:
            sp.Expr: Parsed sympy expression that can be analyzed or evaluated.

        """
        if self.verbose:
            print(f"Parsing formula: {formula}")

        # Create local dictionary of symbols for sympy
        symbols = {name: sp.Symbol(name) for name in self.data_dict.keys()}

        # Parse the formula
        parsed = sp.sympify(formula, locals=symbols)

        if self.verbose:
            print(f"Parsed expression: {parsed}")

        return parsed

    def extract_variables(self, expr: sp.Expr) -> list[str]:
        """Extract variable names from a sympy expression.

        Identifies all variables used in the expression that should be present
        in the data_dict for evaluation.

        Args:
            expr (sp.Expr): Sympy expression to analyze.

        Returns:
            List[str]: List of variable names used in the expression.

        """
        variables = [str(symbol) for symbol in expr.free_symbols]

        if self.verbose:
            print(f"Variables in expression: {variables}")

        return variables

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


class CoefficientCalculator:
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

    Attributes:
        data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary mapping variable
            names to pandas objects used in formula evaluation.
        coefficient_map (pd.DataFrame): DataFrame containing coefficient calculation definitions.
        result_name_col (str): Name of the column in coefficient_map containing result names.
        formula_name_col (str): Name of the column in coefficient_map containing formulas.
        fill_invalid (bool): Whether to replace Inf and NaN values with zeros after computation.
        adp_enabled (bool): Whether arbitrary decimal precision is enabled.
        evaluator (FormulaEvaluator): Formula evaluator instance used for parsing and computing.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame | pd.Series],
        coefficient_map: pd.DataFrame,
        result_name_col: str,
        formula_name_col: str,
        adp_enabled: bool = True,
        decimal_precision: Annotated[int, Field(gt=0)] = 35,
        fill_invalid: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the CoefficientCalculator with data and precision settings.

        Args:
            data_dict (Dict[str, Union[pd.DataFrame, pd.Series]]): Dictionary mapping variable names
                to pandas objects (DataFrames or Series).
            coefficient_map (pd.DataFrame): DataFrame with coefficient definitions including
                columns for result names and formula strings as specified by result_name_col
                and formula_name_col parameters.
            result_name_col (str): Name of the column in coefficient_map containing result names.
            formula_name_col (str): Name of the column in coefficient_map containing formulas.
            adp_enabled (bool): Whether to enable arbitrary decimal precision
                using mpmath. If True, all numeric values will be converted to mpmath.mpf.
                Defaults to True.
            decimal_precision (int): Number of decimal digits for precision when arbitrary precision
                is enabled. Must be positive. Defaults to 35 (roughly equivalent to 128-bit).
            fill_invalid (bool): Whether to replace Inf and NaN values with zeros in the computation
                results. Useful when handling division by zero in diagonal matrices. Defaults to False.
            verbose (bool): Whether to print verbose information during evaluation. Defaults to False.
        """
        #  check for mandatory columns
        self._validate_coefficient_map_headers(
            coefficient_map, [result_name_col, formula_name_col]
        )

        # store data
        self.data_dict = data_dict
        self.coefficient_map = coefficient_map
        self.result_name_col = result_name_col
        self.formula_name_col = formula_name_col

        self.fill_invalid = fill_invalid
        self.adp_enabled = adp_enabled

        self.evaluator = FormulaEvaluator(
            data_dict,
            adp_enabled=self.adp_enabled,
            decimal_precision=decimal_precision,
            fill_invalid=self.fill_invalid,
            verbose=verbose,
        )

    def _validate_coefficient_map_headers(
        self, coefficient_map: pd.DataFrame, mandatory_cols: list[str]
    ) -> None:
        """Validate that required columns exist in the coefficient map.

        Checks if the mandatory columns specified exist as columns in the coefficient_map DataFrame.

        Args:
            coefficient_map (pd.DataFrame): The DataFrame to check for mandatory columns.
            mandatory_cols (list): List of column names that must exist in the coefficient_map.

        Raises:
            KeyError: If any required columns are missing from the coefficient_map.
        """
        missing = set(mandatory_cols).difference(coefficient_map.columns)
        if missing:
            raise KeyError(f"{missing} not found among coefficient_map columns.")

    def compute_coefficients(self) -> dict[str, Any]:
        """Compute coefficient matrices based on formulas in the coefficient map.

        Processes each row in the coefficient_map, evaluating the formula and
        storing the result in a dictionary. Skips entries with missing formulas
        or missing variables, and handles exceptions gracefully.

        The computation process follows these steps for each row in the coefficient map:
        1. Extract the coefficient name and formula from the specified columns
        2. Skip empty or missing formulas with a notification
        3. Parse the formula into a symbolic expression for analysis
        4. Extract all variables used in the formula
        5. Verify all required variables exist in the data dictionary
        6. Skip calculations with missing variables with a notification
        7. Evaluate the formula using the FormulaEvaluator
        8. Store the result in the output dictionary with the coefficient name as key
        9. Provide success feedback for completed calculations

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
            expr = self.evaluator.parse_formula(formula)

            # Check if all variables in the formula exist in the data dictionary
            variables = self.evaluator.extract_variables(expr)
            missing_vars = [var for var in variables if var not in self.data_dict]

            if missing_vars:
                print(f"Skipping coefficient {name}: Missing variables {missing_vars}")
                continue

            # Evaluate the expression
            computed_value = self.evaluator.evaluate_formula(expr)

            # Store the coefficient
            result[name] = computed_value

            print(f"Successfully computed coefficient: {name}")

        return result
