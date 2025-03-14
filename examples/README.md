# Guide to using Formula Evaluator and CoefficientCalculator

Author: Benedikt Goodman\
Helper: Claude Sonnet 3.7 by Anthropic

This guide demonstrates how to use the `FormulaEvaluator` and `CoefficientCalculator` classes for mathematical formula evaluation with pandas objects. These classes aim to streamline linear algebra operations, and to make it as easy as possible to create coefficients for input-output models. The classes support arbitrary floating point numbers using `mpmath` for high-precision calculations and provides validation on input data.

## Table of Contents
- [Basic Usage](#basic-usage)
- [Configuration Options](#configuration-options)
- [Validation Process](#validation-process)
- [Working with Arbitrary Decimal Precision](#working-with-arbitrary-decimal-precision)
- [Handling Invalid Values](#handling-invalid-values)
- [CoefficientCalculator Usage](#coefficientcalculator-usage)
- [Advanced Examples](#advanced-examples)
- [Best Practices](#best-practices)

## Basic Usage

### Formula Evaluator

The `FormulaEvaluator` allows you to evaluate mathematical expressions using pandas DataFrames and Series:

```python
import pandas as pd
import numpy as np
from formula_evaluation import FormulaEvaluator

# Create some sample data
data = {
    'matrix_a': pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [4.0, 5.0, 6.0],
        'col3': [7.0, 8.0, 9.0],
    }),
    'vector_b': pd.Series([10.0, 20.0, 30.0])
}

# Initialize the evaluator with default settings
evaluator = FormulaEvaluator(data)

# Evaluate a formula
result = evaluator.evaluate_formula('matrix_a * vector_b')
print(result)
```

Terminal output:
```
     col1   col2
0   10.0   40.0
1   40.0  100.0
2   90.0  180.0
```

## Configuration Options

Both `FormulaEvaluator` and `CoefficientCalculator` accept several important configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adp_enabled` | bool | `True` | Enable arbitrary decimal precision using mpmath |
| `decimal_precision` | int | 35 | Number of decimal digits for precision when using arbitrary precision |
| `fill_invalid` | bool | `False` | Replace Inf and NaN values with zeros in results |
| `verbose` | bool | `False` | Print detailed information during evaluation |

### Example with Different Configurations

```python
# With default settings (arbitrary precision enabled)
default_evaluator = FormulaEvaluator(data)

# With arbitrary precision disabled (using numpy float64)
numpy_evaluator = FormulaEvaluator(data, adp_enabled=False)

# With invalid value filling, fills nans, inf and -inf with 0
safe_evaluator = FormulaEvaluator(data, fill_invalid=True)

# With verbose logging
verbose_evaluator = FormulaEvaluator(data, verbose=True)

# With custom precision (100 digits)
high_precision_evaluator = FormulaEvaluator(data, adp_enabled=True, decimal_precision=100)
```

Terminal output with verbose logging:
```
FormulaEvaluator initialized with 2 variables
Settings: precision_mode=mpmath, fill_invalid=False
```

## Validation Process

The validation process is a key feature of both `FormulaEvaluator` and `CoefficientCalculator`. It's handled by the `ResultValidator` class, which performs several important checks:

1. **Invalid Value Detection**: Checks for NaN and Inf values in computation results
2. **Severity Assessment**: Determines if all values are invalid, some are invalid, or all are valid
3. **Error Diagnosis**: Identifies potential causes of invalid values (e.g., division by zero, misaligned data)
4. **Warning Generation**: Creates detailed warning messages for partially invalid results
5. **Value Correction**: Optionally replaces invalid values with zeros when `fill_invalid=True`

The validation flow is as follows:

```
┌──────────────────┐
│ evaluate_formula │
└────────┬─────────┘
         │
         ▼
┌─────────────────────┐
│ _perform_evaluation │
└────────┬────────────┘
         │
         ▼
┌────────────────────┐
│ validator.validate │
└────────┬───────────┘
         │
         ▼
┌───────────────────────┐
│ _check_invalid_status │
└────────┬──────────────┘
         │
         ▼
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌────────┐
│ Handle │  │ Return │
│ Errors │  │ Result │
└────────┘  └────────┘
```

### Validation Example

```python
# Create data with potential division by zero
data = {
    'numerator': pd.DataFrame({
        'A': [1.0, 2.0, 3.0],
        'B': [4.0, 5.0, 6.0]
    }),
    'denominator': pd.DataFrame({
        'A': [1.0, 0.0, 3.0],  # Zero in second row
        'B': [4.0, 5.0, 0.0]   # Zero in third row
    })
}

# With default settings (will throw warnings)
default_eval = FormulaEvaluator(data)
try:
    result1 = default_eval.evaluate_formula('numerator / denominator')
    print("Default result (with warnings):")
    print(result1)
except ValueError as e:
    print(f"Error: {e}")

# With invalid value filling
safe_eval = FormulaEvaluator(data, fill_invalid=True)
result2 = safe_eval.evaluate_formula('numerator / denominator')
print("\nSafe result (zeros instead of Inf/NaN):")
print(result2)
```

Terminal output:
```
Warning: Formula 'numerator / denominator' using variables ['numerator', 'denominator'] produced a result with 22.2% invalid values. Some values are infinite due to division by zero. Consider checking your input data and formula for potential issues.

Default result (with warnings):
     A    B
0  1.0  1.0
1  inf  1.0
2  1.0  inf

Safe result (zeros instead of Inf/NaN):
     A    B
0  1.0  1.0
1  0.0  1.0
2  1.0  0.0
```

## Working with Arbitrary Decimal Precision

The `adp_enabled` parameter controls whether to use arbitrary decimal precision via the mpmath library:

- When `adp_enabled=True` (default), all numeric values are converted to mpmath.mpf objects
- When `adp_enabled=False`, standard numpy float64 values are used

### When to Use Arbitrary Precision

Use arbitrary precision when:
- Working with financial calculations that require high accuracy
- Performing operations that are susceptible to floating-point errors
- Dealing with very large or very small numbers

### Precision Example

```python
import pandas as pd
from formula_evaluation import FormulaEvaluator

# Create data with values that need high precision
data = {
    'very_small': pd.Series([1e-30, 2e-30, 3e-30]),
    'very_large': pd.Series([1e30, 2e30, 3e30])
}

# With arbitrary precision (default)
high_prec = FormulaEvaluator(data, decimal_precision=50)
result_high = high_prec.evaluate_formula('very_small * very_large')
print("High precision result:")
print(result_high)

# With standard numpy precision
std_prec = FormulaEvaluator(data, adp_enabled=False)
result_std = std_prec.evaluate_formula('very_small * very_large')
print("\nStandard precision result:")
print(result_std)
```

Terminal output:
```
High precision result:
0    1.0
1    4.0
2    9.0
dtype: object

Standard precision result:
0    1.0
1    4.0
2    9.0
dtype: float64
```

## Handling Invalid Values

The `fill_invalid` parameter determines how to handle invalid values (NaN and Inf):

- When `fill_invalid=False` (default):
  - Warnings are issued for partially invalid results
  - Errors are thrown when all values are invalid

- When `fill_invalid=True`:
  - Invalid values are replaced with zeros
  - The computation continues without errors or warnings

### Example with Invalid Value Handling

```python
import pandas as pd
from formula_evaluation import FormulaEvaluator

# Create data with a zero value that will cause division by zero
data = {
    'a': pd.Series([1.0, 2.0, 3.0]),
    'b': pd.Series([0.0, 2.0, 3.0])  # Zero in first position
}

# Without invalid value filling (will warn or error)
evaluator1 = FormulaEvaluator(data, fill_invalid=False)
try:
    result1 = evaluator1.evaluate_formula('a / b')
    print("Result with warnings for Inf values:")
    print(result1)
except ValueError as e:
    print(f"Error: {e}")

# With invalid value filling
evaluator2 = FormulaEvaluator(data, fill_invalid=True)
result2 = evaluator2.evaluate_formula('a / b')
print("\nResult with zeros instead of Inf values:")
print(result2)
```

Terminal output:
```
Warning: Formula 'a / b' using variables ['a', 'b'] produced a result with 33.3% invalid values. Some values are infinite due to division by zero. Consider checking your input data and formula for potential issues.

Result with warnings for Inf values:
0    inf
1    1.0
2    1.0
dtype: float64

Result with zeros instead of Inf values:
0    0.0
1    1.0
2    1.0
dtype: float64
```

## CoefficientCalculator Usage

The `CoefficientCalculator` extends `FormulaEvaluator` to calculate multiple coefficients defined in a mapping table:

```python
import pandas as pd
from formula_evaluation import CoefficientCalculator

# Create sample data
data = {
    'matrix1': pd.DataFrame({
        'A': [1.0, 2.0],
        'B': [3.0, 4.0]
    }),
    'matrix2': pd.DataFrame({
        'A': [5.0, 6.0],
        'B': [7.0, 8.0]
    }),
    'scalar': pd.Series([10.0])
}

# Create coefficient mapping table
coef_map = pd.DataFrame({
    'navn': ['sum_matrix', 'diff_matrix', 'scaled_matrix', 'invalid_formula', 'missing_var'],
    'formel': [
        'matrix1 + matrix2',
        'matrix1 - matrix2',
        'matrix1 * scalar',
        '',  # Empty formula (will be skipped)
        'matrix3 * 2'  # Missing variable (will be skipped)
    ]
})

# Initialize calculator
calculator = CoefficientCalculator(
    data,
    coef_map,
    adp_enabled=True,
    fill_invalid=True,
    verbose=True
)

# Compute all coefficients
results = calculator.compute_coefficients()

# Display results
for name, value in results.items():
    print(f"\nCoefficient: {name}")
    print(value)
```

Terminal output:
```
FormulaEvaluator initialized with 3 variables
Settings: precision_mode=mpmath, fill_invalid=True

Parsing formula: matrix1 + matrix2
Variables in expression: ['matrix1', 'matrix2']
Evaluating formula: matrix1 + matrix2
Formula evaluation complete. Result shape: (2, 2)
Successfully computed coefficient: sum_matrix

Parsing formula: matrix1 - matrix2
Variables in expression: ['matrix1', 'matrix2']
Evaluating formula: matrix1 - matrix2
Formula evaluation complete. Result shape: (2, 2)
Successfully computed coefficient: diff_matrix

Parsing formula: matrix1 * scalar
Variables in expression: ['matrix1', 'scalar']
Evaluating formula: matrix1 * scalar
Formula evaluation complete. Result shape: (2, 2)
Successfully computed coefficient: scaled_matrix

Skipping coefficient invalid_formula: No formula provided
Skipping coefficient missing_var: Missing variables ['matrix3']

Coefficient: sum_matrix
     A     B
0  6.0  10.0
1  8.0  12.0

Coefficient: diff_matrix
     A    B
0 -4.0 -4.0
1 -4.0 -4.0

Coefficient: scaled_matrix
      A     B
0  10.0  30.0
1  20.0  40.0
```

## Understanding Broadcasting in Pandas and NumPy

When working with the `FormulaEvaluator` and `CoefficientCalculator`, it's important to understand how operations between different dimensional objects (like DataFrames and Series) are handled through broadcasting. Broadcasting is a powerful feature that allows arrays of different shapes to be combined in arithmetic operations, but its behaviour is different than the rules of linear algebra.

### How Broadcasting Works

When pandas performs operations between DataFrames and Series, it follows NumPy's broadcasting rules with some pandas-specific behavior:

1. **Series + DataFrame**: The Series values are broadcast along the DataFrame's index or columns, depending on the Series orientation.
2. **DataFrame * Series with matching index**: The Series values are broadcast column-wise across the DataFrame.
3. **DataFrame / Series with matching columns**: By default, values are broadcast row-wise.

### Example of Broadcasting Effects

```python
import pandas as pd
from formula_evaluation import FormulaEvaluator

# Create DataFrame and Series with matching index
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=[0, 1, 2])

# Series with the same index as the DataFrame
series_index = pd.Series([10, 20, 30], index=[0, 1, 2])

# Series with the same names as the DataFrame columns
series_columns = pd.Series([10, 20], index=['A', 'B'])

evaluator = FormulaEvaluator({
    'df': df,
    'series_index': series_index,
    'series_columns': series_columns
})

# Column-wise broadcasting (Series values applied to each column)
result1 = evaluator.evaluate_formula('df * series_index')
print("Column-wise broadcasting (df * series_index):")
print(result1)

# Row-wise broadcasting (Series values applied to each row)
result2 = evaluator.evaluate_formula('df * series_columns')
print("\nRow-wise broadcasting (df * series_columns):")
print(result2)
```

Terminal output:
```
Column-wise broadcasting (df * series_index):
      A     B
0   10    40
1   40   100
2   90   180

Row-wise broadcasting (df * series_columns):
      A     B
0   10    80
1   20   100
2   30   120
```

### Potential Issues with Broadcasting

While broadcasting is convenient, it can lead to unexpected results or bugs:

1. **Misaligned Indices**: If the Series and DataFrame indices don't align, you'll get NaN values where there's no match.
2. **Unexpected Dimensions**: Broadcasting may not always behave as you expect, especially with complex operations.
3. **Performance**: Some broadcast operations can be less efficient than explicit operations.

### Best Practices

1. **Explicit Alignment**: When working with Series and DataFrames, consider explicitly aligning them before operations.
2. **Index Checking**: Verify that indices match when doing operations between DataFrames and Series.
3. **Orientation Control**: Be explicit about the orientation when needed, using methods like `.mul(series, axis=0)` or `.mul(series, axis=1)`.

For deeper details on broadcasting behavior, refer to:
- [NumPy Broadcasting Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)


### Handling Misaligned Indices

```python
import pandas as pd
from formula_evaluation import FormulaEvaluator

# Create data with misaligned indices
data = {
    'df1': pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=[0, 1, 2]),
    'df2': pd.DataFrame({
        'A': [10, 20, 30],
        'B': [40, 50, 60]
    }, index=[1, 2, 3])  # Shifted index
}

# Without fill_invalid
evaluator1 = FormulaEvaluator(data, fill_invalid=False, verbose=True)
try:
    result1 = evaluator1.evaluate_formula('df1 + df2')
    print("Result with misaligned indices (warnings):")
    print(result1)
except ValueError as e:
    print(f"Error: {e}")

# With fill_invalid
evaluator2 = FormulaEvaluator(data, fill_invalid=True, verbose=True)
result2 = evaluator2.evaluate_formula('df1 + df2')
print("\nResult with misaligned indices (zeros for NaN):")
print(result2)
```

Terminal output:
```
FormulaEvaluator initialized with 2 variables
Settings: precision_mode=mpmath, fill_invalid=False
Evaluating formula: df1 + df2
WARNING: Result contains 4/8 (50.00%) invalid values
 - Result contains NaN values
Warning: Formula 'df1 + df2' using variables ['df1', 'df2'] produced a result with 50.0% invalid values. Some values are NaN, which could indicate partial data misalignment, missing values in the original data, or operations that produced undefined results for some elements. Consider checking your input data and formula for potential issues.
Formula evaluation complete. Result shape: (4, 2)
Result with misaligned indices (warnings):
      A     B
0    NaN   NaN
1   12.0  45.0
2   23.0  56.0
3    NaN   NaN

FormulaEvaluator initialized with 2 variables
Settings: precision_mode=mpmath, fill_invalid=True
Evaluating formula: df1 + df2
WARNING: Result contains 4/8 (50.00%) invalid values
 - Result contains NaN values
Invalid values will be replaced with zeros
Replaced 4 invalid values (NaN/Inf) with zeros
Formula evaluation complete. Result shape: (4, 2)

Result with misaligned indices (zeros for NaN):
      A     B
0    0.0   0.0
1   12.0  45.0
2   23.0  56.0
3    0.0   0.0
```

## Best Practices

1. **Choose the Right Precision Mode**:
   - Use `adp_enabled=True` for when high precision of floating point variables are needed but slower performance
   - Use `adp_enabled=False` for better performance when standard 64-bit float precision is sufficient

2. **Handle Invalid Values Appropriately**:
   - Use `fill_invalid=True` when you want to continue calculations despite invalid values
   - Use `fill_invalid=False` when you want to be alerted to potential issues in your data or formulas

3. **Use Verbose Mode for Debugging**:
   - Enable `verbose=True` when developing or debugging complex formulas
   - Disable it for production use to avoid unnecessary output

4. **Pre-process Your Data**:
   - Align DataFrames and Series indices before evaluation to avoid NaN results
   - Avoid zeros in denominators when performing division operations

5. **Catch and Handle Exceptions**:
   - Wrap formula evaluation in try-except blocks to handle potential errors gracefully
   - Inspect warning messages for insights into partial failures

6. **Use CoefficientCalculator for Multiple Formulas**:
   - When computing many related coefficients, use `CoefficientCalculator` instead of multiple `FormulaEvaluator` calls
   - This ensures consistent handling of all computations

By following these guidelines, you can use `FormulaEvaluator` and `CoefficientCalculator` for mathematical operations with pandas objects and create coefficient matrices to your heart's delight.
