# Guide to using Formula Evaluator and CoefficientCalculator

Author: Benedikt Goodman

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
from ssb_coefficient_maker import FormulaEvaluator

# Create some sample data
data = {
    'matrix_a': pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [4.0, 5.0, 6.0],
        'col3': [7.0, 8.0, 9.0],
    }),
    'vector_b': pd.Series([10.0, 20.0, 30.0])  # Note: length matches the number of rows in matrix_a
}

# Initialize the evaluator with default settings
evaluator = FormulaEvaluator(data)

# Evaluate a formula
result = evaluator.evaluate_formula('matrix_a * vector_b')
print(result)
```

Terminal output:
```
# vector_b contains values [10.0, 20.0, 30.0]
# For matrix_a:
# Row 0: [1.0, 4.0, 7.0]
# Row 1: [2.0, 5.0, 8.0]
# Row 2: [3.0, 6.0, 9.0]

     col1   col2   col3
0   10.0   40.0   70.0  # Row 0 * 10.0
1   40.0  100.0  160.0  # Row 1 * 20.0
2   90.0  180.0  270.0  # Row 2 * 30.0
```

**Note on Broadcasting**: In this example, the Series `vector_b` has the same length as the number of rows in `matrix_a`, which results in row-wise broadcasting. Each row is multiplied by the corresponding value in the Series:
- Row 0 is multiplied by 10.0: `[1.0 × 10.0, 4.0 × 10.0, 7.0 × 10.0]` = `[10.0, 40.0, 70.0]`
- Row 1 is multiplied by 20.0: `[2.0 × 20.0, 5.0 × 20.0, 8.0 × 20.0]` = `[40.0, 100.0, 160.0]`
- Row 2 is multiplied by 30.0: `[3.0 × 30.0, 6.0 × 30.0, 9.0 × 30.0]` = `[90.0, 180.0, 270.0]`

Proper index alignment is crucial for predictable broadcasting behavior in pandas operations.

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
from ssb_coefficient_maker import FormulaEvaluator

# Create data with fractions that produce repeating decimals
data = {
    'numerator': pd.Series([1, 2, 1]),
    'denominator': pd.Series([3, 3, 7])
}

# Compare precision differences in division operations
print("Arbitrary precision result (50 digits):")
high_prec = FormulaEvaluator(data, decimal_precision=50)
print(high_prec.evaluate_formula('numerator / denominator'))

print("\nStandard precision result (float64):")
std_prec = FormulaEvaluator(data, adp_enabled=False)
print(std_prec.evaluate_formula('numerator / denominator'))
```

The actual representation of these values would be:
```
# Arbitrary precision result (50 digits):
# Each value is stored as an mpmath.mpf object with 50 digits of precision
0    0.33333333333333333333333333333333333333333333333333
1    0.66666666666666666666666666666666666666666666666667
2    0.14285714285714285714285714285714285714285714285714
dtype: object

# Standard precision result (float64):
# Each value is stored as a 64-bit floating point number with ~15-17 significant digits
0    0.3333333333333333
1    0.6666666666666666
2    0.14285714285714285
dtype: float64
```

## Handling Invalid Values

The `fill_invalid` parameter determines how to handle invalid values (NaN and Inf):

- When `fill_invalid=False` (default):
  - Warnings are issued for partially invalid results when they occur
  - Errors are thrown when all values are invalid
  - Detailed diagnostics are shown when `verbose=True`

- When `fill_invalid=True`:
  - Invalid values are replaced with zeros
  - The computation continues without errors or warnings

### Example with Invalid Value Handling

```python
import pandas as pd
from ssb_coefficient_maker import FormulaEvaluator

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

The `CoefficientCalculator` extends `FormulaEvaluator` to calculate multiple coefficients defined in a mapping table. This class now supports configurable column names for the coefficient map:

```python
import pandas as pd
from ssb_coefficient_maker import CoefficientCalculator

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
    # Create a scalar Series with the same index as the DataFrame columns for proper broadcasting
    'scalar': pd.Series([10.0, 10.0], index=['A', 'B'])
}

# Create coefficient mapping table with custom column names
coef_map = pd.DataFrame({
    'result_name': ['sum_matrix', 'diff_matrix', 'scaled_matrix', 'invalid_formula'],
    'calculation': [
        'matrix1 + matrix2',
        'matrix1 - matrix2',
        'matrix1 * scalar',
        ''  # Empty formula (will be skipped)
    ]
})

# Initialize calculator with custom column names
calculator = CoefficientCalculator(
    data,
    coef_map,
    result_name_col='result_name',  # Custom column for result names
    formula_name_col='calculation',  # Custom column for formulas
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

Terminal output would show something like:
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



### CoefficientCalculator Workflow

The CoefficientCalculator follows this process for each calculation:

1. Extract the coefficient name and formula from the specified columns
2. Skip empty or missing formulas with a notification
3. Parse the formula into a symbolic expression for analysis
4. Extract all variables used in the formula
5. Verify all required variables exist in the data dictionary
6. Skip calculations with missing variables with a notification
7. Evaluate the formula using the FormulaEvaluator
8. Store the result in the output dictionary with the coefficient name as key

## Understanding Operations in Pandas and NumPy

When working with the `FormulaEvaluator` and `CoefficientCalculator`, it's important to understand how operations between different pandas objects (like DataFrames and Series) are handled. Operations follow pandas and NumPy rules, which require compatible shapes and alignable indices.

Under the hood, pandas leverages NumPy's broadcasting mechanism for elementwise operations. Broadcasting allows NumPy to perform operations on arrays of different shapes by implicitly expanding the smaller array to match the shape of the larger array. This is particularly relevant when performing operations between DataFrames and Series.

### Key Points About Mathematical Operations

1. **Index Alignment**: Pandas automatically aligns indices when performing operations between objects.
2. **Compatible Shapes**: Objects must have compatible shapes for operations to succeed.
3. **Error Handling**: Operations between incompatible objects will result in errors or invalid values.

For more details on how broadcasting works in NumPy, refer to:
- [NumPy Broadcasting Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)

### Handling Misaligned Indices

```python
import pandas as pd
from ssb_coefficient_maker import FormulaEvaluator

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