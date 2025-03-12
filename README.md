# SSB Coefficient Maker

[![PyPI](https://img.shields.io/pypi/v/ssb-coefficient-maker.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/ssb-coefficient-maker.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/ssb-coefficient-maker)][pypi status]
[![License](https://img.shields.io/pypi/l/ssb-coefficient-maker)][license]

[![Documentation](https://github.com/statisticsnorway/ssb-coefficient-maker/actions/workflows/docs.yml/badge.svg)][documentation]
[![Tests](https://github.com/statisticsnorway/ssb-coefficient-maker/actions/workflows/tests.yml/badge.svg)][tests]
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-coefficient-maker&metric=coverage)][sonarcov]
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=statisticsnorway_ssb-coefficient-maker&metric=alert_status)][sonarquality]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)][poetry]

[pypi status]: https://pypi.org/project/ssb-coefficient-maker/
[documentation]: https://statisticsnorway.github.io/ssb-coefficient-maker
[tests]: https://github.com/statisticsnorway/ssb-coefficient-maker/actions?workflow=Tests

[sonarcov]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-coefficient-maker
[sonarquality]: https://sonarcloud.io/summary/overall?id=statisticsnorway_ssb-coefficient-maker
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[poetry]: https://python-poetry.org/

## Features

- High-precision mathematical formula evaluation for pandas DataFrames and Series
- Arbitrary decimal precision support using mpmath for accurate financial and statistical calculations
- Sophisticated validation system for detecting and handling invalid values (NaN, Inf)
- Coefficient calculation from formula definitions stored in mapping tables
- Comprehensive error reporting with detailed diagnostics for debugging complex formulas
- Support for mixed operations between DataFrames and Series with proper broadcasting
- Configurable precision and error handling to suit different use cases

## Requirements

- python >=3.10
- click >=8.0.1
- pandas >=2.2.3
- numpy >=2.2.3
- sympy >=1.13.3
- mpmath >=1.3.0
- pydantic >=2.10.6

## Installation

You can install _SSB Coefficient Maker_ via [pip] from [PyPI]:

```console
pip install ssb-coefficient-maker
```

## Usage

### Basic Formula Evaluation

```python
import pandas as pd
from ssb_coefficient_maker import FormulaEvaluator

# Create input data
data = {
    'matrix_a': pd.DataFrame({
        'col1': [1.0, 2.0, 3.0],
        'col2': [4.0, 5.0, 6.0]
    }),
    'vector_b': pd.Series([10.0, 20.0, 30.0])
}

# Initialize evaluator with default settings (arbitrary precision enabled)
evaluator = FormulaEvaluator(data)

# Evaluate a formula
result = evaluator.evaluate_formula('matrix_a * vector_b')
print(result)
```

### Computing Multiple Coefficients

```python
import pandas as pd
from ssb_coefficient_maker import CoefficientCalculator

# Create input data
data = {
    'input_matrix': pd.DataFrame({
        'A': [1.0, 2.0],
        'B': [3.0, 4.0]
    }),
    'adjustment': pd.Series([0.9, 1.1])
}

# Define coefficient formulas
coef_map = pd.DataFrame({
    'navn': ['adjusted_matrix', 'squared_matrix'],
    'formel': ['input_matrix * adjustment', 'input_matrix * input_matrix']
})

# Create calculator with safe settings
calculator = CoefficientCalculator(
    data,
    coef_map,
    fill_invalid=True,  # Replace invalid values with zeros
    verbose=True        # Print detailed information during calculation
)

# Compute all coefficients
results = calculator.compute_coefficients()

# Access the results
adjusted = results['adjusted_matrix']
squared = results['squared_matrix']
```

### Handling Division by Zero

```python
import pandas as pd
from ssb_coefficient_maker import FormulaEvaluator

# Data with potential division by zero
data = {
    'numerator': pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]}),
    'denominator': pd.DataFrame({'A': [1.0, 0.0], 'B': [0.0, 2.0]})
}

# Safe evaluator that replaces Inf/NaN with zeros
safe_eval = FormulaEvaluator(data, fill_invalid=True)
result = safe_eval.evaluate_formula('numerator / denominator')
```

### Working with High Precision

```python
import pandas as pd
from ssb_coefficient_maker import FormulaEvaluator

# Financial data requiring high precision
data = {
    'principal': pd.Series([1000000.00, 2000000.00, 5000000.00]),
    'rate': pd.Series([0.0325, 0.0310, 0.0295]),
    'periods': pd.Series([360, 240, 180])
}

# Create evaluator with 50 digits of precision
high_precision = FormulaEvaluator(
    data,
    adp_enabled=True,
    decimal_precision=50
)

# Calculate monthly payment using formula
result = high_precision.evaluate_formula(
    'principal * (rate/12) / (1 - (1 + rate/12)**(-periods))'
)
```

Please see the [Reference Guide] for more detailed examples and advanced usage.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_SSB Coefficient Maker_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [Statistics Norway]'s [SSB PyPI Template].

[statistics norway]: https://www.ssb.no/en
[pypi]: https://pypi.org/
[ssb pypi template]: https://github.com/statisticsnorway/ssb-pypitemplate
[file an issue]: https://github.com/statisticsnorway/ssb-coefficient-maker/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/statisticsnorway/ssb-coefficient-maker/blob/main/LICENSE
[contributor guide]: https://github.com/statisticsnorway/ssb-coefficient-maker/blob/main/CONTRIBUTING.md
[reference guide]: https://statisticsnorway.github.io/ssb-coefficient-maker/reference.html
