# Code supplement for ''Macroscopic Properties of Equity Markets: Stylized Facts and Portfolio Performance''

This repository contains all of the code to reproduce the analysis in [Campbell, Song & Wong (2025)]. Each section in the paper has a corresponding code folder that generates the figures and results. The market data used in the study was obtained under license from the Center for Research in Security Prices (CRSP) through Wharton Research Data Services (WRDS). Academic licenses are available at many institutions and the data can be accessed directly or through an API. Our data was obtained online directly from WRDS. For a tutorial on how to download and manipulate the data in Python see https://github.com/johruf/CRSP_on_WRDS_introduction.

Included in this repository is a pre-processing folder whose code files can be used to clean and format the CRSP data. We also provide a backtesting engine for long-only portfolios in markets with dividends, delistings, and transaction costs that implements the methodology of [Ruf & Xie (2020)]. This is used in Section 6 of the paper for an empirical study of portfolio performance. We include a dedicated folder for this backtesting engine that contains the main backtesting functionality and additional code to: 

i) Generate sample data in a format analogous to that of the paper,

ii) ''Pre-process'' the sample data for backtesting, and;

iii) Apply the backtesting engine to analyze trading performance on the sample data.

## Python Implementation

**All R scripts in this repository have been converted to Python.** The Python versions maintain the same functionality and produce equivalent results while using modern Python libraries for data analysis and visualization.

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation and CSV I/O
- `matplotlib>=3.4.0` - Plotting and visualization
- `scipy>=1.7.0` - Optimization and scientific computing
- `scikit-learn>=0.24.0` - Machine learning utilities (for regression analysis)

### File Naming Convention

Python files use lowercase with underscores (snake_case) instead of the original R naming convention. For example:
- `AtlasModelFunctions.R` → `atlas_model_functions.py`
- `PortfolioFunctions_CRSP.R` → `portfolio_functions_crsp.py`
- `CRSP_Backtesting_Test_Data.R` → `crsp_backtesting_test_data.py`

### Running the Code

Each Python script can be run independently. For example:

```bash
# Data preprocessing
cd Data_Pre_Processing_Code
python data_processing.py

# Run backtesting
cd Backtesting_Engine_and_Example
python crsp_backtesting_test_data.py

# Analysis scripts
cd Section_3_Capital_Distribution_and_Diversity
python capital_distribution_analysis.py
```

### Key Differences from R Implementation

1. **Data I/O**: Uses `pandas.read_csv()` instead of R's `readr::read_csv()`
2. **Matrix Operations**: Uses NumPy arrays instead of R matrices
3. **Plotting**: Uses `matplotlib` instead of R's base plotting or `ggplot2`
4. **Optimization**: Uses `scipy.optimize` instead of R's `optimize()`
5. **Array Indexing**: Python uses 0-based indexing (R uses 1-based)

### Original R Code

The original R scripts are preserved in the repository for reference. Both implementations produce equivalent results.
