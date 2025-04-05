# Boston Housing Price Prediction

This repository contains a data analysis and machine learning project that predicts housing prices in Boston using the classic Boston Housing Dataset.

## Overview

This project demonstrates a complete data science workflow:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature selection
- Linear regression modeling
- Model evaluation

## Dataset

The Boston Housing Dataset contains information about various features of houses in Boston suburbs and their corresponding median values (MEDV). The dataset includes the following features:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centers
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
- **LSTAT**: % lower status of the population
- **MEDV**: Median value of owner-occupied homes in $1000's

## Analysis Process

### 1. Exploratory Data Analysis
- Descriptive statistics
- Histograms of each feature
- Correlation analysis using heatmaps
- Scatterplots to identify relationships between variables

### 2. Data Preprocessing
- Handling missing values
- Log transformation of the target variable (MEDV) to normalize distribution
- Feature selection based on correlation and variance inflation factor (VIF)

### 3. Model Building
- Train-test split (70-30 ratio)
- Linear regression using statsmodels
- Multicollinearity check with VIF
- Model refinement by removing high VIF features

### 4. Model Evaluation
- Residual analysis
- Homoscedasticity tests
- R-squared evaluation
- Cross-validation
- Error metrics (RMSE, MAE, MAPE)

## Key Findings

- Strong predictors of housing prices include: CRIM, CHAS, NOX, RM, DIS, RAD, PTRATIO, and LSTAT
- Log transformation of the target variable (MEDV) improved model performance
- The model achieved excellent fit with very low error metrics in both training and test datasets
- No significant multicollinearity in the final model after removing TAX feature

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

## Usage

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv("HousingData.csv")

# Continue with analysis steps as shown in the notebook
```

## Visualization Samples

The analysis includes various visualizations:
- Feature distributions
- Correlation heatmap
- Feature relationship scatterplots
- Residual plots
- Q-Q plots for normality check

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Boston Housing Dataset is a classic dataset in machine learning literature
- This analysis is for educational purposes
