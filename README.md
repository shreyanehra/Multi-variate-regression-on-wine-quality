

# Multi-Variate Linear Regression Example

This repository contains a Python script demonstrating multi-variate linear regression using the scikit-learn library and custom functions. The script aims to showcase how to build a regression model to predict wine quality based on various chemical properties.

## Overview

In this example, we use the `winequality-red.csv` dataset, which contains information about the chemical properties of red wine samples and their associated quality ratings. We perform multi-variate linear regression to predict wine quality based on features such as volatile acidity, citric acid, sulphates, and alcohol content.

## Requirements

To run the script, you'll need:

- Python 3.x
- pandas
- NumPy
- matplotlib
- scikit-learn

You can install the required libraries using pip:

```
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Clone this repository to your local machine:

```
git clone https://github.com/your_username/Multi-variate-regression-on-wine-quality.git

2. Navigate to the project directory:

```bash
cd Multi-variate-regression-on-wine-quality
```

3. Run the script:

```
python multi_variate_linear_regression.py
```

This will execute the script and perform multi-variate linear regression on the provided dataset.

## File Structure

- `multi_variate_linear_regression.py`: Main Python script containing the implementation of multi-variate linear regression.
- `winequality-red.csv`: Dataset containing information about red wine samples.

## Functionality

- The script imports the necessary libraries and loads the dataset using pandas.
- It preprocesses the data by removing certain columns and converting it into a NumPy array.
- The scikit-learn library's `LinearRegression` class is used to train a linear regression model.
- Custom functions are defined to calculate predictions and find the best model parameters.
- The script evaluates the model's performance using mean absolute error and mean squared error metrics.
- Finally, it generates predictions using the trained model and custom functions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

