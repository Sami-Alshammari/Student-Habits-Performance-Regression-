# Student Exam Score Prediction

This project predicts students' exam scores using **Linear Regression** based on their habits and other features. It handles missing data, scales numeric features, encodes categorical features, evaluates model performance, and visualizes results.

---

## Features

- Handles missing data:
  - Numeric features: median imputation
  - Categorical features: most frequent value imputation
- Scales numeric features using **StandardScaler**
- Encodes categorical features using **OneHotEncoder**
- Splits data into training and testing sets
- Trains a **Linear Regression** model
- Evaluates performance with:
  - R² (coefficient of determination)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Visualizations:
  - **Predicted vs True**: shows how close predictions are to actual exam scores
  - **Residuals vs Predicted**: checks prediction errors for bias or patterns

---

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
