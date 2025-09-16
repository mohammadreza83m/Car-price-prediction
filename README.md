# Car Price Prediction 

This project predicts car prices using different machine learning models:

- Dummy Regressor (baseline)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Results
- Gradient Boosting and Random Forest performed slightly better than Linear Regression
- Overall accuracy (R²) was low (~0.3), probably due to:
  - Small dataset
  - Missing or noisy data
  - Limited features (only Make, Colour, Doors, Odometer)

## How to Run
1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
