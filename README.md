# SmartBite – Smart Food Demand Forecasting System

A Django web app that predicts daily demand for food items and groups items by demand pattern.

## Features
- **Demand prediction (Random Forest):** predicts expected units for a food item based on date, temperature and promotion
- **Demand clustering (K-Means):** groups food items by average demand and variation
- **Dashboard:** shows model metrics (MAE, RMSE, R²) and recent prediction history
- **Database:** predictions are stored using SQLite

## Tech Stack
Python, Django, scikit-learn, pandas, NumPy, SQLite, Bootstrap

## How to Run

    py -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python manage.py makemigrations forecast
    python manage.py migrate
    python manage.py runserver

Open http://127.0.0.1:8000/

## Note
This project uses **synthetic demo data**, not real restaurant sales. It is designed so that the model can be retrained on real sales data.

## Future Improvements
- Train on real historical sales data
- Compare models (naive baseline vs Random Forest vs XGBoost)
- Time-series backtesting
- CSV upload and dashboard charts
- Deployment