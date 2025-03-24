This repository contains the code for the ["Energy Forecasting Data"](https://www.kaggle.com/competitions/energy-forecasting-data-challenge-public/overview) Kaggle Competition.

`ResiduaLoadPrediction.ipynb` contains the code to train different models (Linear Regression, K-Nearest Neighbors (KNN), XGBoost) to predict the residual load of an electricity grid. These models correspond to `linear_regression_model.pkl`, `knn_model.pkl`, and `xgboost_model.pkl`.

`residual_load_dashboard.py` contains the code to create a dashboard to visualize the predictions of the models and some insights about the data. The dashboard is created using the Streamlit library.
To run the dashboard, execute the following command from the root directory of the repository (make sure you have Streamlit installed):
```bash
streamlit run residual_load_dashboard.py
```