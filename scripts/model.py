# import libraries

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error




# ------- HANDLING DIRECTORY ------- 
# root folder directory
root_folder = Path.cwd().parent

# Adding parent path to system memory
if str(root_folder) not in sys.path:
    sys.path.insert(0, str(root_folder))


# import Required functions from script
from .preprocess import transform_data, build_full_pipeline, load_and_transform
# ---------------------------------------------------------------------------------------------------





# ----- INSTANTIATE MODELS ------

# linear Regression
lr_model = LinearRegression()

# decision tree
tree_model = DecisionTreeRegressor(random_state=42)

# random_forest model
rf_model = RandomForestRegressor()

# support vector regressor
svr_model = SVR()




# ---------- TRAIM MODEL FUNCTION ---------
def train_model(X_train, X_val, y_train, y_val, model_instance):
    """
    About: 
        Model training pipeline.
    Input:
        X_train (pd.DataFrame),
        X_test (pd.DataFrame),
        y_train (pd.Series)
        y_test (pd.Series)
        model_instance
    Output:
        RMSE print-out message
    """

    # ------ FULL MODEL PIPELINE -------
    pipeline = build_full_pipeline(
        model_instance=model_instance
    )

    # ----- TRAINING AND EVALUATE ------
    # Training the linear model
    pipeline.fit(X_train, y_train)

    # Try prediction on train set
    y_pred_train = pipeline.predict(X_train)

    # Prediction on validation set
    y_pred_val = pipeline.predict(X_val)

    # Evaluation with RMSE on test_data
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    # Evaluation with RMSE on validation data
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Output message
    train_msg = print(f"RMSE on Training set: {rmse_train}")
    test_msg = print(f"RMSE on validation set: {rmse_val}")

    return rmse_train, rmse_val
