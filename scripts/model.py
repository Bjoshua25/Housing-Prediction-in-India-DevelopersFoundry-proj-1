# import libraries

import os
import sys
import joblib 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression, ridge_regression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
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






# ------ GRIDSEARCH CV ---------
def run_gridsearch_model(X_train, y_train, model_instance, parameters):
    """
    About:
        Function that runs gridsearch cross validation to obtain best parameters
    Input:
        X_train,
        y_train,
        model_instance,
        parameters - (dict)
    output:
        grid_search,
        cv_result - (pd.DataFrame)
    """

    # full pipeline
    model = build_full_pipeline(model_instance)

    # grid search 
    grid_search = GridSearchCV(
        model_instance,
        parameters,
        scoring = 'neg_root_mean_squared_error',
        cv = 3,
        verbose= 2,
    )
    grid_search.fit(X_train, y_train)

    # Dataframe of cv_result
    cv_res = pd.DataFrame(grid_search.cv_results_)

    return grid_search, cv_res





# ------- RANDOMIZED SEARCH CV ------

def run_randomize_search(model_instance, X_train, y_train, params_distribs, n_iter=10):
    """
    About:
        Function to perform Randomize search to obtain best model parameters
    Input:
        model_instance, eg LinearRegression()
        X_train - pd.DataFrame
        y_train - pd.Series
        params_distribs - dict
        n_iter - int
    output:
        rand_search,
        cv_result - pd.DataFrame
    """
    
    # full pipeline
    model = build_full_pipeline(model_instance)

    # random search
    rand_search = RandomizedSearchCV(
        estimator= model,
        param_distributions= params_distribs,
        n_iter= n_iter,
        scoring="neg_root_mean_squared_error",
        cv= 3,
        verbose=2,
        random_state=42,
    )
    rand_search.fit(X_train, y_train)

    # Dataframe of cv_result
    cv_res = pd.DataFrame(rand_search.cv_results_)

    # Best parameter
    best_param = rand_search.best_params_

    # best_model 
    best_model = rand_search.best_estimator_

    return rand_search, cv_res





# ----- SAVING BEST MODEL ------
def save_model (model, filename= "india_housing_price_model.pkl"):
    """
    About:
        Save model using joblib as .pkl
    Input: 
        model,
        filename - str
    output:
        completion message
    """
    # save best model
    joblib.dump(
        model,
        Path.cwd().parent / filename #filepath
    )

    print(f"Successfully Saved {filename} to {Path.cwd().parent}")

