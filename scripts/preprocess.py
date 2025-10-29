# import Libraries
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder



# ========= WRANGLE FUNCTION ==========
# pandas-specific column creation
# complex clipping logic, dropping original columns
def transform_data(df):
    """
    About:
        Function to perform initial data wrangling, feature creation, and cleaning (clipping/capping/log transformation)
    Input: 
        pd.Dataframe
    Output:
        pd.DataFrame - cleaned datafarme
    """

    # Target column
    target_col = "TARGET(PRICE_IN_LACS)"
    log_target_col = "TARGET(PRICE_IN_LACS)_LOG"

    # Log transform the target only when it is available
    if target_col in df.columns:
        df[log_target_col] = np.log1p(df[target_col])


    # Features Creation: CITY 
    df["CITY"] = df["ADDRESS"].str.split(",").str[-1].str.strip()
    

    # Capping BHK_NO. 
    BHK_CAP = 6
    df["BHK_NO._CAPPED"] = df["BHK_NO."].apply(lambda x: min(x, BHK_CAP))


    # Clipping function
    def clip_iqr(series, factor=1.5):
        Q1 = series.quantile(0.25)
        Q2 = series.quantile(0.75)
        IQR = Q2 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q2 + factor * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)

    # Apply clipping to LATITUDE, LONGITUDE, SQUARE_FT
    df["LATITUDE_CLIPPED"] = clip_iqr(df["LATITUDE"])
    df["LONGITUDE_CLIPPED"] = clip_iqr(df["LONGITUDE"])
    df["SQUARE_FT_CLIPPED"] = clip_iqr(df["SQUARE_FT"])

    # Creating Ratio Features
    df["SQFT_PER_BHK"] = df["SQUARE_FT_CLIPPED"] / df["BHK_NO._CAPPED"]
    df["SQFT_PER_BHK_LOG"] = np.log1p(df["SQFT_PER_BHK"])


    # list of unnecessary columns
    drop_cols = ["ADDRESS", "BHK_NO.","SQUARE_FT", "LATITUDE", "LONGITUDE", "READY_TO_MOVE", "SQFT_PER_BHK", "SQUARE_FT_CLIPPED"]

    # Add Original Target Column to list of drop_cols if Present
    if target_col in df.columns:
        drop_cols.append(target_col)

    # Drop unnecessary columns
    df = df.drop(columns=drop_cols, errors='ignore')

    return df





# ------- LOAD AND TRANSFORM PIPELINE ------
def load_and_transform (filepath):
    """
    About:
        Load and transform datasets before building full model pipeline.
        It handles the target column conditionally.
    input: (str)
        filepath to the data directory
    return: (pd.DataFrame, pd.Series or None)
        X_train, X_val, y_train, y_val
    """

    # ----- IMPORT DATASET ------
    df = pd.read_csv(filepath)

    # ----- TRANSFORM DATA ------
    X = transform_data(df)

    # initialize "y" as None
    y = None

    # test target separation if exists
    if "TARGET(PRICE_IN_LACS)_LOG" in X.columns:
        y = X.pop("TARGET(PRICE_IN_LACS)_LOG")
    
        # ----- TRAIN-VALIDATION SET SPLIT ------
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val





# --------PIPELINE CONSTRUCTION FUNCTION------------

def build_full_pipeline(model_instance):
    """
    About: 
        Builds full End-to-End ML pipeline including preprocessing (ColumnTransformation) and the model
    Input: 
        Model_instance (instantiated models eg. LinearRegression)
    Output: 
        Full Pipeline (sklearn.pipeline.Pipeline)
    """

    # ====== DEFINE FEATURES FOR COLUMNTRANSFORMER =======
    # list of numeric features
    numeric_features = [
        'UNDER_CONSTRUCTION',
        'RERA',
        'RESALE',
        'BHK_NO._CAPPED',
        'LATITUDE_CLIPPED',
        'LONGITUDE_CLIPPED',
        'SQFT_PER_BHK_LOG'
    ]

    # list of categorical features for OneHotEncoding (Low cardinality)
    ohe_features = ['POSTED_BY', 'BHK_OR_RK']


    # High-Cardinality features for Target Encoding
    target_encode_feature = ["CITY"]



    # ======= CREATE TRANSFORMER ======
    # standard scaling of all numeric features
    numeric_transformer = StandardScaler()

    # onehotencoding for low cardinality features
    ohe_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output = False, drop ="first")

    # Target Encodng for CITY feature due to its high cardinality
    target_transformer = TargetEncoder(cols=target_encode_feature, smoothing=0.1)



    # ======= CREATE THE COLUMN TRANSFORMER ======
    preprocessor = ColumnTransformer(
        transformers=[
            ("tge", target_transformer, target_encode_feature), #Target Encoding
            ("num", numeric_transformer, numeric_features), # Scaling
            ("ohe", ohe_transformer, ohe_features) #oneHotEncoding
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")


    # ====== CREATE FULL PIPELINE ========
    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model_instance)
        ]
    )

    return full_pipeline







