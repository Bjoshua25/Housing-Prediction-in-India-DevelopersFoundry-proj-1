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
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder



# ========= WRANGLE FUNCTION ==========
# pandas-specific column creation
# complex clipping logic, dropping original columns
def transform_data(df):
    """
    Function to perform initial data wrangling, feature creation, and cleaning (clipping/capping/log transformation) that does NOT require statistics from the train set. This function MUST be run on X_train, X_test, and new data.
    """
    # Features Creation: CITY and Log Target Price
    df["CITY"] = df["ADDRESS"].str.split(",").str[-1].str.strip()
    df["TARGET(PRICE_IN_LACS)_LOG"] = np.log1p(df["TARGET(PRICE_IN_LACS)"])


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


    # Drop unnecessary columns
    drop_cols = ["ADDRESS", "BHK_NO.","SQUARE_FT", "LATITUDE", "LONGITUDE", "TARGET(PRICE_IN_LACS)", "READY_TO_MOVE", "SQFT_PER_BHK", "SQUARE_FT_CLIPPED"]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df




# --------PIPELINE CONSTRUCTION FUNCTION------------

def col_transform_pipeline(model_instance):
    """
    Builds full End-to-End ML pipeline including preprocessing and the model
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

    return preprocessor




# # ====== CREATE FULL PIPELINE ========
# full = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("model", model_instance)
#     ]
# )


