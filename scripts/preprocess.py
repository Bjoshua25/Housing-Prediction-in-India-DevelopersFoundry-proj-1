# import Libraries
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# ========= WRANGLE FUNCTION ==========
def transform_data(df):
    """
    Function to perform data wrangling and preprocessing on the housing dataset.
    """
    # ------------ Create CITY column from ADDRESS ------------
    df["CITY"] = (
        df["ADDRESS"]
        .str.split(",")
        .str[-1]
        .str.strip()
    )


    #------------ Log Transformation of TARGET(PRICE_IN_LACS) --------------
    log_features = ["TARGET(PRICE_IN_LACS)"]
    for col in log_features:
        df[col + "_LOG"] = np.log1p(df[col])


    # --------------- Capping BHK_NO. -----------------------
    BHK_CAP = 6
    df["BHK_NO._CAPPED"] = (
        df["BHK_NO."]
        .apply(lambda x: min(x, BHK_CAP))
    )


    # ------------ Clipping LATITUDE ---------------
    lat_Q1 = df["LATITUDE"].quantile(0.25)
    lat_Q3 = df["LATITUDE"].quantile(0.75)
    lat_IQR = lat_Q3 - lat_Q1

    # upper and lower bounds for Latitude
    lat_lower_bound = lat_Q1 - 1.5 * lat_IQR
    lat_upper_bound = lat_Q3 + 1.5 * lat_IQR

    # clipped latitude
    df["LATITUDE_CLIPPED"] = (
        df["LATITUDE"]
        .clip(lower=lat_lower_bound, upper=lat_upper_bound)
    )


    # ------------ Clipping LONGITUDE ---------------
    long_Q1 = df["LONGITUDE"].quantile(0.25)   
    long_Q3 = df["LONGITUDE"].quantile(0.75)
    long_IQR = long_Q3 - long_Q1

    # upper and lower bounds for Longitude
    long_lower_bound = long_Q1 - 1.5 * long_IQR
    long_higher_bound = long_Q3 + 1.5 * long_IQR

    # clipped longitude
    df["LONGITUDE_CLIPPED"] = (
        df["LONGITUDE"]
        .clip(lower=long_lower_bound, upper=long_higher_bound)
    )



    # ------------ Clipping SQUARE_FT_LOG ---------------
    sqr_ft_Q1 = df["SQUARE_FT"].quantile(0.25)
    sqr_ft_Q3 = df["SQUARE_FT"].quantile(0.75)
    sqr_ft_IQR = sqr_ft_Q3 - sqr_ft_Q1  

    # upper and lower bounds for SQUARE_FT
    sqr_ft_lower_bound = sqr_ft_Q1 - 1.5 * sqr_ft_IQR
    sqr_ft_higher_bound = sqr_ft_Q3 + 1.5 * sqr_ft_IQR

    # clipped SQUARE_FT
    df["SQUARE_FT_CLIPPED"] = (
        df["SQUARE_FT"]
        .clip(lower=sqr_ft_lower_bound, upper=sqr_ft_higher_bound)
    )   

    
    # Drop unnecessary columns
    drop_cols = ["ADDRESS", "BHK_NO.","SQUARE_FT", "LATITUDE", "LONGITUDE", "TARGET(PRICE_IN_LACS)"]
    df = df.drop(columns=drop_cols)

    return df