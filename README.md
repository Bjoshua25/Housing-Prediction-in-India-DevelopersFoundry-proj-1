# **🏡 India Housing Price Prediction Model**

## **Project Goal**

The primary objective of this project is to develop a robust regression model to accurately predict residential property prices (TARGET) across major cities in India. The work will focus on creating high-quality features from raw data to ensure the best possible model performance.

---

## **Phase 1: Exploratory Data Analysis (EDA) and Data Transformation**

This initial phase focused heavily on understanding the data's inherent skewness and outlier issues, which are common in real estate datasets, and implementing robust preprocessing to meet the assumptions of linear models.

### **1\. Initial Data Assessment**

The raw data showed critical distributions requiring immediate attention:

* **Severe Right-Skewness:** The target variable (TARGET(PRICE\_IN\_LACS)) and property size (SQUARE\_FT) were highly skewed, with the majority of data points clustered near zero and a long, thin tail extending to extreme values.  
  * *Observation:* This necessitates a log transformation on the target variable.
 
  ![](images/without_transform_1.png)
  
* **Extreme Outliers:** Several features contained highly influential outliers:  
  * SQUARE\_FT: Included an astronomical value ($\\sim 2.5 \\times 10^8$ sq. ft.) likely indicating a data error.  
  * BHK\_NO.: Contained rare counts (e.g., 20 BHK) that disproportionately affect models.  
  * LATITUDE / LONGITUDE: Showed points outside plausible geographical bounds for India, suggesting input errors.


---


### **2\. Implemented Transformation Pipeline**

![](images/with_transform_1.png)

The following steps were implemented to clean and transform the numeric features:


| Step | Feature | Transformation Applied | Rationale |
|------|---------|------------------------|------------|
| 1\. Feature Creation | ADDRESS | Extraction of CITY | Created a high-value categorical feature by parsing the last element of the address string. |  
| 2\. Target Transformation | TARGET(PRICE\_IN\_LACS) | np.log1p (Log of $1+x$) $\\rightarrow$ TARGET(PRICE\_IN\_LACS)\_LOG | Normalized the distribution of the target variable to be near-Gaussian, ensuring optimal model fitting. |  
| 3\. Feature Capping | BHK\_NO. | Capped at 6 $\\rightarrow$ BHK\_NO.\_CAPPED | Mitigated the influence of rare, high-BHK outliers. |  
| 4\. Outlier Clipping (IQR) | SQUARE\_FT | Clipped using $1.5 \\times \\text{IQR}$ $\\rightarrow$ SQUARE\_FT\_CLIPPED | Contained the extreme outlier and compressed the tail of the size distribution. |  
| 5\. Outlier Clipping (IQR) | LATITUDE, LONGITUDE | Clipped using $1.5 \\times \\text{IQR}$ $\\rightarrow$ LATITUDE\_CLIPPED, LONGITUDE\_CLIPPED | Ensured geographic coordinates are within a reasonable and defined range. |

![](images/transformed_boxplot.png)

### **3\. Transformation Success and Resulting Metrics**

The transformations successfully prepared the data, as evidenced by the post-processing visualizations:

#### **A. Transformed Target Distribution**

The histogram for the log-transformed target variable confirms that the skewness has been effectively removed, resulting in a distribution suitable for linear models.

![](images/10_most_cities.png)

#### **B. Feature Correlations (Log Price)**

Post-transformation, the correlation analysis shows strong, expected relationships:

* SQUARE\_FT\_CLIPPED correlates highly with TARGET(PRICE\_IN\_LACS)\_LOG (r $\\approx$ **0.80**).  
* BHK\_NO.\_CAPPED shows a strong positive correlation (r $\\approx$ **0.61**).

![](images/heatplot.png)

This confirms that the transformations preserved the core predictive power of the most important numeric features while eliminating noise.

### **Next Steps**

The next phase will involve applying appropriate **encoding techniques** to the high-cardinality (like CITY) and low-cardinality (like POSTED\_BY) categorical features, finalizing the dataset for model selection and training.

---


## **Phase 2: Full Transformation Pipeline and Features Engineering**
The Transformation was splitted into two segments. The first segments involves mere pandas-based transformation that do not involve statistics tranformation. These involves processing such as creating new columns, log transformation, applying IQR clipping to handle outliers and column capping. Meanwhile the second segment involved statistical and more advanced transformations such as normalization of numeric features, one hot encoding of categorical features, target encoding of high cardinality categorical features, all of which were applied through sklearn `ColumnTransformer`.  
### **Pandas-based Transformation**
- Log transformation of Target Column `TARGET(PRICE_IN_LACS)` to handle its skewdness
- Create a new feature `CITY` by extracting city from `ADDRESS` column
- Capping number of bedrooms `BHK_NO` to 6 to handle outliers
- Clipping `LATITUDE, LONGITUDE, SQUARE_FT_CLIPPED` with the aid of Inter Quantile Range (IQR) to handle outliers
- Creating ratio Features for robust EDA and model prediction. like, square_ft of each rooms: `SQFT_PER_BHK`. which helps to know if the size of the bedrooms affect the price.
- Drop Unnecessary columns due to their multicollinearity, Leakage, or duplicate.
- This transformation function can be found in [transform_data()](scripts/preprocess.py)


### **sklearn ColumnTransformer Pipeline**
- Standard Scaling on numeric features
- OneHotEncoding on low cardinality categorical features, such as `POSTED_BY, BHK_OR_RK`.
- Target Encoding of high cardinality categorical feature like `CITY`. This help replace the cities by the `Mean House Prices` without offering leakage. This function was offered by `category_encoders`
- Finally, the column transformer was compiled with a list of tuples of all steps listed above.
- full pipeline function can be found in the preprocess script; [build_full_pipeline()](scripts/preprocess.py)


----


## **Phase 3: Training Different Models with Default Hyperparameters**
The goal of this phase is to discover models that will perform more effectly on the data before proceeding to tuning the model hyperparameter tuning or giving any model any focused attention. four models were trained with their default values. These models include; `Linear Regression`, `Decision Tree Regressor`, `Randomn Forest Regressor`, `Support Vector Regressor`. The outcome of this phase should that the linear regression performed poorly and followed by untuned SVR. However, Decision tree experienced a very serious Overfitting. While Random forest also had a slight but significant Overfiting.
- Ridge and Lasso Regression was trained on the data with GridSearchCV on the `alpha`. Result turns out to be slightly better but still poor which suggests that the data is non-linear 
- GridSearchCV was carried on decision tree with result still showing a slight overfit
- RandomSearchCV was carried in Random forest. result still showed slight overfit, but better than decision tree
- Check out the `model.py` script [here](scripts/model.py) and the model notebook [here](notebooks/models.ipynb) 
