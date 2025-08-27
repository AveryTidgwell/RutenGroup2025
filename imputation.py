import pandas as pd
import numpy as np

def imputation(df, imputation_type):
    if imputation_type == "mean":
        # Fill numeric columns with their own mean
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        if df.isnull().sum().sum() !=0: # See if any NaNs remain
            print('There are still NA values in the loaded dataset')
        return df
    else:
        print("Imputation is implemented for mean only")
        return np.nan
      
