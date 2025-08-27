import pandas as pd
import numpy as np

def z_imputation(df, imputation_type):
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
      
def imputation(df, biomarker_columns, imputation_type):
    if imputation_type == "mean":
        # Ensure biomarker_columns is a list
        biomarker_cols = list(biomarker_columns)

        for col in biomarker_cols:
            df[col] = df[col].fillna(df[col].mean())

        # Check for any remaining NaNs
        if df[biomarker_cols].isnull().values.any():
            print("There are still NA values in the loaded dataset")
        else:
            print("All missing values imputed.")

        return df
    else:
        print("Imputation is implemented for mean only")
        return np.nan


      
