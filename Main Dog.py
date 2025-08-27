import pandas as pd
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
#load data
current_dir = os.getcwd()

def process_data(df, kalman = False):
    df, biomarker_columns = preprocessing()
    df = df.sort_values(by=['dog_id', 'Age']).reset_index(level=0, drop=True)

    for id in df['dog_id'].unique():
        df.loc[df['dog_id'] == id, biomarker_columns] = (
            df.loc[df['dog_id'] == id, biomarker_columns].interpolate(method='linear', limit_direction='both'))

    df = imputation(df, biomarker_columns, imputation_type = "mean")

    # Create a tuple of identifying fields
    df['dog_id'] = list(zip(df['dog_id'], df['Sex'], df['Fixed']))
    # Convert each unique tuple to a unique number
    df['dog_id'] = df.groupby('dog_id').ngroup()+1


    #do log transformation (original dolphin paper)
    df = log_transform(df, list(biomarker_columns)) #to convert left skewed data to one that resembles normal distribution

    #normalize to mean 0 and std 1
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df) #convert sex to [0,1] , Fixed to [0,1] and normalize age

    return df, biomarker_columns

def prepare_model_data(df, kalman = False):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman)
    
    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['dog_id', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('dog_id')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('dog_id')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'Fixed', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index
  
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, df, biomarker_columns


data, df, biomarker_columns = prepare_model_data(df, kalman = False)

W, L = linear_regression(data)
#W = (W + W.T) / 2
mu = estimate_mu(L, data['x_cov'])  
df_valid = data['df_valid']

z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map = get_z_variables(W, mu, df, biomarker_columns, 'dog_eigenvalues')
z_df, z_mu_df = get_z_variables(W, mu, df, biomarker_columns, plotname = None)
  
  
  
z_df = z_imputation(z_df, imputation_type = 'mean')
check_nan(z_df)

#plot_biomarker_heatmap(W, biomarker_columns, plotname = 'W Interaction Network')
plot_normal_heatmap(W, biomarker_columns, plotname = 'W Interaction Network')


biomarker_cols = list(biomarker_columns)
