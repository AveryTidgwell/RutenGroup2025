import pandas as pd
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from imputation import imputation
from utils import *
from optimization import *
from pykalman import KalmanFilter


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
#load data
current_dir = os.getcwd()
file_path = os.path.abspath(os.path.join('/Users/summer/Downloads/dolphins-master', 'data', 'dolphin_data.csv'))
check_nan(df)
df = pd.read_csv(file_path, index_col=None, header = 4)
for biomarker in biomarker_columns:
    lower = df[biomarker].quantile(0.02)
    upper = df[biomarker].quantile(0.98)
    
    # Set outliers to NaN directly using boolean indexing
    df.loc[(df[biomarker] < lower) | (df[biomarker] > upper), biomarker] = np.nan

def process_data(df, kalman = False):

    df = df[df['Age'] > 5] #remove baby dolphins
    df = remove_nan_data(df, 0.4) #remove columns that contain more than 40% missing data
    
    df = df.sort_values(by=['AnimalID', 'Age']).reset_index(level=0, drop=True)
    biomarker_columns = df.columns[4:]

    #count how many times each dolphin appears, drop ones with 1 appearances, plot histogram
    df = count_appearances(df, plot_histogram=False)

    #fill missing values with mean
    #df = imputation(df,imputation_type="mean")

    #check_nan(df)

    for id in df['AnimalID'].unique():
        df.loc[df['AnimalID'] == id, biomarker_columns] = (
            df.loc[df['AnimalID'] == id, biomarker_columns].interpolate(method='linear', limit_direction='both'))

    df = imputation(df, imputation_type = "mean")
    #check_nan(df)

    # Create a tuple of identifying fields
    df['AnimalID'] = list(zip(df['AnimalID'], df['Sex'], df['Species']))
    # Convert each unique tuple to a unique number
    df['AnimalID'] = df.groupby('AnimalID').ngroup()+1

    #do log transformation (original dolphin paper)
    df = log_transform(df) #to convert left skewed data to one that resembles normal distribution

    #normalize to mean 0 and std 1
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df) #convert sex to [0,1] , species to [0,1,2] and normalize age
    
    #W only for F[0] or M[1]
    df = df[df['Sex'] == 1]
    return df, biomarker_columns

def prepare_model_data(df, kalman = False):
    df, biomarker_columns = process_data(df, kalman)

    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['AnimalID', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('AnimalID')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('AnimalID')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'Species', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index
  
    # Filter outliers
    for biomarker in biomarker_columns:
        lower_percentile = df[biomarker].quantile(0.02)
        upper_percentile = df[biomarker].quantile(0.98)

        df_filtered = df[
            (df[biomarker] >= lower_percentile) &
            (df[biomarker] <= upper_percentile)
        ]


    
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, biomarker_columns, df


m_data, biomarker_columns, m_df = prepare_model_data(df, kalman = False)
m_W, m_L = linear_regression(m_data)


plot_biomarker_heatmap(m_W, biomarker_columns, plotname = 'Male LM')



diff_W = np.subtract(f_W, m_W)

plot_biomarker_heatmap(m_W, biomarker_columns, plotname = 'Male Interation Network')

import pandas as pd
import numpy as np

def top_diff_interactions(diff_W, biomarker_columns, top_n=10):
    n = len(biomarker_columns)
    
    interactions = []

    for i in range(n):
        for j in range(n):
            #if i == j:
                #continue  # optionally skip self-interactions
            value = diff_W[i, j]
            interactions.append({
                'from': biomarker_columns[i],
                'to': biomarker_columns[j],
                'diff': value,
                'abs_diff': abs(value)
            })

    # Convert to DataFrame for easy sorting
    df_diff = pd.DataFrame(interactions)

    # Get top and bottom differences
    top_diffs = df_diff.sort_values(by='abs_diff', ascending=False).head(top_n)
    bottom_diffs = df_diff.sort_values(by='abs_diff', ascending=True).head(top_n)

    return top_diffs, bottom_diffs

top_diffs, bottom_diffs = top_diff_interactions(diff_W, biomarker_columns, top_n=10)
print("Top Differences (Sex-specific Interactions):")
print(top_diffs)

print("Most Similar Interactions:")
print(bottom_diffs)







