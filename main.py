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
    
    df.loc[(df[biomarker] < lower) | (df[biomarker] > upper), biomarker] = np.nan

def process_data(df, kalman = False):

    df = df[df['Age'] > 5] #remove baby dolphins
    df = remove_nan_data(df, 0.4) #remove columns that contain more than 40% missing data
    
    df = df.sort_values(by=['AnimalID', 'Age']).reset_index(level=0, drop=True)
    biomarker_columns = df.columns[4:]

    #count how many times each dolphin appears, drop ones with 1 appearances, plot histogram
    df = df[df['AnimalID'].map(df['AnimalID'].value_counts()) > 1]
    
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

    return df, biomarker_columns

def prepare_model_data(df, kalman = False):
    df = df.copy()
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
    #for biomarker in biomarker_columns:
        #lower_percentile = df[biomarker].quantile(0.02)
        #upper_percentile = df[biomarker].quantile(0.98)

        #df_filtered = df[
            #(df[biomarker] >= lower_percentile) &
            #(df[biomarker] <= upper_percentile)
        #]


    
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
mu = estimate_mu(L, data['x_cov'])  
df_valid = data['df_valid']
z_df, z_mu_df = get_z_variables(W, mu, data['df_valid'], biomarker_columns, plotname = 'get_z_variables')
z_df = imputation(z_df, imputation_type = 'mean')
check_nan(z_df)

plot_biomarker_heatmap(W, biomarker_columns, plotname = 'W Interaction Network')
#calculate_loss(data, W, L)
print_prediction(data, W, L,biomarker_columns, id=4, biomarker = 2, filename = 'LM_pred.png')

z_df, z_mu_df = get_z_variables(W, estimate_mu(L,data['x_cov']), data['df_valid'], biomarker_columns, plotname = 'eigenvalues.png')

plot_mean_bio(data, L,biomarker_columns, 'Iron', data['x_cov'], plotname = 'Iron.png')
print(len(W))
























##########################

def get_z_variables(W, mu, df, biomarker_columns, plotname=None, debug=False):
    """
    Returns:
      z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map
    Ensures df is positional-aligned with z matrices (resets index).
    """
    df = df.reset_index(drop=True)  # before calling get_z_variables

    # --- eigen / basis stuff (unchanged) ---
    eigenvalues, eigenvectors = np.linalg.eig(W)
    real_eigenvalue_order = np.argsort(-eigenvalues.real)

    sorted_eigenvalues = eigenvalues[real_eigenvalue_order]
    sorted_eigenvectors = eigenvectors[:, real_eigenvalue_order]
    P_inv = np.linalg.inv(sorted_eigenvectors)

    # --- make a positional copy of df so that we avoid index alignment issues ---
    df_pos = df.reset_index(drop=True).copy()   # <--- important fix

    # compute z's from biomarker columns (use .to_numpy() for speed / no alignment)
    z_biomarkers = np.matmul(P_inv, df_pos[biomarker_columns].to_numpy().T).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(P_inv.shape[0])]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(P_inv.shape[0])]
    lambda_names = [f'lambda_{i+1}' for i in range(P_inv.shape[0])]

    # Create dataframes (they will have 0..N-1 index)
    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    #z_bio_df = z_imputation(z_bio_df, imputation_type='mean')   # if you want this here
    z_mu_df = pd.DataFrame(z_mu.real, columns=natural_mu_names)

    # Concatenate positionally (no index alignment surprises)
    z_df = pd.concat([z_bio_df.reset_index(drop=True),
                      df_pos[['AnimalID', 'Sex', 'Species', 'Age']].reset_index(drop=True)],
                     axis=1)

    # Rank maps
    z_col_rank_map = [f'z_{i+1}' for i in real_eigenvalue_order]
    mu_col_rank_map = [f'mu_z_{i+1}' for i in real_eigenvalue_order]

    biomarker_weights = pd.DataFrame(
        P_inv.real,
        columns=biomarker_columns,
        index=natural_var_names
    )

    # optional debug checks
    if debug:
        # check for any NaNs in the covariates we just attached
        cov_na = z_df[['AnimalID', 'Sex', 'Species', 'Age']].isnull().any(axis=1)
        print(f"DEBUG: total rows: {len(z_df)}")
        print(f"DEBUG: rows with any covariate NA: {cov_na.sum()}")
        if cov_na.any():
            print("DEBUG: indices with covariate NA:", z_df.index[cov_na].tolist())
            # Show a sample of z rows with missing covariates
            display(z_df.loc[cov_na, z_df.columns[:10]])  # show first cols for context

    # plotting (unchanged, except using df_pos if needed)
    if plotname is not None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=lambda_names, y=sorted_eigenvalues.real, marker='o', color="lightseagreen")
        plt.xlabel("Natural Variable")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W (by stability)")
        plt.xticks(rotation=90)
        plt.tight_layout()

        save_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, plotname + '.png'), dpi=300)
        plt.show()

    return z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map

