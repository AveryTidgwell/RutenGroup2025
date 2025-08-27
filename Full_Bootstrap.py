import pandas as pd
import numpy as np
import os
import warnings
import pickle
from imputation import imputation
from utils import *
from optimization import *
from pykalman import KalmanFilter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

def get_z_variables(W, mu, df, biomarker_columns, plotname = None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    print(np.real(eigenvalues))
    P_inv = np.linalg.inv(eigenvectors)
    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T

    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(len(biomarker_columns))]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(len(biomarker_columns))]
    lambda_names = [f'lambda_{i+1}' for i in range(len(biomarker_columns))]
    
    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_mu_df  = pd.DataFrame(z_mu.real, columns=natural_mu_names)


    #z_df = pd.concat([z_bio_df,z_mu_df],axis = 1)
    z_df[['AnimalID','Sex','Species','Age']] = df[['AnimalID','Sex','Species','Age']].copy()

    if plotname is not None:
        # Sort eigenvalues and corresponding biomarkers
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        sorted_biomarkers = [lambda_names[i] for i in np.argsort(eigenvalues)[::-1]]

        # Plot the sorted eigenvalues with biomarker names on the x-axis
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_biomarkers, y=sorted_eigenvalues, palette="viridis")
        plt.xlabel("Name")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W")
        plt.xticks(rotation=90)  # Rotate x labels if needed for better readability
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300, bbox_inches='tight')

    return z_df, z_mu_df

# ==========================
# Parameters
# ==========================
n_bootstraps = 250
quantile_lower = 0.02
quantile_upper = 0.98
kalman = False  # Set to True if needed
save_dir = 'bootstrap_results'  # Will save pickle files here

# ==========================
# Load and Clean Data
# ==========================
file_path = os.path.abspath(os.path.join('/Users/summer/Downloads/dolphins-master', 'data', 'dolphin_data.csv'))
df = pd.read_csv(file_path, index_col=None, header=4)

# Define biomarker_columns here if not already in scope
# biomarker_columns = [col for col in df.columns if ...] 

# Outlier removal using quantiles
for biomarker in biomarker_columns:
    lower = df[biomarker].quantile(quantile_lower)
    upper = df[biomarker].quantile(quantile_upper)
    df.loc[(df[biomarker] < lower) | (df[biomarker] > upper), biomarker] = np.nan

# ==========================
# Bootstrap Function
# ==========================
def process_bootstrap_iteration(df, biomarker_columns, iteration, kalman=False):
    boot_df = df.sample(n=len(df), replace=True, random_state=iteration).reset_index(drop=True)
    
    data = prepare_model_data(boot_df, kalman=kalman)
    W, L = linear_regression(data)
    mu = estimate_mu(L, data['x_cov'])

    z_df, z_mu_df = get_z_variables(W, mu, data['df_valid'], biomarker_columns, plotname=None)
    z_df = imputation(z_df, imputation_type='mean')

    # Add Age and AnimalID for plotting and alignment
    z_df['Age'] = data['df_valid']['Age'].values
    z_df['AnimalID'] = data['df_valid']['AnimalID'].values
    z_mu_df['Age'] = data['df_valid']['Age'].values
    z_mu_df['AnimalID'] = data['df_valid']['AnimalID'].values

    return mu, z_df, z_mu_df, W

# ==========================
# Run Bootstraps and Save
# ==========================
for i in range(n_bootstraps):
    # print(f"Processing bootstrap {i + 1} / {n_bootstraps}")
    mu_i, z_df_i, z_mu_df_i, W_i = process_bootstrap_iteration(df, biomarker_columns, i, kalman=kalman)

    # Save each file immediately
    with open(os.path.join(save_dir, f'mu_{i:04d}.pkl'), 'wb') as f:
        pickle.dump(mu_i, f)

    with open(os.path.join(save_dir, f'z_df_{i:04d}.pkl'), 'wb') as f:
        pickle.dump(z_df_i, f)

    with open(os.path.join(save_dir, f'z_mu_df_{i:04d}.pkl'), 'wb') as f:
        pickle.dump(z_mu_df_i, f)

    with open(os.path.join(save_dir, f'W_{i:04d}.pkl'), 'wb') as f:
        pickle.dump(W_i, f)
