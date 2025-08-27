import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels

def get_z_variables(W, mu, df, biomarker_columns, plotname=None):
    eigenvalues, eigenvectors = np.linalg.eig(W)

    # Sort eigenvalues by absolute value (least stable to most stable)
    abs_eigenvalue_order = np.argsort(np.abs(eigenvalues))  # indices from least to most stable

    # Reorder eigenvalues and eigenvectors accordingly
    sorted_eigenvalues = eigenvalues[abs_eigenvalue_order]
    sorted_eigenvectors = eigenvectors[:, abs_eigenvalue_order]

    # Rename based on stability order
    natural_var_names = [f'z_{i+1}' for i in range(len(biomarker_columns))]  # z_1 = least stable
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(len(biomarker_columns))]
    lambda_names = [f'lambda_{i+1}' for i in range(len(biomarker_columns))]

    # Get inverse of reordered eigenvectors
    P_inv = np.linalg.inv(sorted_eigenvectors)

    # Transform the biomarker data
    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    # Build dataframes
    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_bio_df = imputation(z_bio_df, imputation_type='mean')
    z_mu_df  = pd.DataFrame(z_mu.real, columns=natural_mu_names)

    # Add covariates
    z_df = pd.concat([z_bio_df, z_mu_df], axis=1)
    z_df[['AnimalID','Sex','Species','Age']] = df[['AnimalID','Sex','Species','Age']].copy()

    if plotname is not None:
        # Plot sorted eigenvalues
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=lambda_names, y=sorted_eigenvalues.real, s = 50, c="lightseagreen", label='Eigenvalues')
        smoothed = lowess(sorted_eigenvalues.real, np.arange(0, len(sorted_eigenvalues)), frac=0.3)
        #plt.plot(smoothed[:, 0], smoothed[:, 1], color='aquamarine', label='Eigenvalues')
        W_df = pd.DataFrame(W)
        diag_values = np.diag(W_df)
        diag_series = pd.Series(diag_values, index=W_df.index)
        sorted_labels = diag_series.sort_values(ascending=False).index
        W_df_sorted = W_df.loc[sorted_labels, sorted_labels]
        sns.scatterplot(x=range(len(W_df_sorted)), y=np.diag(W_df_sorted), s = 50, c='orangered', label='W Diagonal')
        smoothed = lowess(np.diag(W_df_sorted), np.arange(0, len(np.diag(W_df_sorted))), frac=0.3)
        #plt.plot(smoothed[:, 0], smoothed[:, 1], color='darkorange', label='W Diagonals')
        
        
        plt.legend()
        plt.xlabel("Name")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W (by stability)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return z_df, z_mu_df
z_df, z_mu_df = get_z_variables(W, mu, data['df_valid'], biomarker_columns, plotname = 'get_z_variables')
z_df = imputation(z_df, imputation_type = 'mean')
check_nan(z_df)


def get_z_variables(W, mu, df, biomarker_columns, plotname=None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    abs_eigenvalue_order = np.argsort(np.abs(eigenvalues))  # least to most stable

    sorted_eigenvalues = eigenvalues[abs_eigenvalue_order]
    sorted_eigenvectors = eigenvectors[:, abs_eigenvalue_order]
    P_inv = np.linalg.inv(sorted_eigenvectors)

    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(P_inv.shape[0])]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(P_inv.shape[0])]
    lambda_names = [f'lambda_{i+1}' for i in range(P_inv.shape[0])]

    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_bio_df = imputation(z_bio_df, imputation_type='mean')
    z_mu_df = pd.DataFrame(z_mu.real, columns=natural_mu_names)

    z_df = pd.concat([z_bio_df, z_mu_df], axis=1)
    z_df[['AnimalID', 'Sex', 'Species', 'Age']] = df[['AnimalID', 'Sex', 'Species', 'Age']].copy()

    # Rank map: for each stability rank i, what original z_col was it?
    # If abs_eigenvalue_order[0] == 3, then z_4 is least stable and should be plotted first
    z_col_rank_map = [f'z_{i+1}' for i in abs_eigenvalue_order]
    mu_col_rank_map = [f'mu_z_{i+1}' for i in abs_eigenvalue_order]

    biomarker_weights = pd.DataFrame(
        P_inv.real,
        columns=biomarker_columns,
        index=natural_var_names
    )

    if plotname is not None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=lambda_names, y=sorted_eigenvalues.real, marker='o', color="lightseagreen")
        plt.xlabel("Natural Variable")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W (by stability)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300)
        plt.show()

    return z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map





z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map = get_z_variables(W, mu, df_valid, biomarker_columns, 'EigenStuff.png')




