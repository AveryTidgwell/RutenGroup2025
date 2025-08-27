import pandas as pd
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
#load data
current_dir = os.getcwd()
file_path = os.path.abspath(os.path.join('/Users/summer/Downloads/dolphins-master', 'data', 'dolphin_data.csv'))
check_nan(df)
df = pd.read_csv(file_path, index_col=None, header = 4)

def mask_outliers_by_pvalue(df, biomarker_columns, p_thresh=0.01):
    """Mask biomarker values that are statistically unlikely under a normal distribution assumption."""
    z_thresh = norm.ppf(1 - p_thresh / 2)  # two-tailed test

    removed_counts = {}  # optional: track how many per biomarker

    for biomarker in biomarker_columns:
        z_vals = df[biomarker]

        # Identify extreme z-values
        outliers = z_vals.abs() > z_thresh
        removed_counts[biomarker] = outliers.sum()

        # Mask
        df.loc[outliers, biomarker] = np.nan

    total_removed = sum(removed_counts.values())
    print(f"Total outliers removed: {total_removed}")
    print("Top biomarkers by number of removed values:")
    print(pd.Series(removed_counts).sort_values(ascending=False).head(10))

    return df


def process_data(df, kalman = False):

    df = df[df['Age'] > 5] #remove baby dolphins
    df = remove_nan_data(df, 0.4) #remove columns that contain more than 40% missing data
    
    df = df.sort_values(by=['AnimalID', 'Age']).reset_index(level=0, drop=True)
    biomarker_columns = df.columns[4:]

    #count how many times each dolphin appears, drop ones with 1 appearances, plot histogram
    df = count_appearances(df, plot_histogram=False)

    for id in df['AnimalID'].unique():
        df.loc[df['AnimalID'] == id, biomarker_columns] = (
            df.loc[df['AnimalID'] == id, biomarker_columns].interpolate(method='linear', limit_direction='both'))

    #df = imputation(df, imputation_type = "mean")
    

    # Create a tuple of identifying fields
    df['AnimalID'] = list(zip(df['AnimalID'], df['Sex'], df['Species']))
    # Convert each unique tuple to a unique number
    df['AnimalID'] = df.groupby('AnimalID').ngroup()+1

    #do log transformation (original dolphin paper)
    df = log_transform(df) #to convert left skewed data to one that resembles normal distribution

    #normalize to mean 0 and std 1
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df) #convert sex to [0,1] , species to [0,1,2] and normalize age
    
    df = mask_outliers_by_pvalue(df, biomarker_columns, p_thresh=0.05)
    df = imputation(df, imputation_type = "mean")
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

def plot_normal_heatmap(W, biomarker_names, plotname='Heatmap Biomarkers'):
    plt.figure(figsize=(6, 6))
    cmap = 'bwr'

    # Normalize W to [-1, 1]
    W = W / np.abs(W).max()

    # Normalize colors to center at 0
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # Create main axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create divider to host colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Draw heatmap
    sns.heatmap(W, ax=ax, cbar=True, cbar_ax=cax, cmap=cmap, norm=norm,
                xticklabels=biomarker_names, yticklabels=biomarker_names)

    # Formatting
    ax.set(xlabel='', ylabel='')
    ax.set_aspect('equal')
    ax.set_title(plotname)
    ax.set_xlabel('Biomarkers')
    ax.set_ylabel('Biomarkers')
    plt.tight_layout()
    plt.show()

    # Save and cleanup
    #plt.savefig(os.path.join(current_dir, '..', 'figures', plotname + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
plot_normal_heatmap(W=W, biomarker_names=biomarker_columns, plotname='Normal Dolphin Heatmap')


