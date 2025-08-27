import pandas as pd
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from collections import Counter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
#load data
current_dir = os.getcwd()

from scipy.stats import norm

def process_data(df, kalman=False, pval_thresh=0.001):
    df, biomarker_columns = preprocessing()

    outlier_counts = {}
    removed_values = []  # store removed rows

    # Identify and remove statistical outliers
    for biomarker in biomarker_columns:
        values = df[biomarker]
        mean = values.mean()
        std = values.std()

        if std == 0:
            continue  # Skip constant columns

        z_scores = (values - mean) / std
        p_values = 2 * norm.sf(abs(z_scores))  # two-tailed

        mask = (p_values < pval_thresh)
        num_removed = mask.sum()
        outlier_counts[biomarker] = num_removed

        if num_removed > 0:
            removed = df.loc[mask, ['dog_id', 'Age', biomarker]].copy()
            removed['biomarker'] = biomarker
            removed_values.append(removed)

            #print(f"[{biomarker}] Removed {num_removed} outlier(s) with p < {pval_thresh}")
            #print(removed)

        df.loc[mask, biomarker] = np.nan

    #print("\n===== Outlier Counts Per Biomarker =====")
    total_removed = 0
    for biomarker, count in outlier_counts.items():
        #print(f"{biomarker}: {count} removed")
        total_removed += count

    #print(f"\n🎯 Total outliers removed across all biomarkers: {total_removed}")

    # Combine all removed rows into one dataframe for plotting
    removed_df = pd.concat(removed_values) if removed_values else pd.DataFrame(columns=['dog_id', 'Age', biomarker, 'biomarker'])

    # Continue preprocessing
    df = df.sort_values(by=['dog_id', 'Age']).reset_index(level=0, drop=True)
    df = imputation(df, biomarker_columns, imputation_type="mean")

    df['dog_id'] = list(zip(df['dog_id'], df['Sex'], df['Fixed']))
    df['dog_id'] = df.groupby('dog_id').ngroup() + 1

    df = log_transform(df, list(biomarker_columns))
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df)

    return df, biomarker_columns




def prepare_model_data(df, kalman = False):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman, pval_thresh = 0.05)
    
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
mu = estimate_mu(L, data['x_cov'])  
df_valid = data['df_valid']
W = (W + W.T) / 2
plot_normal_heatmap(W, biomarker_columns, plotname = 'W Interaction Network')
print(W.diagonal())

from matplotlib.backends.backend_pdf import PdfPages
df_raw, biomarker_columns = preprocessing()
def plot_outlier_removal(df, removed_df, biomarker_columns, pdf_path='biomarker_outlier_plots_2.pdf'):
    with PdfPages(pdf_path) as pdf:
        for biomarker in biomarker_columns:
            plt.figure(figsize=(10, 6))

            # Plot all points in blue
            plt.scatter(df['Age'], df[biomarker], color='blue', label='Retained')

            # Plot removed outliers in red (if any)
            if not removed_df.empty:
                removed_subset = removed_df[removed_df['biomarker'] == biomarker]
                if not removed_subset.empty:
                    plt.scatter(removed_subset['Age'], removed_subset[biomarker], color='red', label='Removed', edgecolor='black')

            plt.title(f'{biomarker} over Age')
            plt.xlabel('Age')
            plt.ylabel(biomarker)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"✅ Outlier plots saved to: {pdf_path}")

#plot_outlier_removal(df = df_raw, removed_df = removed_df, biomarker_columns = biomarker_columns, pdf_path='biomarker_outlier_plots2.pdf')
