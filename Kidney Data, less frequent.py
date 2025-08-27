import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm  # Progress bar

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")


def prepare_spaced_data(df, kalman = False, save_path = None):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman)
    
    df = thin_by_fixed_spacing(df)
    
    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['Record ID', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('Record ID')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('Record ID')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'DM Type', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index

    # Now filter all arrays using cleaned valid_rows
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to: {save_path}")
    
    
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, df

# ========== BASELINE TRANSFORMATION ==========
def thin_by_fixed_spacing(df):
    spacing_years = 4 / 365.25
    thinned_rows = []

    for _, group in df.groupby('Record ID'):
        group = group.sort_values('Age')
        current_time = group['Age'].iloc[0]
        thinned_rows.append(group.iloc[0])

        for _, row in group.iterrows():
            if row['Age'] >= current_time + spacing_years:
                thinned_rows.append(row)
                current_time = row['Age']

    return pd.DataFrame(thinned_rows).reset_index(drop=True)

os.getcwd()

# ========== CONFIGURATION ==========
n_bootstraps = 200
kalman = False
output_pdf = '4_day_spacing_kidney.pdf'
data_path = '/Volumes/Health Files/'
file_path = os.path.abspath(os.path.join('/Volumes/Health Files/', 'TransplantDataRepo-PhysicsValidation_DATA_LABELS_2024-11-13_1619.csv'))

# ========== LOAD & CLEAN DATA ==========
df = pd.read_csv(file_path, index_col=None)

# ========== BASELINE TRANSFORMATION ==========
data0, df0 = prepare_spaced_data(df, kalman=kalman)
W0, L0 = linear_regression(data0)
mu0 = estimate_mu(L0, data0['x_cov'])

eigenvalues0, eigenvectors0 = np.linalg.eig(W0)
abs_order = np.argsort(-eigenvalues0.real)
V_sorted = eigenvectors0[:, abs_order]
P_inv0 = np.linalg.inv(V_sorted)
sorted_eigenvalues = eigenvalues0[abs_order]

n_z = P_inv0.shape[0]
z_col_rank_map = [f'z_{i+1}' for i in range(n_z)]
mu_col_rank_map = [f'mu_z_{i+1}' for i in range(n_z)]

def reproject_to_W0_basis(df_valid, mu, P_inv, biomarker_columns):
    z_data = np.matmul(P_inv, df_valid[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T
    z_df = pd.DataFrame(z_data.real, columns=[f'z_{i+1}' for i in range(P_inv.shape[0])])
    z_df[['Age', 'Record ID']] = df_valid[['Age', 'Record ID']].values
    z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
    z_mu_df[['Age', 'Record ID']] = df_valid[['Age', 'Record ID']].values
    return z_df, z_mu_df

z_df0, z_mu_df0 = reproject_to_W0_basis(data0['df_valid'], mu0, P_inv0, biomarker_columns)
z_df0 = imputation(z_df0, imputation_type='mean')

# ========== BOOTSTRAP: IN-MEMORY STORAGE ONLY ==========
z_df_list = []
z_mu_df_list = []
eigen_boot = []

print("Running bootstraps...")
unique_ids = df['Record ID'].unique()

for i in tqdm(range(n_bootstraps), desc="Half-population subsampling"):
    np.random.seed(i)
    sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
    df_half = df[df['Record ID'].isin(sampled_ids)].copy().reset_index(drop=True)
    data, df_half_cleaned = prepare_model_data(df_half, kalman=kalman)
    W, L = linear_regression(data)
    mu = estimate_mu(L, data['x_cov'])

    try:
        eigvals, _ = np.linalg.eig(W)
        eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
        eigen_boot.append(np.real(eigvals_sorted))

        z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)
        z_df_list.append(z_df_j)
        z_mu_df_list.append(z_mu_df_j)
    except:
        continue



# ========== PLOTTING ==========
def plot_all_z_with_bootstrap(z_df0, z_mu_df0, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map,
                              z_df_list, z_mu_df_list, eigen_boot, P_inv0, biomarker_columns,
                              output_pdf, W0):
    os.makedirs('Volumes/Health Files/censoring_results', exist_ok=True)
    pdf_path = os.path.join('Volumes/Health Files/censoring_results', output_pdf)

    with PdfPages(pdf_path) as pdf:
        for i, (z_col, mu_col) in enumerate(zip(z_col_rank_map, mu_col_rank_map)):
            plt.figure(figsize=(9, 8))
            t = z_df0['Age']
            z_vals = z_df0[z_col]
            mu_vals = z_mu_df0[mu_col]
            x_pred = np.linspace(t.min(), t.max(), 200)

            df_lowess0 = pd.DataFrame({'Age': t, 'z': z_vals}).dropna()
            lowess0 = lowess(df_lowess0['z'], df_lowess0['Age'], frac=0.2, return_sorted=True)
            x_lowess0, y_lowess0 = lowess0[:, 0], lowess0[:, 1]
            plt.plot(x_lowess0, y_lowess0, color='magenta', lw=2.5, label='Original LOWESS')

            df_mu0 = pd.DataFrame({'Age': t, 'mu': mu_vals}).dropna()
            if len(df_mu0) >= 2:
                slope, intercept = np.polyfit(df_mu0['Age'], df_mu0['mu'], deg=1)
                y_mu0 = slope * x_pred + intercept
                plt.plot(x_pred, y_mu0, '--', lw=2, color='red', label='Original Linear Fit')

            mu_boot_stack = []
            lowess_stack = []
            for z_df_j, z_mu_df_j in zip(z_df_list, z_mu_df_list):
                try:
                    df_mu = z_mu_df_j[['Age', mu_col]].dropna().sort_values('Age')
                    if len(df_mu) >= 2:
                        slope_j, intercept_j = np.polyfit(df_mu['Age'], df_mu[mu_col], deg=1)
                        yj = slope_j * x_pred + intercept_j
                        mu_boot_stack.append(yj)

                    df_z = z_df_j[['Age', z_col]].dropna().sort_values('Age')
                    if len(df_z) >= 2:
                        lowess_j = lowess(df_z[z_col], df_z['Age'], frac=0.2, return_sorted=True)
                        yj_interp = np.interp(x_lowess0, lowess_j[:, 0], lowess_j[:, 1])
                        lowess_stack.append(yj_interp)
                except:
                    continue

            if mu_boot_stack:
                mu_boot_stack = np.vstack(mu_boot_stack)
                y_mu_std = mu_boot_stack.std(axis=0)
                plt.fill_between(x_pred, y_mu0 - y_mu_std, y_mu0 + y_mu_std, color='red', alpha=0.3, label='±1 SD (Linear Fit)')

            if lowess_stack:
                lowess_stack = np.vstack(lowess_stack)
                y_lowess_std = lowess_stack.std(axis=0)
                plt.fill_between(x_lowess0, y_lowess0 - y_lowess_std, y_lowess0 + y_lowess_std, color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

            plt.scatter(t, z_vals, color='gray', alpha=0.1, s=10, label='All Data')
            plt.ylim(-5, 5)
            plt.xlabel('Age (yrs)')
            plt.ylabel(f'Natural Variable: z_{i+1}')
            plt.title(f'z_{i+1}')
            plt.legend(loc='best', fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # ========== Final eigenvalue plot ==========
        if eigen_boot:
            eigen_boot = np.vstack(eigen_boot)
        else:
            print("No eigenvalues were collected during bootstrapping!")
            return

        mean_eigs = np.mean(eigen_boot, axis=0)
        std_eigs = np.std(eigen_boot, axis=0)
        eigvals0 = np.linalg.eigvals(W0)
        eigvals0_sorted = eigvals0[np.argsort(-eigvals0.real)]

        x = np.arange(1, len(eigvals0_sorted) + 1)
        plt.figure(figsize=(9, 6))
        plt.errorbar(x, eigvals0_sorted.real, yerr=std_eigs, fmt='o', capsize=3, color='black', label='Baseline ±1 SD')
        plt.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o', capsize=3, color='red', label='Bootstrap Mean ±1 SD')
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.xlabel('Eigenvalue Rank')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of W with ±1 SD Error from Bootstrap')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"All plots (z and eigenvalues) saved to {pdf_path}")

# Call the plotting function
plot_all_z_with_bootstrap(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    z_df_list=z_df_list,
    z_mu_df_list=z_mu_df_list,
    eigen_boot=eigen_boot,
    P_inv0=P_inv0,
    biomarker_columns=biomarker_columns,
    output_pdf=output_pdf,
    W0=W0
)

end_time = datetime.now()
elapsed = end_time - start_time
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")
