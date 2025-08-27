import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from imputation import imputation
from utils import prepare_model_data
from optimization import linear_regression, estimate_mu
from datetime import datetime
from tqdm import tqdm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
n_bootstraps = 250
quantile_lower = 0.02
quantile_upper = 0.98
kalman = False
output_pdf = 'z_bootstrap_mean_W_june_30.pdf'
data_path = '/Users/summer/Downloads/dolphins-master/data/dolphin_data.csv'

# ========== LOAD & CLEAN DATA ==========
df = pd.read_csv(data_path, index_col=None, header=4)
biomarker_columns = [col for col in df.columns if col not in ['AnimalID', 'Sex', 'Species', 'Age', 'Reason', 'Fasting', 'LabCode', 'Mg']]

for biomarker in biomarker_columns:
    q_low = df[biomarker].quantile(quantile_lower)
    q_high = df[biomarker].quantile(quantile_upper)
    df.loc[(df[biomarker] < q_low) | (df[biomarker] > q_high), biomarker] = np.nan

# ========== BASELINE TRANSFORMATION ==========
data0 = prepare_model_data(df, kalman=kalman)
W0, L0 = linear_regression(data0)
mu0 = estimate_mu(L0, data0['x_cov'])

# ========== BOOTSTRAP & W_MEAN ==========
z_df_list = []
z_mu_df_list = []
eigen_boot = []
W_list = []

print("Running bootstraps...")
for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
    df_boot = df.sample(n=len(df), replace=True, random_state=i).reset_index(drop=True)
    data = prepare_model_data(df_boot, kalman=kalman)
    W, L = linear_regression(data)
    mu = estimate_mu(L, data['x_cov'])

    try:
        eigvals, _ = np.linalg.eig(W)
        eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]
        eigen_boot.append(np.real(eigvals_sorted))

        W_list.append(W)

        # defer reprojection to after W_mean is computed
        z_df_list.append((data['df_valid'], mu))
    except:
        continue

# ========== MEAN W BASIS & PROJECTIONS ==========
W_mean = np.mean(np.array(W_list), axis=0)
eigvals_meanW, eigvecs_meanW = np.linalg.eig(W_mean)
abs_order = np.argsort(np.abs(eigvals_meanW))
V_sorted = eigvecs_meanW[:, abs_order]
P_inv_mean = np.linalg.inv(V_sorted)
sorted_eigenvalues = eigvals_meanW[abs_order]

n_z = P_inv_mean.shape[0]
z_col_rank_map = [f'z_{i+1}' for i in range(n_z)]
mu_col_rank_map = [f'mu_z_{i+1}' for i in range(n_z)]

def reproject(df_valid, mu, P_inv, biomarker_columns):
    z_data = np.matmul(P_inv, df_valid[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T
    z_df = pd.DataFrame(z_data.real, columns=[f'z_{i+1}' for i in range(P_inv.shape[0])])
    z_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values
    z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
    z_mu_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values
    return z_df, z_mu_df

z_df0, z_mu_df0 = reproject(data0['df_valid'], mu0, P_inv_mean, biomarker_columns)
z_df0 = imputation(z_df0, imputation_type='mean')

z_df_list_proj = []
z_mu_df_list_proj = []
for df_valid, mu in z_df_list:
    try:
        z_df_j, z_mu_df_j = reproject(df_valid, mu, P_inv_mean, biomarker_columns)
        z_df_list_proj.append(z_df_j)
        z_mu_df_list_proj.append(z_mu_df_j)
    except:
        continue

# ========== PLOTTING ==========
def plot_all_z_with_bootstrap(z_df0, z_mu_df0, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map,
                              z_df_list, z_mu_df_list, eigen_boot, output_pdf, W0, W_mean):
    os.makedirs('results/Smoother_z', exist_ok=True)
    pdf_path = os.path.join('results/Smoother_z', output_pdf)

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
            for df_boot, mu_boot in zip(z_mu_df_list, z_df_list):
                try:
                    df_b = df_boot[['Age', mu_col]].dropna().sort_values('Age')
                    if len(df_b) >= 2:
                        slope_j, intercept_j = np.polyfit(df_b['Age'], df_b[mu_col], deg=1)
                        yj = slope_j * x_pred + intercept_j
                        mu_boot_stack.append(yj)
                except:
                    continue
            if mu_boot_stack:
                mu_boot_stack = np.vstack(mu_boot_stack)
                y_mu_std = mu_boot_stack.std(axis=0)
                plt.fill_between(x_pred, y_mu0 - y_mu_std, y_mu0 + y_mu_std,
                                 color='red', alpha=0.3, label='±1 SD (Linear Fit)')

            lowess_stack = []
            for z_df in z_df_list:
                try:
                    df_b = z_df[['Age', z_col]].dropna().sort_values('Age')
                    if len(df_b) >= 2:
                        lowess_j = lowess(df_b[z_col], df_b['Age'], frac=0.2, return_sorted=True)
                        yj_interp = np.interp(x_lowess0, lowess_j[:, 0], lowess_j[:, 1])
                        lowess_stack.append(yj_interp)
                except:
                    continue
            if lowess_stack:
                lowess_stack = np.vstack(lowess_stack)
                y_lowess_std = lowess_stack.std(axis=0)
                plt.fill_between(x_lowess0, y_lowess0 - y_lowess_std, y_lowess0 + y_lowess_std,
                                 color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

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
        eigen_boot = np.vstack(eigen_boot)
        mean_eigs = np.mean(eigen_boot, axis=0)
        std_eigs = np.std(eigen_boot, axis=0)
        eigvals0 = np.linalg.eigvals(W0)
        eigvals0_sorted = eigvals0[np.argsort(np.abs(eigvals0))]
        eigvalsWmean_sorted = np.linalg.eigvals(W_mean)[np.argsort(np.abs(np.linalg.eigvals(W_mean)))]

        x = np.arange(1, len(eigvals0_sorted) + 1)

        plt.figure(figsize=(9, 6))
        plt.errorbar(x, eigvals0_sorted.real, yerr=std_eigs, fmt='o', capsize=3, color='black', label='Baseline (W0)')
        plt.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o', capsize=3, color='red', label='Bootstrap Mean')
        plt.plot(x, eigvalsWmean_sorted.real, 'o-', color='blue', label='Eigenvalues of Mean(W)')

        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.xlabel('Eigenvalue Rank')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues with ±1 SD Error')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"All plots (z and eigenvalues) saved to {pdf_path}")


plot_all_z_with_bootstrap(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    z_df_list=z_df_list_proj,
    z_mu_df_list=z_mu_df_list_proj,
    eigen_boot=eigen_boot,
    output_pdf=output_pdf,
    W0=W0,
    W_mean=W_mean
)

end_time = datetime.now()
elapsed = end_time - start_time
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")
