import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
n_bootstraps = 200
kalman = False
output_pdf = f'july30_dog_bootstrap_({start_time}).pdf'

# ========== BASELINE TRANSFORMATION ==========
data0, df0, biomarker_columns = prepare_model_data(df, kalman=kalman)
W0, L0 = linear_regression(data0)
mu0 = estimate_mu(L0, data0['x_cov'])

eigenvalues0, eigenvectors0 = np.linalg.eig(W0)
abs_order = np.argsort(np.abs(eigenvalues0))
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
    z_df[['Age', 'dog_id']] = df_valid[['Age', 'dog_id']].values

    z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
    z_mu_df[['Age', 'dog_id']] = df_valid[['Age', 'dog_id']].values

    return z_df, z_mu_df

z_df0, z_mu_df0 = reproject_to_W0_basis(data0['df_valid'], mu0, P_inv0, biomarker_columns)
z_df0 = z_imputation(z_df0, imputation_type='mean')

z_df_list = []
z_mu_df_list = []
eigen_boot = []


print("Running bootstraps by dog_id...")
unique_ids = df['dog_id'].unique()

for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
    sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
    df_boot = pd.concat([df[df['dog_id'] == aid] for aid in sampled_ids], ignore_index=True)
    
    data, _, _ = prepare_model_data(df_boot, kalman=kalman)
    try:
        W, L = linear_regression(data)
        mu = estimate_mu(L, data['x_cov'])

        eigvals, _ = np.linalg.eig(W)
        eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]
        eigen_boot.append(np.real(eigvals_sorted))

        z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)
        z_df_list.append(z_df_j)
        z_mu_df_list.append(z_mu_df_j)
    except Exception as e:
        print(f"Bootstrap {i} failed: {e}")
        continue


# ========== PLOTTING ==========
def plot_all_z_with_bootstrap(z_df0, z_mu_df0, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map,
                              z_df_list, z_mu_df_list, eigen_boot, output_pdf, W0):
    os.makedirs('results/Smoother_z', exist_ok=True)
    pdf_path = os.path.join(os.getcwd(), output_pdf)

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
            for df_boot in z_mu_df_list:
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

        eigen_boot = np.vstack(eigen_boot)
        mean_eigs = np.mean(eigen_boot, axis=0)
        std_eigs = np.std(eigen_boot, axis=0)
        eigvals0 = np.linalg.eigvals(W0)
        eigvals0_sorted = eigvals0[np.argsort(np.abs(eigvals0))]

        x = np.arange(1, len(eigvals0_sorted) + 1)

        plt.figure(figsize=(9, 6))
        plt.errorbar(x, eigvals0_sorted.real, yerr=std_eigs, fmt='o', capsize=3, color='black', label='Baseline ±1 SD')
        plt.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o', capsize=3, color='red', label='Bootstrap Mean ±1 SD')
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.xlabel('Eigenvalue Rank')
        plt.ylabel('Eigenvalue')
        plt.xticks(np.arange(0,len(eigvals0_sorted) + 1, 1))
        plt.title('Eigenvalues of W with ±1 SD Error from Bootstrap')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        
        plt.figure(figsize=(10, 6))

        mean_drifts = drift_matrix.mean(axis=0)
        std_drifts = drift_matrix.std(axis=0)

        plt.errorbar(x, mean_drifts, yerr=std_drifts, fmt=div_markers[idx], color=div_colors[idx], 
                    label=f'd={d} Mean ±1SD', alpha=0.8, capsize=3, markersize=ms, linewidth=lw)
        plt.plot(x, mean_drifts, linestyle='-', color=div_colors[idx], alpha=0.8, markersize=ms, linewidth=lw)
        
        baseline_drifts = []
        for col in mu_col_rank_map:
            smooth = lowess(z_mu_df0[col], z_mu_df0['Age'], frac=0.2, return_sorted=True)
            slope = np.polyfit(smooth[:, 0], smooth[:, 1], 1)[0]
            baseline_drifts.append(slope)

    plt.plot(x, baseline_drifts, linestyle='-', marker=div_markers[idx], mfc='none', 
            color='black', label='Baseline')

    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Latent Dimension Rank')
    plt.ylabel('Drift Slope')
    plt.title('Combined Drift Slope Comparison Across Sample Frequency')
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close()
    print(f"All plots saved to {pdf_path}")

    print(f"All plots (z and eigenvalues) saved to {pdf_path}")


plot_all_z_with_bootstrap(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    z_df_list=z_df_list,
    z_mu_df_list=z_mu_df_list,
    eigen_boot=eigen_boot,
    output_pdf=output_pdf,
    W0=W0
)

end_time = datetime.now()
elapsed = end_time - start_time
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")


