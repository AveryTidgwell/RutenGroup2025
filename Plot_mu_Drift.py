import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_mu_slopes_with_error(P_inv0, biomarker_columns, bootstrap_dir, n_bootstraps,
                               mu_col_rank_map, z_mu_df0):
    n_z = P_inv0.shape[0]
    slope_matrix = []

    for j in range(n_bootstraps):
        try:
            with open(os.path.join(bootstrap_dir, f'df_valid_{j:04d}.pkl'), 'rb') as f:
                df_valid_j = pickle.load(f)
            with open(os.path.join(bootstrap_dir, f'mu_{j:04d}.pkl'), 'rb') as f:
                mu_j = pickle.load(f)

            # Reproject
            _, z_mu_df_j = reproject_to_W0_basis(df_valid_j, mu_j, P_inv0, biomarker_columns)

            # Estimate slopes for each natural variable
            slopes = []
            for mu_col in mu_col_rank_map:
                df_mu = z_mu_df_j[['Age', mu_col]].dropna()
                if len(df_mu) >= 2:
                    slope, _ = np.polyfit(df_mu['Age'], df_mu[mu_col], deg=1)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            slope_matrix.append(slopes)

        except Exception as e:
            print(f"Bootstrap {j:04d} failed: {e}")
            continue

    slope_matrix = np.array(slope_matrix)
    mean_slopes = np.nanmean(slope_matrix, axis=0)
    std_slopes = np.nanstd(slope_matrix, axis=0)

    # === Baseline slopes ===
    baseline_slopes = []
    for mu_col in mu_col_rank_map:
        df_mu0 = z_mu_df0[['Age', mu_col]].dropna()
        if len(df_mu0) >= 2:
            slope, _ = np.polyfit(df_mu0['Age'], df_mu0[mu_col], deg=1)
            baseline_slopes.append(slope)
        else:
            baseline_slopes.append(np.nan)

    baseline_slopes = np.array(baseline_slopes)

    # === Plot ===
    x = np.arange(1, n_z + 1)
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, mean_slopes, yerr=std_slopes, fmt='o', capsize=3, color='black', label='Bootstrap Mean ±1 SD')
    plt.errorbar(x, baseline_slopes, yerr=std_slopes, color='red', fmt='o', capsize=3, alpha = 0.7, label='Baseline Slope')
    plt.axhline(0, linestyle='--', color='gray', lw=1)
    plt.xlabel('Natural Variable z_i Ranked by Increasing Stability')
    plt.ylabel('Slope of mu_z_i vs Age')
    plt.title('Linear Drift of mu')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.clf()

    return mean_slopes, std_slopes, baseline_slopes


mean_slopes, std_slopes, baseline_slopes = plot_mu_slopes_with_error(
    P_inv0=P_inv0,
    biomarker_columns=biomarker_columns,
    bootstrap_dir=save_dir,
    n_bootstraps=n_bootstraps,
    mu_col_rank_map=mu_col_rank_map,
    z_mu_df0=z_mu_df0
)
