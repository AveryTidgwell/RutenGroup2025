def plot_all_z_with_bootstrap_streamed(
    z_df0,
    z_mu_df0,
    sorted_eigenvalues,
    z_col_rank_map,
    mu_col_rank_map,
    bootstrap_dir,
    n_bootstraps,
    output_pdf='z_ranked_properly.pdf'
):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import os
    import pickle
    import pandas as pd

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'Smoother_z')
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, output_pdf)

    with PdfPages(pdf_path) as pdf:
        for i, (z_col, mu_col) in enumerate(zip(z_col_rank_map, mu_col_rank_map)):
            plt.figure(figsize=(9, 8))

            t = z_df0['Age']
            z_vals = z_df0[z_col]
            mu_vals = z_mu_df0[mu_col]

            x_pred = np.linspace(t.min(), t.max(), 200)

            # === Plot original LOWESS ===
            df_lowess0 = pd.DataFrame({'Age': t, 'z': z_vals}).dropna().sort_values('Age')
            lowess0 = lowess(df_lowess0['z'], df_lowess0['Age'], frac=0.3, return_sorted=True)
            x_lowess0 = lowess0[:, 0]
            y_lowess0 = lowess0[:, 1]
            plt.plot(x_lowess0, y_lowess0, color='magenta', linestyle='-', linewidth=2.5, label='Original LOWESS')

            # === Plot original mu (linear fit using polyfit) ===
            df_mu0 = pd.DataFrame({'Age': t, 'mu': mu_vals}).dropna().sort_values('Age')
            if len(df_mu0) >= 2:
                # Fit a straight line: mu ≈ slope * Age + intercept
                slope0, intercept0 = np.polyfit(df_mu0['Age'], df_mu0['mu'], deg=1)
                y_mu0 = slope0 * x_pred + intercept0
                plt.plot(x_pred, y_mu0, linestyle='--', linewidth=2, color='red', label='Original Linear Fit')

                # === Bootstrap slope/intercept SD for mu ===
                slopes, intercepts = [], []
                for j in range(n_bootstraps):
                    try:
                        with open(os.path.join(bootstrap_dir, f'z_mu_df_{j:04d}.pkl'), 'rb') as f:
                            z_mu_df_j = pickle.load(f)
                        df_boot = pd.DataFrame({'Age': z_mu_df_j['Age'], 'mu': z_mu_df_j[mu_col]}).dropna().sort_values('Age')
                        if len(df_boot) >= 2:
                            s, i_ = np.polyfit(df_boot['Age'], df_boot['mu'], deg=1)
                            slopes.append(s)
                            intercepts.append(i_)
                    except Exception:
                        continue

                if slopes and intercepts:
                    slope_sd = np.std(slopes)
                    intercept_sd = np.std(intercepts)

                    y_upper = (slope0 + slope_sd) * x_pred + (intercept0 + intercept_sd)
                    y_lower = (slope0 - slope_sd) * x_pred + (intercept0 - intercept_sd)

                    plt.fill_between(x_pred, y_lower, y_upper, color='red', alpha=0.3, label='±1 SD (Linear Fit)')
            else:
                print(f"[z_{i+1}] Not enough points to fit linear trend.")

            # === Bootstrap SD for LOWESS ===
            lowess_bootstrap_stack = []
            for j in range(n_bootstraps):
                try:
                    with open(os.path.join(bootstrap_dir, f'z_df_{j:04d}.pkl'), 'rb') as f:
                        z_df_j = pickle.load(f)
                    df_boot = z_df_j[['Age', z_col]].dropna().sort_values('Age')
                    if len(df_boot) >= 2:
                        lowess_j = lowess(df_boot[z_col], df_boot['Age'], frac=0.3, return_sorted=True)
                        yj_interp = np.interp(x_lowess0, lowess_j[:, 0], lowess_j[:, 1])
                        lowess_bootstrap_stack.append(yj_interp)
                except Exception:
                    continue

            if lowess_bootstrap_stack:
                lowess_bootstrap_stack = np.vstack(lowess_bootstrap_stack)
                y_lowess_std = lowess_bootstrap_stack.std(axis=0)
                plt.fill_between(x_lowess0, y_lowess0 - y_lowess_std, y_lowess0 + y_lowess_std,
                                 color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

            # === Plot settings ===
            eigenvalue = np.real(sorted_eigenvalues[i])
            if eigenvalue != 0:
                recovery_time = -1 / eigenvalue
                marker_positions = np.arange(t.min(), t.max(), recovery_time)
                plt.plot(marker_positions, [-4.5]*len(marker_positions), linestyle='None', marker='x',
                         color='black', label=f'Auto-Corr Time: {recovery_time:.1f} yrs')

            plt.scatter(t, z_vals, color='blue', alpha=0.1, s=10, label='All Data')
            plt.ylim(-5, 5)
            plt.grid(True)
            plt.xlabel('Age (yrs)')
            plt.ylabel(f'Natural Variable: z_{i+1}')
            plt.title(f'z_{i+1}')
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()

            if i == 0:
                pdf.savefig()
                plt.show()  # Show the first plot for error detection
            else:
                pdf.savefig()
                plt.close()

    print(f"All plots saved to {pdf_path}")




def sort_W(W, z_df, z_mu_df):
    import numpy as np

    eigenvalues, eigenvectors = np.linalg.eig(W)
    sort_idx = np.argsort(-np.real(eigenvalues))
    sorted_eigenvalues = eigenvalues[sort_idx]

    # Just use first 43 columns assuming they are ordered correctly
    z_cols = z_df.columns[:43].tolist()
    mu_cols = z_mu_df.columns[:43].tolist()

    z_col_rank_map = [z_cols[i] for i in sort_idx]
    mu_col_rank_map = [mu_cols[i] for i in sort_idx]

    return sorted_eigenvalues, z_col_rank_map, mu_col_rank_map


import pickle
import os
import numpy as np

# Path to the directory with saved z_mu_df_XXXX.pkl files
bootstrap_dir = 'bootstrap_results'
n_bootstraps = 250  # Set this to your actual number

slopes = []
intercepts = []

for i in range(n_bootstraps):
    with open(os.path.join(bootstrap_dir, f'z_mu_df_{i:04d}.pkl'), 'rb') as f:
        z_mu_df_i = pickle.load(f)

    # Fit a line to this bootstrap sample
    df = z_mu_df_i[['Age', 'z_mu']].dropna()
    if len(df) >= 2:
        slope, intercept = np.polyfit(df['Age'], df['z_mu'], deg=1)
        slopes.append(slope)
        intercepts.append(intercept)

# Now compute standard deviations
slope_sd = np.std(slopes)
intercept_sd = np.std(intercepts)

print(f"Slope SD: {slope_sd:.4f}")
print(f"Intercept SD: {intercept_sd:.4f}")



# Load example z_df and z_mu_df (e.g., bootstrap 0)
with open(os.path.join('bootstrap_results', 'z_df_0000.pkl'), 'rb') as f:
    z_df = pickle.load(f)
with open(os.path.join('bootstrap_results', 'z_mu_df_0000.pkl'), 'rb') as f:
    z_mu_df = pickle.load(f)

# Sort W and get mappings
with open(os.path.join('bootstrap_results', 'W_0000.pkl'), 'rb') as f:
    W = pickle.load(f)



sorted_eigenvalues, z_col_rank_map, mu_col_rank_map = sort_W(W, z_df, z_mu_df)

plot_all_z_with_bootstrap_streamed(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    bootstrap_dir='bootstrap_results',
    n_bootstraps=250,
    output_pdf='z_bootstrapped_again.pdf'
)


