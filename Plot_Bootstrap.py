def plot_all_z_with_bootstrap_streamed(
    z_df,
    z_mu_df,
    sorted_eigenvalues,
    z_col_rank_map,
    mu_col_rank_map,
    bootstrap_dir,
    n_bootstraps,
    output_pdf='z_ranked_properly.pdf'
):
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import pickle
    import pandas as pd

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'Smoother_z')
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, output_pdf)

    # Filter dolphins with more than 5 data points
    dolphin_counts = z_df['AnimalID'].value_counts()
    eligible_ids = dolphin_counts[dolphin_counts > 5].index.tolist()

    if len(eligible_ids) < highlight_n_dolphins:
        raise ValueError(f"Not enough dolphins with >5 data points to highlight. Found only {len(eligible_ids)}.")

    highlighted_ids = np.random.choice(eligible_ids, size=highlight_n_dolphins, replace=False)

    # Assign colors
    cmap = cm.get_cmap('tab10' if highlight_n_dolphins <= 10 else 'tab20')
    colors = {aid: cmap(i % cmap.N) for i, aid in enumerate(highlighted_ids)}

    with PdfPages(pdf_path) as pdf:
        for i, (z_col, mu_col) in enumerate(zip(z_col_rank_map, mu_col_rank_map)):
            plt.figure(figsize=(9, 8))

            t = z_df['Age']
            z_vals = z_df[z_col]
            mu_vals = z_mu_df[mu_col]

            # === Scatter all data ===
            plt.scatter(t, z_vals, c='blue', alpha=0.1, label='All Data')

            # === Highlight selected dolphins ===
            for aid in highlighted_ids:
                df_dolphin = z_df[z_df['AnimalID'] == aid].sort_values('Age')
                plt.plot(df_dolphin['Age'], df_dolphin[z_col],
                         marker='o', linewidth=2.5, markersize=6, alpha=0.8,
                         color=colors[aid], label=f'Dolphin {aid}')

            # === Linear mu line (interpolated) ===
            x_pred = np.linspace(t.min(), t.max(), 200)

            df_mu = pd.DataFrame({'Age': z_df['Age'].values, 'mu': mu_vals.values}).dropna().sort_values('Age')
            y_base = np.interp(x_pred, df_mu['Age'].values, df_mu['mu'].values)

            # Accumulate bootstrap interpolations
            mu_bootstrap_stack = []
            for j in range(n_bootstraps):
                try:
                    with open(os.path.join(bootstrap_dir, f'z_mu_df_{j:04d}.pkl'), 'rb') as f:
                        z_mu_df_j = pickle.load(f)

                    df_boot = pd.DataFrame({
                        'Age': z_mu_df_j['Age'].values,
                        'mu': z_mu_df_j[mu_col].values
                    }).dropna().sort_values('Age')

                    if len(df_boot) >= 2:
                        yj = np.interp(x_pred, df_boot['Age'].values, df_boot['mu'].values)
                        mu_bootstrap_stack.append(yj)

                except Exception as e:
                    print(f"Skipping z_mu_df_{j:04d}: {e}")
                    continue

            if len(mu_bootstrap_stack) > 0:
                mu_bootstrap_stack = np.vstack(mu_bootstrap_stack)
                y_mean = mu_bootstrap_stack.mean(axis=0)
                y_std = mu_bootstrap_stack.std(axis=0)

                plt.plot(x_pred, y_mean, linestyle='--', linewidth=2, color='red', label='Linear Fit (mu)')
                plt.fill_between(x_pred, y_mean - y_std, y_mean + y_std, color='red', alpha=0.3, label='±1 SD (Linear)')

            # === LOWESS ===
            lowess_y = lowess(z_vals, t, frac=0.3, return_sorted=True)
            x_lowess = lowess_y[:, 0]
            y_lowess = lowess_y[:, 1]

            lowess_bootstrap_stack = []
            for j in range(n_bootstraps):
                try:
                    with open(os.path.join(bootstrap_dir, f'z_df_{j:04d}.pkl'), 'rb') as f:
                        z_df_j = pickle.load(f)

                    if z_col in z_df_j.columns and 'Age' in z_df_j.columns:
                        df_boot_lowess = z_df_j[['Age', z_col]].dropna().sort_values('Age')

                        if len(df_boot_lowess) >= 2:
                            lowess_j = lowess(df_boot_lowess[z_col], df_boot_lowess['Age'], frac=0.3, return_sorted=True)
                            yj_interp = np.interp(x_lowess, lowess_j[:, 0], lowess_j[:, 1])
                            lowess_bootstrap_stack.append(yj_interp)

                except Exception as e:
                    print(f"Skipping z_df_{j:04d}: {e}")
                    continue

            if len(lowess_bootstrap_stack) > 0:
                lowess_bootstrap_stack = np.vstack(lowess_bootstrap_stack)
                y_lowess_mean = lowess_bootstrap_stack.mean(axis=0)
                y_lowess_std = lowess_bootstrap_stack.std(axis=0)

                plt.plot(x_lowess, y_lowess_mean, color='magenta', linestyle='-', linewidth=2.5, label='Smooth Fit (LOWESS)')
                plt.fill_between(x_lowess, y_lowess_mean - y_lowess_std, y_lowess_mean + y_lowess_std,
                                 color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

            # Optional: autocorrelation marker
            eigenvalue = np.real(sorted_eigenvalues[i])
            if eigenvalue != 0:
                recovery_time = -1 / eigenvalue
                marker_positions = np.arange(t.min(), t.max(), recovery_time)
                plt.plot(marker_positions, [-4.5]*len(marker_positions), linestyle='None', marker='x',
                         color='black', label=f'Auto-Corr Time: {recovery_time:.1f} yrs')

            plt.ylim(-5, 5)
            plt.grid()
            plt.xlabel('Age (yrs)')
            plt.ylabel(f'Natural Variable: z_{i+1}')
            plt.title(f'z_{i+1}')
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()

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
    z_df=z_df,
    z_mu_df=z_mu_df,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    bootstrap_dir='bootstrap_results',
    n_bootstraps=250,
    output_pdf='z_bootstrapped_ci_streamed.pdf'
)

