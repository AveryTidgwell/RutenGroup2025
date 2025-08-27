from statsmodels.nonparametric.smoothers_lowess import lowess
def bootstrap_regression(x, y, n_boot=1000, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    boot_preds = []

    x_pred = np.linspace(x.min(), x.max(), 100)

    for _ in range(n_boot):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_sample, y_sample = x[indices], y[indices]
        if len(np.unique(x_sample)) < 2:
            continue  # skip degenerate resample
        slope, intercept = np.polyfit(x_sample, y_sample, 1)
        y_pred = slope * x_pred + intercept
        boot_preds.append(y_pred)

    boot_preds = np.array(boot_preds)
    lower = np.percentile(boot_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    mean = np.mean(boot_preds, axis=0)

    return x_pred, mean, lower, upper



def bootstrap_lowess_ci(x, y, frac=0.3, n_boot=300, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    x_eval = np.linspace(x.min(), x.max(), 100)
    all_smooths = []

    for _ in range(n_boot):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]

        if len(np.unique(x_sample)) < 2:
            continue

        try:
            smooth = lowess(y_sample, x_sample, frac=frac, xvals=x_eval)
            if np.any(np.isnan(smooth)):
                continue
            all_smooths.append(smooth)
        except Exception as e:
            print(f"LOWESS failed on bootstrap: {e}")
            continue


    all_smooths = np.array(all_smooths)
    if len(all_smooths) < 10:
        raise ValueError("LOWESS bootstrap failed too often.")

    lower = np.percentile(all_smooths, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_smooths, 100 * (1 - alpha / 2), axis=0)
    mean = np.median(all_smooths, axis=0)

    return x_eval, mean, lower, upper




from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.cm as cm
import matplotlib.colors as mcolors



def plot_all_z(z_df, z_mu_df, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map,
               output_pdf='z_ranked_properly.pdf', highlight_n_dolphins=5):
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'Smoother_z')
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, output_pdf)

    # Filter dolphins with more than 5 data points
    dolphin_counts = z_df['AnimalID'].value_counts()
    eligible_ids = dolphin_counts[dolphin_counts > 5].index.tolist()

    # Check if we have enough eligible dolphins
    if len(eligible_ids) < highlight_n_dolphins:
        raise ValueError(f"Not enough dolphins with >5 data points to highlight. Found only {len(eligible_ids)}.")

    # Randomly select AnimalIDs to highlight
    highlighted_ids = np.random.choice(eligible_ids, size=highlight_n_dolphins, replace=False)

    # Assign unique colors
    cmap = cm.get_cmap('tab10' if highlight_n_dolphins <= 10 else 'tab20')
    colors = {aid: cmap(i % cmap.N) for i, aid in enumerate(highlighted_ids)}

    with PdfPages(pdf_path) as pdf:
        for i, (z_col, mu_col) in enumerate(zip(z_col_rank_map, mu_col_rank_map)):
            plt.figure(figsize=(9, 8))

            t = z_df['Age']
            z_vals = z_df[z_col]
            mu_vals = z_df[mu_col]

            # Faint scatter for all data
            plt.scatter(t, z_vals, c='blue', alpha=0.1, label='All Data')

            # Highlight selected dolphins
            for aid in highlighted_ids:
                df_dolphin = z_df[z_df['AnimalID'] == aid].sort_values('Age')
                plt.plot(df_dolphin['Age'], df_dolphin[z_col],
                         marker='o', linewidth=2.5, markersize=6, alpha=0.8,
                         color=colors[aid], label=f'Dolphin {aid}')

            # Regression on mu
            x_pred, mean_pred, lower_ci, upper_ci = bootstrap_regression(t, mu_vals)
            plt.plot(x_pred, mean_pred, linestyle='--', linewidth=2, c='red', label='Linear Fit (mu)')
            plt.fill_between(x_pred, lower_ci, upper_ci, color='red', alpha=0.3, label='95% CI (Linear)')

            # LOWESS fit on raw data
            x_vals = np.array(t)
            y_vals = np.array(z_vals)
            x_eval, mean_lowess, lower_lowess_ci, upper_lowess_ci = bootstrap_lowess_ci(x_vals, y_vals)

            plt.plot(x_eval, mean_lowess, color='magenta', linestyle='-', linewidth=2.5, label='Smooth Fit Line')
            plt.fill_between(x_eval, lower_lowess_ci, upper_lowess_ci, color='magenta', alpha=0.2, label='95% CI (LOWESS)')

            # Autocorrelation markers
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
            plt.show()
            plt.close()

    print(f"All plots saved to {pdf_path}")



plot_all_z(
    z_df, z_mu_df, sorted_eigenvalues,
    z_col_rank_map, mu_col_rank_map,
    output_pdf='z_with_38_biomarkers.pdf',
    highlight_n_dolphins=3
)
