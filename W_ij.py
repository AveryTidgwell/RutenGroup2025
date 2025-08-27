W_boot_list = []

for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
    sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
    df_boot = pd.concat([df[df['AnimalID'] == aid] for aid in sampled_ids], ignore_index=True)

    data = prepare_model_data(df_boot, kalman=kalman)
    try:
        W, L = linear_regression(data)
        mu = estimate_mu(L, data['x_cov'])

        eigvals, _ = np.linalg.eig(W)
        eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]
        eigen_boot.append(np.real(eigvals_sorted))

        z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)
        z_df_list.append(z_df_j)
        z_mu_df_list.append(z_mu_df_j)

        W_boot_list.append(W)
    except:
        continue





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_W_scatter_with_bootstrap(W0, W_boot_list, n_bins=100, frac=0.2):
    d = W0.shape[0]

    # Get (i, j) index pairs for off-diagonal elements
    ij_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]

    # Compute x = |W_ii - W_jj| and y = W_ij for W0
    x_vals = [np.abs(W0[i, i] - W0[j, j]) for i, j in ij_pairs]
    y_vals = [W0[i, j] for i, j in ij_pairs]

    # Stack bootstrapped y values
    y_boot_matrix = np.array([
        [Wb[i, j] for i, j in ij_pairs]
        for Wb in W_boot_list
    ])  # shape (n_bootstraps, n_pairs)

    # Compute mean and std of y across bootstraps
    y_mean = y_boot_matrix.mean(axis=0)
    y_std = y_boot_matrix.std(axis=0)

    # Bin the x-values and aggregate smoothed mean/std of y
    df = pd.DataFrame({
        'x': x_vals,
        'y': y_vals,
        'y_mean': y_mean,
        'y_std': y_std
    })

    # Sort by x
    df = df.sort_values('x')

    # LOWESS smoothing
    lowess_mean = lowess(df['y_mean'], df['x'], frac=frac, return_sorted=True)
    lowess_std = lowess(df['y_std'], df['x'], frac=frac, return_sorted=True)

    # Plotting
    plt.figure(figsize=(8, 7))
    plt.scatter(df['x'], df['y'], alpha=0.3, color='red', s=10, label='W_ij (off-diagonal)')
    plt.plot(lowess_mean[:, 0], lowess_mean[:, 1], color='blue', lw=2.5, label='Smoothed Mean')
    plt.fill_between(lowess_std[:, 0],
                    lowess_mean[:, 1] - lowess_std[:, 1],
                    lowess_mean[:, 1] + lowess_std[:, 1],
                    color='blue', alpha=0.2, label='±1 SD')
    plt.hlines(y=0, linestyles='--', xmin=0, xmax=43, colors='grey')
    plt.xlabel(r'$|W_{ii} - W_{jj}|$')
    plt.ylabel(r'$W_{ij}$')
    plt.title('W Off-Diagonal vs |W_ii - W_jj| with Smoothed Mean ±1 SD')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()



plot_W_scatter_with_bootstrap(W0, W_boot_list)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_W_by_rank_distance(W0, W_boot_list, frac=0.2):
    d = W0.shape[0]

    # Step 1: Sort indices by closeness of W_ii to zero
    diag_abs = np.abs(np.diag(W0))
    sorted_indices = np.argsort(diag_abs)  # indices of biomarkers sorted from near-0 to large |W_ii|

    # Step 2: Reorder W0 and each W in the bootstrap list
    W0_sorted = W0[np.ix_(sorted_indices, sorted_indices)]
    W_boot_sorted = [W[np.ix_(sorted_indices, sorted_indices)] for W in W_boot_list]

    # Step 3: Build off-diagonal i, j pairs and collect data
    ij_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    x_vals = [np.abs(i - j) for i, j in ij_pairs]
    y_vals = [W0_sorted[i, j] for i, j in ij_pairs]

    # Step 4: Bootstrapped W_ij values
    y_boot_matrix = np.array([
        [Wb[i, j] for i, j in ij_pairs]
        for Wb in W_boot_sorted
    ])  # shape (n_bootstraps, n_pairs)

    y_mean = y_boot_matrix.mean(axis=0)
    y_std = y_boot_matrix.std(axis=0)

    df = pd.DataFrame({
        'rank_distance': x_vals,
        'W_ij': y_vals,
        'mean': y_mean,
        'std': y_std
    }).sort_values('rank_distance')

    # Step 5: LOWESS smoothing
    lowess_mean = lowess(df['mean'], df['rank_distance'], frac=frac, return_sorted=True)
    lowess_std = lowess(df['std'], df['rank_distance'], frac=frac, return_sorted=True)

    # Step 6: Plot
    plt.figure(figsize=(8, 7))
    plt.scatter(df['rank_distance'], df['W_ij'], alpha=0.3, color='red', s=10, label='W_ij')
    plt.plot(lowess_mean[:, 0], lowess_mean[:, 1], color='blue', lw=2.5, label='Smoothed Mean')
    plt.fill_between(lowess_std[:, 0],
                     lowess_mean[:, 1] - lowess_std[:, 1],
                     lowess_mean[:, 1] + lowess_std[:, 1],
                     color='blue', alpha=0.2, label='±1 SD')
    plt.hlines(y=0, linestyles='--', xmin=0, xmax=43, colors='grey')
    plt.xlabel(r'|$i - j$|')
    plt.ylabel(r'$W_{ij}$ (off-diagonal)')
    plt.title(r'$W_{ij}$ vs $|i-j|$')
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_W_by_rank_distance(W0, W_boot_list)






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_W_scatter_with_bootstrap(W0, W_boot_list, n_bins=100, frac=0.2):
    d = W0.shape[0]

    # Get (i, j) index pairs for off-diagonal elements
    ij_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]

    # Compute x = |W_ii - W_jj| and y = W_ij for W0
    x_vals = [np.abs(W0[i, i] - W0[j, j]) for i, j in ij_pairs]
    y_vals = [W0[i, j] for i, j in ij_pairs]

    # Stack bootstrapped y values
    y_boot_matrix = np.array([
        [Wb[i, j] for i, j in ij_pairs]
        for Wb in W_boot_list
    ])  # shape (n_bootstraps, n_pairs)

    # Compute mean and std of y across bootstraps
    y_mean = y_boot_matrix.mean(axis=0)
    y_std = y_boot_matrix.std(axis=0)

    # Bin the x-values and aggregate smoothed mean/std of y
    df = pd.DataFrame({
        'x': x_vals,
        'y': y_vals,
        'y_mean': y_mean,
        'y_std': y_std
    })

    # Sort by x
    df = df.sort_values('x')

    # LOWESS smoothing
    lowess_mean = lowess(df['y_mean'], df['x'], frac=frac, return_sorted=True)
    lowess_std = lowess(df['y_std'], df['x'], frac=frac, return_sorted=True)

    # Global statistics for all W_ij (off-diagonal)
    y_arr = np.array(y_vals)
    mean_global = np.mean(y_arr)
    std_global = np.std(y_arr)
    sem_global = std_global / np.sqrt(len(y_arr))  # Standard Error of the Mean

    # Plotting
    plt.figure(figsize=(9, 7))
    plt.scatter(df['x'], df['y'], alpha=0.3, color='grey', s=10, label='W_ij (off-diagonal)')
    plt.plot(lowess_mean[:, 0], lowess_mean[:, 1], color='red', lw=2.5, label='Smoothed Mean')
    plt.fill_between(lowess_std[:, 0],
                     lowess_mean[:, 1] - lowess_std[:, 1],
                     lowess_mean[:, 1] + lowess_std[:, 1],
                     color='red', alpha=0.2, label='±1 SD')

    # Global mean and SEM band
    plt.hlines(mean_global, xmin=0, xmax=1.2, color='blue', linestyle='-', lw=1, label='Mean of W_ij')
    plt.fill_between([0, max(df['x'])],
                     [mean_global - sem_global] * 2,
                     [mean_global + sem_global] * 2,
                     color='blue', alpha=0.5, label='±1 SEM')

    # Global ±1 SD lines
    plt.hlines(mean_global + std_global, xmin=0, xmax=1.2, color='black', linestyle='--', lw=1, label='±1 SD')
    plt.hlines(mean_global - std_global, xmin=0, xmax=1.2, color='black', linestyle='--', lw=1)

    # Other plot settings
    plt.axhline(y=0, linestyle='--', color='grey', lw=1)
    plt.xlabel(r'$|W_{ii} - W_{jj}|$')
    plt.ylabel(r'$W_{ij}$')
    plt.title('W Off-Diagonal vs |W_ii - W_jj|')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    print(f'Mean: {mean_global}')

plot_W_scatter_with_bootstrap(W0, W_boot_list)






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_W_rank_distance_with_bootstrap(W0, W_boot_list, frac=0.2):
    d = W0.shape[0]

    # Step 1: Sort by |W_ii| to define rank order
    sorted_indices = np.argsort(np.abs(np.diag(W0)))
    W0_sorted = W0[np.ix_(sorted_indices, sorted_indices)]
    W_boot_sorted = [W[np.ix_(sorted_indices, sorted_indices)] for W in W_boot_list]

    # Step 2: Extract off-diagonal (i, j) pairs and compute rank distance and W_ij
    ij_pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    x_vals = [abs(i - j) for i, j in ij_pairs]
    y_vals = [W0_sorted[i, j] for i, j in ij_pairs]

    # Step 3: Bootstrapped W_ij values
    y_boot_matrix = np.array([
        [W[i, j] for i, j in ij_pairs]
        for W in W_boot_sorted
    ])  # shape: (n_bootstraps, n_pairs)

    y_mean = y_boot_matrix.mean(axis=0)
    y_std = y_boot_matrix.std(axis=0)

    # Step 4: Prepare dataframe
    df = pd.DataFrame({
        'x': x_vals,
        'y': y_vals,
        'y_mean': y_mean,
        'y_std': y_std
    }).sort_values('x')

    # Step 5: LOWESS smoothing
    lowess_mean = lowess(df['y_mean'], df['x'], frac=frac, return_sorted=True)
    lowess_std = lowess(df['y_std'], df['x'], frac=frac, return_sorted=True)

    # Step 6: Global statistics
    y_arr = np.array(y_vals)
    mean_global = y_arr.mean()
    std_global = y_arr.std()
    sem_global = std_global / np.sqrt(len(y_arr))

    # Step 7: Plot
    plt.figure(figsize=(8, 7))
    plt.scatter(df['x'], df['y'], alpha=0.3, color='grey', s=10, label='W_ij (off-diagonal)')
    plt.plot(lowess_mean[:, 0], lowess_mean[:, 1], color='red', lw=2.5, label='Smoothed Mean')
    plt.fill_between(lowess_std[:, 0],
                     lowess_mean[:, 1] - lowess_std[:, 1],
                     lowess_mean[:, 1] + lowess_std[:, 1],
                     color='red', alpha=0.2, label='±1 SD (boot)')

    # Global horizontal lines
    xmax = max(df['x'])
    plt.hlines(mean_global, xmin=0, xmax=xmax, color='blue', linestyle='-', lw=1, label='Mean of W_ij')
    plt.fill_between([0, xmax],
                     [mean_global - sem_global] * 2,
                     [mean_global + sem_global] * 2,
                     color='blue', alpha=0.5, label='±1 SEM')
    plt.hlines([mean_global + std_global, mean_global - std_global],
               xmin=0, xmax=xmax, color='black', linestyle='--', lw=1, label='±1 SD (global)')

    # Axis labels and formatting
    plt.axhline(0, color='grey', linestyle='--', lw=1)
    plt.xlabel(r'$|i - j|$')
    plt.ylabel(r'$W_{ij}$')
    plt.title('W Off-Diagonal vs Rank Distance')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    print(f'Mean: {mean_global:.4f} | SD: {std_global:.4f} | SEM: {sem_global:.4f}')


plot_W_rank_distance_with_bootstrap(W0, W_boot_list)
