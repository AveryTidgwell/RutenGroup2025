import xlsxwriter
# ========== REPEATED HALF-DUPLICATE SAMPLING ==========
n_repeats = 25
eigvals_half_dup_list = []

for seed in range(n_repeats):
    np.random.seed(seed)
    df_half = df.sample(frac=0.5, random_state=seed)
    df_half_dup = pd.concat([df_half, df_half.copy()], ignore_index=True)

    try:
        data_half_dup = prepare_model_data(df_half_dup, kalman=kalman)
        W_half_dup, L_half_dup = linear_regression(data_half_dup)
        eigvals_half = np.linalg.eigvals(W_half_dup)
        eigvals_half_sorted = np.real(eigvals_half[np.argsort(np.abs(eigvals_half))])
        eigvals_half_dup_list.append(eigvals_half_sorted)
    except Exception as e:
        print(f"Repeat {seed} failed: {e}")
        continue

# Convert to array
eigvals_half_dup_array = np.vstack(eigvals_half_dup_list)
mean_eigs_half_dup = np.mean(eigvals_half_dup_array, axis=0)
std_eigs_half_dup = np.std(eigvals_half_dup_array, axis=0)



# ========== 10 RANDOM GROUPS ==========
n_groups = 10
group_size = len(df) // n_groups
shuffled_df = df.sample(frac=1, random_state=999).reset_index(drop=True)

eigvals_grouped = []

for i in range(n_groups):
    start_idx = i * group_size
    end_idx = (i + 1) * group_size if i < n_groups - 1 else len(shuffled_df)
    df_group = shuffled_df.iloc[start_idx:end_idx]

    # Process group
    try:
        data_group = prepare_model_data(df_group, kalman=kalman)
        W_group, L_group = linear_regression(data_group)
        eigvals_group = np.linalg.eigvals(W_group)
        eigvals_sorted = np.real(eigvals_group[np.argsort(np.abs(eigvals_group))])
        eigvals_grouped.append(eigvals_sorted)
    except Exception as e:
        print(f"Group {i} failed: {e}")
        continue

# Stack and compute mean/std
eigvals_grouped = np.vstack(eigvals_grouped)
mean_eigs_groups = np.mean(eigvals_grouped, axis=0)
std_eigs_groups = np.std(eigvals_grouped, axis=0)


n_repeats = 25
sample_frac = 0.1
eigvals_subsample_list = []

for seed in range(n_repeats):
    np.random.seed(seed)
    df_sample = df.sample(frac=sample_frac, random_state=seed)

    try:
        data_sample = prepare_model_data(df_sample, kalman=kalman)
        W_sample, _ = linear_regression(data_sample)
        eigvals_sample = np.linalg.eigvals(W_sample)
        eigvals_sorted = np.real(eigvals_sample[np.argsort(np.abs(eigvals_sample))])
        eigvals_subsample_list.append(eigvals_sorted)
    except Exception as e:
        print(f"Subsample {seed} failed: {e}")
        continue
eigvals_subsample_array = np.vstack(eigvals_subsample_list)
mean_eigs_subsample = np.mean(eigvals_subsample_array, axis=0)
std_eigs_subsample = np.std(eigvals_subsample_array, axis=0)



def plot_comparison_eigenvalues(W0, mean_eigs_half_dup, std_eigs_half_dup, mean_eigs_groups, std_eigs_groups, bootstrap_dir, n_bootstraps=250):
    eigen_boot = []
    for j in range(n_bootstraps):
        try:
            with open(os.path.join(bootstrap_dir, f'W_{j:04d}.pkl'), 'rb') as f:
                W_j = pickle.load(f)
            eigvals = np.linalg.eigvals(W_j)
            eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]  # sort by abs value
            eigen_boot.append(np.real(eigvals_sorted))
        except Exception:
            continue

    if not eigen_boot:
        print("No valid bootstrap eigenvalues found.")
        return

    eigen_boot = np.vstack(eigen_boot)
    mean_eigs_boot = np.mean(eigen_boot, axis=0)
    std_eigs_boot = np.std(eigen_boot, axis=0)

    eigvals0 = np.linalg.eigvals(W0)
    eigvals0_sorted = np.real(eigvals0[np.argsort(np.abs(eigvals0))])
    x = np.arange(1, len(eigvals0_sorted) + 1)

    plt.figure(figsize=(8, 6))

    # Baseline
    plt.errorbar(x, eigvals0_sorted, yerr=std_eigs_boot, fmt='o', capsize=3, color='black', label='Baseline Population Values')

    # Bootstrap
    plt.errorbar(x, mean_eigs_boot, yerr=std_eigs_boot, fmt='o', capsize=3, color='red', label='Bootstrap Mean')

    # Half-Dropped + Duplicated
    plt.errorbar(x, mean_eigs_half_dup, yerr=std_eigs_half_dup, fmt='o', capsize=3, color='blue', label='Half-Dropped + Duplicated (Mean ± SD)')

    # 10 Group Average
    plt.errorbar(x, mean_eigs_groups, yerr=std_eigs_groups, fmt='o', capsize=3, color='purple', label='10 Random Groups Avg')

    # In the plotting section
    plt.errorbar(x, mean_eigs_subsample, yerr=std_eigs_subsample, fmt='o', capsize=3, color='green', label='10% Subsample')

    plt.ylim(-2, 0)
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue')
    plt.title('Comparison of Eigenvalues of $W$')
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()

    
    
    
    

    # ========= EIGENVALUE RATIO ANALYSIS =========
    # Get baseline eigenvalues
    eigvals0 = np.linalg.eigvals(W0)
    eigvals0_sorted = np.real(eigvals0[np.argsort(np.abs(eigvals0))])
    n_eigs = len(eigvals0_sorted)

    # Truncate all to same length for safety
    mean_eigs_half_dup = mean_eigs_half_dup[:n_eigs]
    std_eigs_half_dup = std_eigs_half_dup[:n_eigs]

    mean_eigs_groups = mean_eigs_groups[:n_eigs]
    mean_eigs_boot = mean_eigs_boot[:n_eigs]

    # Compute ratios
    ratio_bootstrap = eigvals0_sorted / mean_eigs_boot
    ratio_half_dup = eigvals0_sorted / eigvals_half_dup_sorted
    ratio_group_avg = eigvals0_sorted / mean_eigs_groups

    # Combine into DataFrame
    comparison_df = pd.DataFrame({
        f'EV_{i+1}': [ratio_bootstrap[i], ratio_half_dup[i], ratio_group_avg[i]]
        for i in range(n_eigs)
    }, index=['Bootstrap Mean', 'Half-Duplicated', '10-Group Avg'])

    # Add mean column
    comparison_df['Mean Ratio'] = comparison_df.mean(axis=1)

    # Show result
    print(comparison_df.round(3))
    # --------- Propagate SDs for ratios ---------
    eps = 1e-12
    eigvals0_sorted = eigvals0_sorted[:n_eigs] + eps
    mean_eigs_half_dup = mean_eigs_half_dup[:n_eigs] + eps
    std_eigs_half_dup = std_eigs_half_dup[:n_eigs]
    mean_eigs_boot = mean_eigs_boot[:n_eigs] + eps
    mean_eigs_groups = mean_eigs_groups[:n_eigs] + eps
    std_eigs_boot = std_eigs_boot[:n_eigs]
    std_eigs_groups = std_eigs_groups[:n_eigs]

    # Compute ratio SDs
    std_ratio_bootstrap = np.abs(ratio_bootstrap) * (std_eigs_boot / mean_eigs_boot)
    std_ratio_groups = np.abs(ratio_group_avg) * (std_eigs_groups / mean_eigs_groups)
    std_ratio_half_dup = np.full_like(std_ratio_bootstrap, np.nan)  # No SD for duplicated group
    ratio_half_dup = eigvals0_sorted / mean_eigs_half_dup
    std_ratio_half_dup = np.abs(ratio_half_dup) * (std_eigs_half_dup / mean_eigs_half_dup)
    ratio_subsample = eigvals0_sorted / (mean_eigs_subsample + eps)
    std_ratio_subsample = np.abs(ratio_subsample) * (std_eigs_subsample / (mean_eigs_subsample + eps))

    std_df = pd.DataFrame({
        f'EV_{i+1}': [
            std_ratio_bootstrap[i],
            std_ratio_half_dup[i],
            std_ratio_groups[i],
            std_ratio_subsample[i]
        ]
        for i in range(n_eigs)
    }, index=['Bootstrap Mean', 'Half-Duplicated', '10-Group Avg', '10% Subsample'])

    # --------- Combined DataFrame: Mean ± SD ---------
    combined_data = {}

    for i in range(n_eigs):
        ev_col = f'EV_{i+1}'
        sd_col = f'{ev_col}_SD'

        combined_data[ev_col] = [
            ratio_bootstrap[i],
            ratio_half_dup[i],
            ratio_group_avg[i],
            ratio_subsample[i]
        ]

        combined_data[sd_col] = [
            std_ratio_bootstrap[i],
            std_ratio_half_dup[i],
            std_ratio_groups[i],
            std_ratio_subsample[i]
        ]



    # Add mean columns
    combined_data['Mean Ratio'] = comparison_df.mean(axis=1)
    combined_data['Mean SD'] = std_df.mean(axis=1, skipna=True)

    # Create final DataFrame
    combined_df = pd.DataFrame(combined_data, index=['Bootstrap Mean', 'Half-Duplicated', '10-Group Avg', '10% Subsample'])

    # --------- Export to Excel (one sheet) ---------
    output_path = 'more_eigenvalues.xlsx'
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        combined_df.to_excel(writer, sheet_name='Ratios ± SD')

    print(f"Exported combined ratio summary to: {output_path}")

    return comparison_df


comparison_df = plot_comparison_eigenvalues(
    W0,
    mean_eigs_half_dup,
    std_eigs_half_dup,
    mean_eigs_groups,
    std_eigs_groups,
    bootstrap_dir=save_dir,
    n_bootstraps=n_bootstraps
)

