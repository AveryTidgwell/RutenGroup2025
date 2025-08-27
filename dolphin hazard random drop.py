def run_c_index_reduction(input_df, n_bootstrap=750, test_frac=0.2, random_seed=42):
    final_df = preprocess_final_samples(input_df)  # Your existing preprocessing function
    final_df = prepare_cox_covariates(final_df)
    np.random.seed(random_seed)
    
    excluded_cols = ['Age', 'event']
    all_covariates = [c for c in final_df.columns if c not in excluded_cols]
    results = []

    # Number of variables to drop in this "layer" will increase each iteration
    num_drops = 0  

    while len(all_covariates) > num_drops:
        print(f"\nRunning with {len(all_covariates) - num_drops} covariates per bootstrap...")
        try:
            bootstrap_c_indices = []

            for _ in range(n_bootstrap):
                # Randomly drop `num_drops` covariates for this bootstrap run
                dropped_vars = np.random.choice(all_covariates, size=num_drops, replace=False)
                current_covariates = [c for c in all_covariates if c not in dropped_vars]

                # Shuffle and split
                df_shuffled = final_df.sample(frac=1, random_state=np.random.randint(0, 1e6)).reset_index(drop=True)
                train_size = int(len(df_shuffled) * (1 - test_frac))
                train_df = df_shuffled.iloc[:train_size]
                test_df = df_shuffled.iloc[train_size:]

                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(train_df[['Age', 'event'] + current_covariates],
                        duration_col='Age', event_col='event')

                # Risk prediction
                test_df = test_df[['Age', 'event'] + current_covariates].copy()
                test_df['risk_score'] = cph.predict_partial_hazard(test_df)

                # Calculate bootstrap C-index manually
                concordant_list = []
                for _ in range(150):
                    pair = test_df.sample(n=2, replace=False)
                    times = pair['Age'].values
                    risks = pair['risk_score'].values
                    if times[0] == times[1]:
                        continue
                    if (risks[0] > risks[1] and times[0] < times[1]) or \
                       (risks[1] > risks[0] and times[1] < times[0]):
                        concordant_list.append(1)
                    else:
                        concordant_list.append(0)

                if concordant_list:
                    bootstrap_c_indices.append(np.mean(concordant_list))

            # Summary for this number of dropped variables
            mean_c = np.mean(bootstrap_c_indices)
            sd_c = np.std(bootstrap_c_indices)
            print(f"Dropped {num_drops} vars: C-index = {mean_c:.4f} ± {sd_c:.4f}")
            results.append((len(all_covariates) - num_drops, mean_c, sd_c))

        except Exception as e:
            print("Error:", e)
            break

        # Increase number of drops for next layer
        num_drops += 1

    return np.array(results)




def c_index_boot_plot_compare(df, z_df, n_bootstrap=750, test_frac=0.2, random_seed=42):
    results_df = run_c_index_reduction(df, n_bootstrap, test_frac, random_seed)
    results_zdf = run_c_index_reduction(z_df, n_bootstrap, test_frac, random_seed)

    plt.figure(figsize=(9, 8))

    plt.errorbar(results_df[:, 0], results_df[:, 1], yerr=results_df[:, 2],
                 fmt='o', capsize=5, color='orange', label='Original Biomarkers')
    plt.errorbar(results_zdf[:, 0], results_zdf[:, 1], yerr=results_zdf[:, 2],
                 fmt='s', capsize=5, color='blue', label='Z-transformed Biomarkers')

    plt.hlines(y=0.5, xmin=1, xmax=max(results_df[:, 0].max(), results_zdf[:, 0].max()),
               linestyles='dashed', colors='grey', linewidths=2)

    # X ticks at every integer, but labels every 2nd one
    max_cov = max(results_df[:, 0].max(), results_zdf[:, 0].max())
    xticks = np.arange(1, max_cov + 1, 1)
    xtick_labels = [f"{int(x)}" if x % 2 == 0 else '' for x in xticks]  # labels every 2 ticks as ints
    plt.xticks(xticks, xtick_labels)
    

    plt.xlabel("Number of covariates")
    plt.ylabel("Bootstrap C-index")
    plt.title("C-index vs Number of Covariates: Raw vs Z-transformed")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.show()

    return results_df, results_zdf

c_index_boot_plot_compare(df, z_df, n_bootstrap=750, test_frac=0.2, random_seed=42)
