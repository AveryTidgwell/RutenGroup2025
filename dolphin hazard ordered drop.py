cph = CoxPHFitter()
final_df = preprocess_final_samples(df)
final_df = prepare_cox_covariates(final_df)
cph.fit(final_df, duration_col='Age', event_col='event')
cph.print_summary()
cph_summary = cph.summary
cph_summary = cph_summary.sort_values(by='coef', key=lambda col: abs(col))
removal_order = cph_summary.index

z_cph = CoxPHFitter()
z_final_df = preprocess_final_samples(z_df)
z_final_df = prepare_cox_covariates(z_final_df)
z_cph.fit(z_final_df, duration_col='Age', event_col='event')
z_cph.print_summary()
z_cph_summary = z_cph.summary
z_cph_summary = z_cph_summary.sort_values(by='coef', key=lambda col: abs(col))
z_removal_order = z_cph_summary.index






def run_c_index_reduction(input_df, removal_order, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    final_df = preprocess_final_samples(input_df)  # Your existing preprocessing function
    final_df = prepare_cox_covariates(final_df)
    np.random.seed(random_seed)
    
    excluded_cols = ['Age', 'event']
    all_covariates = list(removal_order)
    results = []

    while len(all_covariates) > 0:
        print(f"\nRunning with {len(all_covariates)} covariates per bootstrap...")
        try:
            bootstrap_c_indices = []

            for _ in range(n_bootstrap):
                current_covariates = all_covariates.copy()

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
                for _ in range(200):
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
            print(f"{len(all_covariates)} vars: C-index = {mean_c:.4f} ± {sd_c:.4f}")
            results.append((len(all_covariates), mean_c, sd_c))

        except Exception as e:
            print("Error:", e)
            break

        # Increase number of drops for next layer
        all_covariates.pop(0)

    return np.array(results)




def c_index_boot_plot_compare(df, z_df, removal_order, z_removal_order, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    results_df = run_c_index_reduction(df, removal_order, n_bootstrap, test_frac, random_seed)
    results_zdf = run_c_index_reduction(z_df, z_removal_order, n_bootstrap, test_frac, random_seed)

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
    #plt.gca().tick_params(axis='x', which='both', top=True, bottom=True)

    plt.xlabel("Number of covariates")
    plt.ylabel("Bootstrap C-index")
    plt.title("C-index vs Number of Covariates: Raw vs Z-transformed")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.show()

    return results_df, results_zdf

c_index_boot_plot_compare(df, z_df, removal_order, z_removal_order, n_bootstrap=1000, test_frac=0.2, random_seed=42)



###========= INCLUDE ALL SAMPLES ========###

def preprocess_final_samples(df):
    # Copy and ensure sorted
    df = df[df['AnimalID'].map(df['AnimalID'].value_counts()) > 1]
    df = df.sort_values(by=["AnimalID", "Age"]).copy()

    # Drop rows missing Age
    df = df.dropna(subset=["Age"])

    # Mark event = 1 for last sample per AnimalID, else 0
    df['event'] = 0
    df.loc[df.groupby("AnimalID")['Age'].idxmax(), 'event'] = 1

    return df.reset_index(drop=True)






z_cph = CoxPHFitter()
z_final_df = preprocess_final_samples(z_df)
z_final_df = prepare_cox_covariates(z_final_df)
z_cph.fit(z_final_df, duration_col='Age', event_col='event')
z_cph.print_summary()
z_cph_summary = z_cph.summary


from scipy.stats import linregress
def plot_beta_vs_eigvals():
    plt.figure(figsize=(9,6))
    eigval_df = pd.DataFrame(
        1 / abs(eigvals.real), 
        index=[f"z_{i+1}" for i in range(len(z_cols))], 
        columns=["inv_abs_eigval"]
    )

    # Select only the z’s that appear in the cox summary
    used_cols = z_cph_summary.index
    x = eigval_df.loc[used_cols, "inv_abs_eigval"]
    y = abs(z_cph_summary.loc[used_cols, "coef"])
    
    for label, xi, yi in zip(used_cols, x, y):
        label = label.replace('z_', '')
        plt.text(xi, yi, label, fontsize=9, ha="right", va="bottom")
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color="grey", linestyle="--")

    
    plt.scatter(x, y, color="blue", s=10)
    plt.xlabel("Auto-correlation Time, 1 / |λ| (yrs)")
    plt.ylabel("Proportional Hazard Coefficient, |β|")
    plt.title("Hazard Coefficients vs. Auto-correlation Times \n of Natural Variables")
    #plt.grid()
    plt.show()
    plt.close()

plot_beta_vs_eigvals()


from scipy.stats import linregress

def plot_hazard_vs_drift(z_df, z_mu_df, z_cph_summary):
    plt.figure(figsize=(9,6))
    
    # Compute steady-state drift (slope of each z_mu over Age)
    drift_slopes = {}
    for col in z_mu_df.columns:
        slope, _ = np.polyfit(z_df['Age'], z_mu_df[col], 1)
        # Ensure the key in the dict is "z_i" to match Cox summary
        drift_slopes[f"z_{col.split('_')[-1]}"] = slope

    drift_df = pd.DataFrame.from_dict(drift_slopes, orient='index', columns=["drift"])
    
    # Make sure both indices are strings and stripped
    drift_index = drift_df.index.astype(str).str.strip()
    cph_index = z_cph_summary.index.astype(str).str.strip()
    
    # Only keep z's that exist in both
    used_cols = drift_index.intersection(cph_index)
    
    if used_cols.empty:
        print("Warning: no matching z's found between drift and Cox summary!")
        print("Drift index:", drift_index.tolist())
        print("Cox summary index:", cph_index.tolist())
        return

    x = drift_df.loc[used_cols, "drift"]
    y = z_cph_summary.loc[used_cols, "coef"]
    
    # Annotate each point
    for label, xi, yi in zip(used_cols, x, y):
        label_short = label.replace('z_', '')
        plt.text(xi, yi, label_short, fontsize=9, ha="right", va="bottom")
    
    # Linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color="grey", linestyle="--")
    
    # Scatter points
    plt.scatter(x, y, color="blue", s=10)
    plt.xlabel("Steady-state Drift (slope of μ over Age)")
    plt.ylabel("Proportional Hazard Coefficient, β")
    plt.title("Hazard Coefficients vs. Steady-state Drift of Natural Variables")
    plt.show()
    plt.close()


# Call the function
plot_hazard_vs_drift(z_df, z_mu_df, z_cph_summary)



