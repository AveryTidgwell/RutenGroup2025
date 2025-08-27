def bootstrap_beta_drift():
    n_bootstraps = 200
    kalman = False
    data_path = '/Users/summer/Downloads/dolphins-master/data/dolphin_data.csv'

    # ========== LOAD & CLEAN DATA ==========
    df = pd.read_csv(data_path, index_col=None, header=4)
    biomarker_columns = [col for col in df.columns if col not in ['AnimalID', 'Sex', 'Species', 'Age', 'Reason', 'Fasting', 'LabCode', 'Mg']]

    # ========== BASELINE TRANSFORMATION ==========
    data0, df0, biomarker_columns = prepare_model_data(df, kalman=kalman)
    W0, L0 = linear_regression(data0)
    mu0 = estimate_mu(L0, data0['x_cov'])

    eigenvalues0, eigenvectors0 = np.linalg.eig(W0)
    abs_order = np.argsort(np.abs(eigenvalues0))
    V_sorted = eigenvectors0[:, abs_order]
    P_inv0 = np.linalg.inv(V_sorted)

    n_z = P_inv0.shape[0]

    def reproject_to_W0_basis(df_valid, mu, P_inv, biomarker_columns):
        z_data = np.matmul(P_inv, df_valid[biomarker_columns].T.to_numpy()).T
        z_mu = np.matmul(P_inv, mu.T.to_numpy()).T
    
        z_df = pd.DataFrame(z_data.real, columns=[f'z_{i+1}' for i in range(P_inv.shape[0])])
        z_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

        z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
        z_mu_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

        return z_df, z_mu_df

    # Containers
    drift_boot = []
    beta_boot = []

    print("Running bootstraps for drift vs β...")
    unique_ids = df['AnimalID'].unique()

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        df_boot = pd.concat([df[df['AnimalID'] == aid] for aid in sampled_ids], ignore_index=True)

        data, df_boot, biomarker_columns = prepare_model_data(df_boot, kalman=kalman)
        try:
            W, L = linear_regression(data)
            mu = estimate_mu(L, data['x_cov'])

            # project into W₀ basis
            z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)

            # --- Fit CoxPH ---
            z_final_df = preprocess_final_samples(z_df_j)
            z_final_df = prepare_cox_covariates(z_final_df)
            z_cph = CoxPHFitter()
            z_cph.fit(z_final_df, duration_col='Age', event_col='event')

            # β's (signed, not abs!)
            betas_ordered = []
            for col in [f"z_{k+1}" for k in range(n_z)]:
                if col in z_cph.summary.index:
                    betas_ordered.append(z_cph.summary.loc[col, "coef"])
                else:
                    betas_ordered.append(np.nan)
            beta_boot.append(betas_ordered)

            # --- Drift slopes (slope of μ_z over Age) ---
            drift_slopes = []
            for col in [f"mu_z_{k+1}" for k in range(n_z)]:
                if col in z_mu_df_j.columns:
                    try:
                        slope, _ = np.polyfit(z_mu_df_j['Age'], z_mu_df_j[col], 1)
                    except Exception:
                        slope = np.nan
                    drift_slopes.append(slope)
                else:
                    drift_slopes.append(np.nan)
            drift_boot.append(drift_slopes)

        except Exception as e:
            print(f"Bootstrap {i} failed: {e}")
            continue

    # Convert to DataFrames
    drift_boot_df = pd.DataFrame(drift_boot, columns=[f"z_{i+1}" for i in range(n_z)])
    beta_boot_df  = pd.DataFrame(beta_boot,  columns=[f"z_{i+1}" for i in range(n_z)])

    # Means and stds
    drift_mean = drift_boot_df.mean()
    drift_std  = drift_boot_df.std()
    beta_mean  = beta_boot_df.mean()
    beta_std   = beta_boot_df.std()

    # Filter common z's
    used_cols = drift_boot_df.columns.intersection(beta_boot_df.columns)

    x_mean = drift_mean[used_cols]
    x_std  = drift_std[used_cols]
    y_mean = beta_mean[used_cols]
    y_std  = beta_std[used_cols]

    # --- Plot ---
    plt.figure(figsize=(10,6))
    plt.scatter(x_mean, y_mean, color="blue")
    
    # error bars
    plt.errorbar(
        x_mean, y_mean,
        xerr=x_std, yerr=y_std,
        fmt='o', color='blue',
        ecolor='lightgray', elinewidth=2, capsize=3
    )
    
    # annotate
    for label, xi, yi in zip(used_cols, x_mean, y_mean):
        label_str = label.replace("z_", "")
        plt.text(xi, yi, label_str, fontsize=9, ha="right", va="bottom")

    # bootstrap linear regression with conf. band
    slopes, intercepts = [], []
    for x_b, y_b in zip(drift_boot_df.values, beta_boot_df.values):
        valid = np.isfinite(x_b) & np.isfinite(y_b)
        if np.sum(valid) > 1:
            slope_b, intercept_b, _, _, _ = linregress(x_b[valid], y_b[valid])
            slopes.append(slope_b)
            intercepts.append(intercept_b)

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    x_line = np.linspace(x_mean.min(), x_mean.max(), 100)
    y_lines = np.outer(slopes, x_line) + intercepts[:, None]

    y_line_mean = np.mean(y_lines, axis=0)
    y_line_std  = np.std(y_lines, axis=0)

    plt.plot(x_line, y_line_mean, color="green", linestyle="--", label="Linear fit (bootstrap mean)")
    plt.fill_between(x_line, y_line_mean - y_line_std, y_line_mean + y_line_std, color="green", alpha=0.2)

    plt.xlabel("Steady-state Drift (slope of μ over Age)")
    plt.ylabel("Proportional Hazard Coefficient β")
    plt.title("Hazard Coefficients vs. Steady-state Drift with Bootstrap Error")
    plt.legend()
    plt.show()
    plt.clf()

bootstrap_beta_drift()
