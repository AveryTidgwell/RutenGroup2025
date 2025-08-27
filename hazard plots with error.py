def do_it():

    n_bootstraps = 200
    kalman = False
    output_pdf = f'bootstrap_ids_({start_time}).pdf'
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
    sorted_eigenvalues = eigenvalues0[abs_order]

    n_z = P_inv0.shape[0]
    z_col_rank_map = [f'z_{i+1}' for i in range(n_z)]
    mu_col_rank_map = [f'mu_z_{i+1}' for i in range(n_z)]

    def reproject_to_W0_basis(df_valid, mu, P_inv, biomarker_columns):
        z_data = np.matmul(P_inv, df_valid[biomarker_columns].T.to_numpy()).T
        z_mu = np.matmul(P_inv, mu.T.to_numpy()).T
    
        z_df = pd.DataFrame(z_data.real, columns=[f'z_{i+1}' for i in range(P_inv.shape[0])])
        z_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

        z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
        z_mu_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

        return z_df, z_mu_df

    z_df0, z_mu_df0 = reproject_to_W0_basis(data0['df_valid'], mu0, P_inv0, biomarker_columns)
    z_df0 = imputation(z_df0, imputation_type='mean')

    z_df_list = []
    z_mu_df_list = []
    eigen_boot = []
    beta_boot_list = []

    print("Running bootstraps by AnimalID...")
    unique_ids = df['AnimalID'].unique()

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        df_boot = pd.concat([df[df['AnimalID'] == aid] for aid in sampled_ids], ignore_index=True)

        data, df_boot, biomarker_columns = prepare_model_data(df_boot, kalman=kalman)
        try:
            W, L = linear_regression(data)
            mu = estimate_mu(L, data['x_cov'])

            eigvals, _ = np.linalg.eig(W)
            eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]
            eigen_boot.append(np.real(eigvals_sorted))

            z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)
            z_df_list.append(z_df_j)
            z_mu_df_list.append(z_mu_df_j)
        
            z_cph = CoxPHFitter()
            z_final_df = preprocess_final_samples(z_df_j)
            z_final_df = prepare_cox_covariates(z_final_df)
            z_cph.fit(z_final_df, duration_col='Age', event_col='event')
            z_cph.print_summary()
            z_cph_summary = z_cph.summary
            beta_abs = np.abs(z_cph_summary['coef'])
            beta_boot_list.append(beta_abs)

        except:
            continue
      
    # After bootstrap loop
    eigen_boot_df = pd.DataFrame(eigen_boot, columns=[f"z_{i+1}" for i in range(len(eigen_boot[0]))])
    beta_boot_df = pd.DataFrame(beta_boot_list)

    # Compute inverse absolute eigenvalues ± std
    inv_abs_eig_mean = 1 / np.abs(eigen_boot_df).mean()
    inv_abs_eig_std = 1 / np.abs(eigen_boot_df).std()
    beta_mean = beta_boot_df.mean()
    beta_std = beta_boot_df.std()

    # Debug: Check computed values
    print("inv_abs_eig_mean:", inv_abs_eig_mean)
    print("beta_mean:", beta_mean)

    # Use columns from beta_boot_df for consistency
    used_cols = beta_boot_df.columns
    print("used_cols:", used_cols)

    # Filter data
    x_mean = inv_abs_eig_mean[used_cols]
    x_std = inv_abs_eig_std[used_cols]
    y_mean = beta_mean[used_cols]
    y_std = beta_std[used_cols]

    # Debug: Check filtered data
    print("x_mean:", x_mean)
    print("y_mean:", y_mean)

    plt.figure(figsize=(10, 6))

    # Plot error bars
    plt.errorbar(
        x_mean,
        y_mean,
        xerr=x_std,
        yerr=y_std,  # Include y_std for error bars
        fmt='o',
        color='blue',
        ecolor='lightgray',
        elinewidth=2,
        capsize=3
    )

    # Add labels
    for label, xi, yi in zip(used_cols, x_mean, y_mean):
        plt.text(xi, yi, label.replace('z_', ''), fontsize=9, ha="right", va="bottom")

    # Optional regression line (only if enough data)
    if len(x_mean) > 1 and not x_mean.isna().any() and not y_mean.isna().any():
        slope, intercept, r_value, p_value, std_err = linregress(x_mean, y_mean)
        line_x = np.linspace(x_mean.min(), x_mean.max(), 100)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, color="grey", linestyle="--")
    else:
        print("Not enough valid data for regression line")

    plt.xlabel("Auto-correlation Time, 1 / |λ| (yrs)")
    plt.ylabel("Proportional Hazard Coefficient, |β|")
    plt.title("Hazard Coefficients vs. Auto-correlation Times with Bootstrap Error")
    plt.show()
    plt.clf()
 
 
 
 
do_it() 

 
 
 

 
 
    
