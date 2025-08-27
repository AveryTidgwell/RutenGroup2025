import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess
from lifelines import CoxPHFitter
from datetime import datetime
start_time = datetime.now()

def bootstrap_beta_eigvals():
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
    lowess_lines = []

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

            # --- project onto the *baseline* W0 basis ---
            z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)

            # --- Fit CoxPH on z's with consistent columns ---
            z_final_df = preprocess_final_samples(z_df_j)
            z_final_df = prepare_cox_covariates(z_final_df)

            z_cph = CoxPHFitter()
            z_cph.fit(z_final_df, duration_col='Age', event_col='event')
        
            # force column order to match z_1...z_n
            betas_ordered = []
            for col in [f"z_{k+1}" for k in range(n_z)]:
                if col in z_cph.summary.index:  
                    betas_ordered.append(np.abs(z_cph.summary.loc[col, "coef"]))
                else:
                    betas_ordered.append(np.nan)  # placeholder if Cox dropped it
            beta_boot_list.append(betas_ordered)


            # --- LOWESS regression in aligned basis ---
            x_boot = 1 / np.abs(eigvals_sorted)
            y_boot = np.array(betas_ordered)

            valid_mask = np.isfinite(x_boot) & np.isfinite(y_boot)
            if np.sum(valid_mask) > 1:
                lowess_result = lowess(y_boot[valid_mask], x_boot[valid_mask], frac=0.3, it=3)
                lowess_lines.append(lowess_result)

        except Exception as e:
            print(f"Bootstrap {i} failed: {e}")
            continue

      
    # After bootstrap loop
    eigen_boot_df = pd.DataFrame(eigen_boot, columns=[f"z_{i+1}" for i in range(len(eigen_boot[0]))])
    beta_boot_df = pd.DataFrame(beta_boot_list)

    # Compute inverse absolute eigenvalues ± std
    inv_abs_eig = 1 / np.abs(eigen_boot_df)
    inv_abs_eig_mean = inv_abs_eig.mean()
    inv_abs_eig_std  = inv_abs_eig.std()

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
    
    plt.scatter(x_mean, y_mean, marker='o', color='blue')
    '''
    # Plot error bars
    plt.errorbar(
        x_mean,
        y_mean,
        xerr=x_std,
        yerr=y_std,
        fmt='o',
        color='blue',
        ecolor='lightgray',
        elinewidth=2,
        capsize=3
    )
    '''
    # Add labels
    for label, xi, yi in zip(used_cols, x_mean, y_mean):
        label_str = str(label)
        label_str = label_str.replace('z_', '')
        label_str = str(int(label_str) + 1)
        plt.text(xi, yi, label_str, fontsize=9, ha="right", va="bottom")

    # Process LOWESS lines
    if lowess_lines:
        # Create common x grid for interpolation
        x_grid = np.linspace(min(x_mean.min(), *[min(line[:, 0]) for line in lowess_lines]),
                           max(x_mean.max(), *[max(line[:, 0]) for line in lowess_lines]), 100)
        
        # Interpolate each LOWESS line to common x grid
        interpolated_y = []
        for lowess_line in lowess_lines:
            interp_y = np.interp(x_grid, lowess_line[:, 0], lowess_line[:, 1])
            interpolated_y.append(interp_y)
        
        interpolated_y = np.array(interpolated_y)
        y_mean_lowess = np.mean(interpolated_y, axis=0)
        y_std_lowess = np.std(interpolated_y, axis=0)
        
        
        # --- Bootstrap Linear Regression with Confidence Band ---
        slopes, intercepts = [], []
        for x_b, y_b in zip(inv_abs_eig.values, beta_boot_df.values):
            valid_mask = np.isfinite(x_b) & np.isfinite(y_b)
            if np.sum(valid_mask) > 1:
                slope_b, intercept_b, _, _, _ = linregress(x_b[valid_mask], y_b[valid_mask])
                slopes.append(slope_b)
                intercepts.append(intercept_b)

        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        # Evaluate all regression lines on a common grid
        x_line = np.linspace(x_mean.min(), x_mean.max(), 100)
        y_lines = np.outer(slopes, x_line) + intercepts[:, None]

        # Mean and std across bootstrap regressions
        y_mean_line = np.mean(y_lines, axis=0)
        y_std_line = np.std(y_lines, axis=0)

        # Plot mean regression line
        plt.plot(x_line, y_mean_line, color='green', linestyle='--', label='Linear fit (bootstrap mean)')

        # Shaded confidence interval
        plt.fill_between(
            x_line,
            y_mean_line - y_std_line,
            y_mean_line + y_std_line,
            color='green',
            alpha=0.2,
            label='Linear fit ±1σ'
        )

        '''
        
        # Plot mean LOWESS line
        plt.plot(x_grid, y_mean_lowess, color='red', label='Mean LOWESS')
        
        # Plot 1 std confidence interval
        plt.fill_between(x_grid, 
                        y_mean_lowess - y_std_lowess,
                        y_mean_lowess + y_std_lowess,
                        color='red', alpha=0.2, label='1σ Confidence')
        '''
    plt.xlabel("Auto-correlation Time, 1 / |λ| (yrs)")
    plt.ylabel("Proportional Hazard Coefficient, |β|")
    plt.title("Hazard Coefficients vs. Auto-correlation Times with Bootstrap Error")
    plt.legend()
    plt.show()
    plt.clf()

bootstrap_beta_eigvals()






