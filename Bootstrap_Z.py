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

def bootstrap_lowess(x, y, frac=0.3, n_boot=1000, alpha=0.05):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.array(x)
    y = np.array(y)
    x_eval = np.linspace(x.min(), x.max(), 100)
    all_smooths = []

    failed = 0

    for _ in range(n_boot):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_sample = x[indices]
        y_sample = y[indices]

        if len(np.unique(x_sample)) < 2:
            failed += 1
            print(failed)
            continue  # LOWESS requires variability in x

        try:
            smoothed = lowess(y_sample, x_sample, frac=frac, xvals=x_eval)
            all_smooths.append(smoothed)
        except Exception:
            failed += 1
            continue

    if len(all_smooths) < 10:
        raise ValueError(f"LOWESS bootstrap failed too often ({failed} failures, only {len(all_smooths)} successful). Consider reducing `frac` or checking data.")

    all_smooths = np.array(all_smooths)
    lower = np.percentile(all_smooths, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_smooths, 100 * (1 - alpha / 2), axis=0)
    mean = np.median(all_smooths, axis=0)

    return x_eval, mean, lower, upper


  



import statsmodels.api as sm

def bootstrap_z(df, mu):
  z_cols = [col for col in z_df.columns if col.startswith("z_") and not col.startswith("mu_")]

  for z_col in z_cols:
    fig, ax = plt.subplots()

    maleAge, maleBio = [], []
    femaleAge, femaleBio = [], []

    for idx, sex in enumerate(df['Sex']):
        biomarker_val = df[z_col].iloc[idx]
        age_val = df['Age'].iloc[idx]

        if np.isnan(biomarker_val) or np.isnan(age_val):
            continue

        if sex == 'F' or sex == 0:
            femaleBio.append(biomarker_val)
            femaleAge.append(age_val)
        elif sex == 'M' or sex == 1:
            maleBio.append(biomarker_val)
            maleAge.append(age_val)
        else:
            continue

    # Plot scatter
    ax.scatter(maleAge, maleBio, color='C0', alpha = 0.4, label='Male')
    ax.scatter(femaleAge, femaleBio, color='C6', alpha = 0.4, label='Female')

    # Male LOWESS + CI
    if len(maleAge) >= 5:
        m_x_eval, m_mean, m_lower, m_upper = bootstrap_lowess(maleAge, maleBio, frac=0.2, n_boot=300)
        ax.plot(m_x_eval, m_mean, color='black', linestyle='-', linewidth=2.5, label='LOWESS (Male)')
        #ax.fill_between(m_x_eval, m_lower, m_upper, color='black', alpha=0.2, label='95% CI (Male)')

    # Female LOWESS + CI
    if len(femaleAge) >= 5:
        f_x_eval, f_mean, f_lower, f_upper = bootstrap_lowess(femaleAge, femaleBio, frac=0.2, n_boot=300)
        ax.plot(f_x_eval, f_mean, color='red', linestyle='-', linewidth=2.5, label='LOWESS (Female)')
        #ax.fill_between(f_x_eval, f_lower, f_upper, color='red', alpha=0.2, label='95% CI (Female)')

    # Male bootstrap linear fit
    if len(maleAge) >= 2:
        m_x_fit, m_mean, m_lower, m_upper = bootstrap_regression(maleAge, maleBio)
        ax.plot(m_x_fit, m_mean, color='black', linestyle='--', label='Male Linear Fit (Bootstrap)')
        ax.fill_between(m_x_fit, m_lower, m_upper, color='black', alpha=0.2)
    else:
        print("Not enough male data points for bootstrap fit.")

    # Female bootstrap linear fit
    if len(femaleAge) >= 2:
        f_x_fit, f_mean, f_lower, f_upper = bootstrap_regression(femaleAge, femaleBio)
        ax.plot(f_x_fit, f_mean, color='red', linestyle='--', label='Female Linear Fit (Bootstrap)')
        ax.fill_between(f_x_fit, f_lower, f_upper, color='red', alpha=0.2)
    else:
        print("Not enough female data points for bootstrap fit.")


    plt.title(z_col + ' Over Age (Bootstrap)')
    plt.xlabel('Age')
    plt.ylabel(z_col)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    # === Save Plot ===
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'z_Bootstrap')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{z_col}_mvf_boot_over_age.png")
    #plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")

    plt.clf()
    del fig

# Run
bootstrap_z(z_df2, z_mu_df2)

