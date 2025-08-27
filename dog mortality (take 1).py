import math
def preprocess_mortality(df):
    # Sort and keep only the last sample per dog
    df_last = df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()

    # Create event column: 1 if death_age is known, 0 if not
    df_last['event'] = df_last['death_age'].notna().astype(int)

    # For censored dogs (missing death_age), use their current age as the censoring time
    df_last['death_age'] = df_last['death_age'].fillna(df_last['Age'])

    # Optional: rename to indicate it's been preprocessed for survival
    death_df = df_last.reset_index(drop=True)

    return death_df

death_df = preprocess_mortality(df)



def prepare_cox_covariates(df, duration_col='death_age', event_col='event', corr_thresh=0.95):
    # Keep only numeric data
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Remove duration and event from covariates
    covariates = df_numeric.drop(columns=[duration_col, event_col, 'dog_id', 'Fixed', 'Age', 'Age_norm', 'hgb'], errors='ignore')

    # Drop columns with zero variance
    covariates = covariates.loc[:, covariates.nunique() > 1]

    # Drop highly correlated variables
    corr_matrix = covariates.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    print(f"Dropping {len(to_drop)} highly correlated columns:\n", to_drop)

    covariates = covariates.drop(columns=to_drop)

    # Return a cleaned dataframe for CoxPHFitter
    return pd.concat([df[[duration_col, event_col]].reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)

death_df_clean = prepare_cox_covariates(death_df)

from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(death_df_clean, duration_col='death_age', event_col='event')
cph.print_summary()




import matplotlib.pyplot as plt

def plot_cox_results(cph_model, top_n=41, show_hr=False):
    summary = cph_model.summary.sort_values(by='p', ascending=True).head(top_n)

    plt.figure(figsize=(8, 6))
    if show_hr:
        values = summary['exp(coef)']  # Hazard Ratio
        errors = summary['exp(coef) upper 95%'] - summary['exp(coef)']
        label = 'Hazard Ratio'
    else:
        values = summary['coef']       # Log Hazard Ratio
        errors = summary['se(coef)']
        label = 'Log Hazard Ratio'

    plt.barh(summary.index, values, xerr=errors, align='center')
    plt.axvline(x=0 if not show_hr else 1, color='black', linestyle='--')
    plt.xlabel(label)
    plt.title('Top Significant Predictors in Cox Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

plot_cox_results(cph)         # Log hazard ratios
plot_cox_results(cph, show_hr=True)  # Hazard ratios

cph.check_assumptions(death_df_clean, p_value_threshold=0.05)

