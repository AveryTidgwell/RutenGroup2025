import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from tqdm import tqdm

def preprocess_mortality(df, z_df):
    df_last = df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()
    df_last['Age'] = df_last['Age'].round(2)
    #z_df = z_df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()
    #z_df = z_df[[col for col in z_df.columns if 'mu' not in col and 'x' not in col]]
    #z_df['dog_id'] = z_df['dog_id'].round(0)
    #z_df['Age'] = z_df['Age'].round(2)
    #df_last = df_last.merge(z_df, on='Age', how='left')
    df_last['event'] = df_last['death_age'].notna().astype(int)
    df_last['death_age'] = df_last['death_age'].fillna(df_last['Age'])
    #df_last = df_last.drop(columns=['dog_id_y'])
    #df_last.rename(columns={'dog_id_x': 'dog_id'})
    return df_last.reset_index(drop=True)

def prepare_cox_covariates(df, duration_col='death_age', event_col='event', corr_thresh=0.95, reference_columns=None):
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [duration_col, event_col, 'dog_id', 'Fixed', 'Age', 'Age_norm']
    covariates = df_numeric.drop(columns=[c for c in drop_cols if c in df_numeric.columns], errors='ignore')

    covariates = covariates.loc[:, covariates.nunique() > 1]  # Drop zero-variance
    if reference_columns is None:
        corr_matrix = covariates.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
        covariates = covariates.drop(columns=to_drop)
        final_cols = covariates.columns.tolist()
    else:
        covariates = covariates[[col for col in reference_columns if col in covariates.columns]]
        final_cols = reference_columns

    return pd.concat([df[[duration_col, event_col]].reset_index(drop=True), covariates.reset_index(drop=True)], axis=1), final_cols

def bootstrap_cox_cindex(df, n_bootstraps=100, test_size=0.2, corr_thresh=0.95, verbose=True):
    df_proc = preprocess_mortality(df, z_df)

    train_cindex_list = []
    test_cindex_list = []

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Split by dog_id
        dog_ids = df_proc['dog_id'].unique()
        train_ids, test_ids = train_test_split(dog_ids, test_size=test_size)
        train_df = df_proc[df_proc['dog_id'].isin(train_ids)].copy()
        test_df = df_proc[df_proc['dog_id'].isin(test_ids)].copy()

        # Prepare covariates (training first)
        try:
            train_clean, feature_cols = prepare_cox_covariates(train_df, corr_thresh=corr_thresh)
            test_clean, _ = prepare_cox_covariates(test_df, reference_columns=feature_cols)

            if train_clean.shape[1] < 3:  # Need at least 1 covariate + duration/event
                if verbose:
                    print("Skipping bootstrap due to insufficient covariates.")
                continue

            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(train_clean, duration_col='death_age', event_col='event')

            # Evaluate on train
            train_cindex_list.append(cph.concordance_index_)

            # Evaluate on test
            partial_hazards = cph.predict_partial_hazard(test_clean)
            test_cindex = concordance_index(
                test_clean['death_age'],
                -partial_hazards,
                test_clean['event']
            )
            test_cindex_list.append(test_cindex)
        except Exception as e:
            if verbose:
                print("Bootstrap failed:", e)
            continue

    return train_cindex_list, test_cindex_list


train_scores, test_scores = bootstrap_cox_cindex(df, n_bootstraps=1000)

print(f"\nTrain C-index: Mean = {np.mean(train_scores):.3f}, SD = {np.std(train_scores):.3f}")
print(f"Test  C-index: Mean = {np.mean(test_scores):.3f}, SD = {np.std(test_scores):.3f}")

death_df = preprocess_mortality(df, z_df)
death_df = prepare_cox_covariates(death_df)[0]
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(death_df, duration_col='death_age', event_col='event')
cph.print_summary()
check_nan(death_df)






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


import random
import matplotlib.pyplot as plt

def plot_individual_survivals(cph, df, id_col="dog_id", duration_col="duration", event_col="event", n_ids=3):
    """
    Plots survival curves for n_ids individuals with an event=1 outcome,
    along with vertical lines marking their actual death time.
    
    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        A fitted Cox proportional hazards model.
    df : pd.DataFrame
        Dataset containing covariates, id_col, duration_col, event_col.
    id_col : str
        Column name for individual IDs.
    duration_col : str
        Column name for the survival time / age.
    event_col : str
        Column name for the event indicator (1 = death, 0 = censored).
    n_ids : int
        Number of individuals to plot.
    """
    
    # Filter to individuals who died
    died_df = df[df[event_col] == 1]
    unique_ids = died_df[id_col].unique()
    
    if len(unique_ids) < n_ids:
        raise ValueError(f"Not enough death events to sample {n_ids} individuals.")
    
    # Randomly pick individuals
    chosen_ids = random.sample(list(unique_ids), n_ids)
    
    plt.figure(figsize=(10, 6))
    
    for i, cid in enumerate(chosen_ids):
        indiv_data = df[df[id_col] == cid]
        death_age = indiv_data[duration_col].max()
        
        # Get last observation (covariates just before/at death)
        last_obs = indiv_data[indiv_data[duration_col] == death_age]
        
        # Drop survival-specific columns before prediction
        covariates = last_obs.drop(columns=[id_col, duration_col, event_col])
        
        # Predict survival function
        surv_func = cph.predict_survival_function(covariates)
        
        # Plot survival curve
        plt.plot(surv_func.index, surv_func.iloc[:, 0], label=f"{cid}")
        
        # Add vertical line at death age
        plt.axvline(death_age, color=plt.gca().lines[-1].get_color(),
                    linestyle="--", alpha=0.7)
    
    plt.xlabel("Time / Age")
    plt.ylabel("Predicted Survival Probability")
    plt.title("Individual Survival Predictions vs Actual Death")
    plt.legend(title=id_col)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_individual_survivals(cph, train_df, id_col="dog_id", duration_col="death_age", event_col="event", n_ids=3)



