import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from tqdm import tqdm

def preprocess_mortality(df, z_df):
    df_last = df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()
    #z_df = z_df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()
    #df_last = df_last.merge(z_df, on=['dog_id', 'Age', 'Fixed', 'Sex'], how='left')
    df_last['event'] = df_last['death_age'].notna().astype(int)
    df_last['death_age'] = df_last['death_age'].fillna(df_last['Age'])
    
    return df_last.reset_index(drop=True)

def prepare_cox_covariates_iterative(df, duration_col='death_age', event_col='event', corr_thresh=0.95, reference_columns=None):
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    drop_cols = [duration_col, event_col, 'dog_id', 'Fixed', 'Age', 'Age_norm']
    
    covariates = df_numeric.drop(columns=[c for c in drop_cols if c in df_numeric.columns], errors='ignore')
    
    # Drop zero-variance columns
    covariates = covariates.loc[:, covariates.nunique() > 1]

    if reference_columns is None:
        while True:
            if covariates.shape[1] < 2:
                # Not enough covariates to check correlation; exit loop
                break

            corr_matrix = covariates.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Check if upper matrix is empty or all NaN
            if upper.isnull().all().all():
                # No correlations to check
                break

            max_corr = upper.max().max()
            if pd.isnull(max_corr) or max_corr <= corr_thresh:
                break
            
            # Identify the first pair with the max correlation
            to_drop_col = upper.stack().idxmax()[1]  # second column in the pair
            print(f"Dropping '{to_drop_col}' due to correlation {max_corr:.3f}")
            covariates = covariates.drop(columns=[to_drop_col])
        
        final_cols = covariates.columns.tolist()
    else:
        covariates = covariates[[col for col in reference_columns if col in covariates.columns]]
        final_cols = reference_columns

    return pd.concat(
        [df[[duration_col, event_col]].reset_index(drop=True),
         covariates.reset_index(drop=True)], axis=1
    ), final_cols



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
            train_clean, feature_cols = prepare_cox_covariates_iterative(train_df, corr_thresh=corr_thresh)
            test_clean, _ = prepare_cox_covariates_iterative(test_df, reference_columns=feature_cols)

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


train_scores, test_scores = bootstrap_cox_cindex_z(df, z_df, n_bootstraps=1000)

print(f"\nTrain C-index: Mean = {np.mean(train_scores):.3f}, SD = {np.std(train_scores):.3f}")
print(f"Test  C-index: Mean = {np.mean(test_scores):.3f}, SD = {np.std(test_scores):.3f}")

death_df = preprocess_mortality(df, z_df)
death_df, final_cols = prepare_cox_covariates_iterative(death_df)
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


def prepare_cox_covariates_z(df, z_df, duration_col='death_age', event_col='event', corr_thresh=0.95):
    df = preprocess_mortality(df, z_df)
    # Merge survival info with z variables by dog_id and Age
    merged = df[['dog_id', 'Age', duration_col, event_col]].merge(
        z_df.drop(columns=[col for col in z_df.columns if 'mu' in col or col in ['Fixed', 'Sex']]),
        on=['dog_id', 'Age'], how='inner'
    )

    # Filter numeric covariates only, exclude duration/event columns
    covariates = merged.select_dtypes(include=[np.number]).copy()
    drop_cols = [duration_col, event_col]
    covariates = covariates.drop(columns=drop_cols, errors='ignore')

    # Drop zero variance columns
    covariates = covariates.loc[:, covariates.nunique() > 1]

    # Iterative correlation removal
    while True:
        if covariates.shape[1] < 2:
            break
        corr_matrix = covariates.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        if upper.isnull().all().all():
            break
        max_corr = upper.max().max()
        if pd.isnull(max_corr) or max_corr <= corr_thresh:
            break
        to_drop_col = upper.stack().idxmax()[1]
        print(f"Dropping {to_drop_col} due to correlation {max_corr:.3f}")
        covariates = covariates.drop(columns=[to_drop_col])

    # Combine duration/event, dog_id, Age and covariates
    final_df = pd.concat(
        [merged[['dog_id', 'Age', duration_col, event_col]].reset_index(drop=True),
         covariates.reset_index(drop=True)], axis=1)

    return final_df



def bootstrap_cox_cindex_z(df, z_df, n_bootstraps=100, test_size=0.2, corr_thresh=0.95, verbose=True):
    # First merge and prepare full dataset with z-covariates
    full_df = prepare_cox_covariates_z(df, z_df, corr_thresh=corr_thresh)

    train_cindex_list = []
    test_cindex_list = []
    print(type(full_df['dog_id']))

    dog_ids = full_df['dog_id']
    if isinstance(dog_ids, pd.DataFrame):
        dog_ids = dog_ids.iloc[:, 0]
    dog_ids = dog_ids.unique()



    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        try:
            train_ids, test_ids = train_test_split(dog_ids, test_size=test_size)

            train_df = full_df[full_df['dog_id'].isin(train_ids)].copy()
            test_df = full_df[full_df['dog_id'].isin(test_ids)].copy()

            # Drop dog_id before fitting
            train_data = train_df.drop(columns=['dog_id'])
            test_data = test_df.drop(columns=['dog_id'])

            if train_data.shape[1] < 3:
                if verbose:
                    print("Skipping bootstrap due to insufficient covariates.")
                continue

            cph = CoxPHFitter()
            cph.fit(train_data, duration_col='death_age', event_col='event')

            train_cindex_list.append(cph.concordance_index_)

            partial_hazards = cph.predict_partial_hazard(test_data)
            test_cindex = concordance_index(test_data['death_age'], -partial_hazards, test_data['event'])
            test_cindex_list.append(test_cindex)

        except Exception as e:
            if verbose:
                print("Bootstrap failed:", e)
            continue

    return train_cindex_list, test_cindex_list


