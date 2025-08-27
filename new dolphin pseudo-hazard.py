import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm

def preprocess_final_samples(df):
    # Sort and grab latest sample for each dolphin
    df = df[df['AnimalID'].map(df['AnimalID'].value_counts()) > 1]
    final_df = df.sort_values(by=["AnimalID", "Age"]).groupby("AnimalID").tail(1).copy()

    # Drop any that are missing Age or biomarker data
    final_df = final_df.dropna(subset=["Age"])
    final_df['event'] = 1
    
    return final_df

def prepare_cox_covariates(df, duration_col='Age', event_col='event', corr_thresh=0.95):
    # Keep only numeric data
    df_numeric = df.select_dtypes(include=[np.number]).copy()

    # Remove duration and event from covariates
    covariates = df_numeric.drop(columns=[duration_col, event_col, 'AnimalID', 'Species', 'Age', 'Sex', 'Age_norm', 'ones'], errors='ignore')

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






import seaborn as sns
import matplotlib.pyplot as plt

def plot_cox_results(cph_model, top_n=41, show_hr=False, sorter='p'):
    if sorter == 'coef_minus_se':
    
        summary = cph_model.summary.copy()
    
        # Create the sort metric
        summary["coef_minus_se"] = abs(summary["coef"]) - summary["se(coef)"]
    
        summary = summary.sort_values(by='coef_minus_se', ascending=False).head(top_n)
    else:
        summary = cph_model.summary.sort_values(by=sorter, ascending=True).head(top_n)
    
    plt.figure(figsize=(8, 6))
    if show_hr:
        values = summary['exp(coef)']  # Hazard Ratio
        errors = summary['exp(coef) upper 95%'] - summary['exp(coef)']
        label = 'Hazard Ratio'
    else:
        values = summary['coef']       # Log Hazard Ratio
        errors = summary['se(coef)']
        label = 'Log Hazard Ratio'

    plt.barh(summary.index, values, xerr=errors, color='orange', align='center')
    plt.axvline(x=0 if not show_hr else 1, color='black', linestyle='--')
    plt.xlabel(label)
    plt.title('Top Significant Predictors in Cox Model')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

plot_cox_results(cph, sorter='coef_minus_se')         # Log hazard ratios
#plot_cox_results(cph, show_hr=True)  # Hazard ratios







# # # ----- C-INDEX TESTING ----- # # #

import numpy as np
from lifelines.utils import concordance_index

def c_index_bootstrap_pairwise(final_df, cph_model, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    np.random.seed(random_seed)
    
    # Shuffle and split into train/test
    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_size = int(len(final_df) * (1 - test_frac))
    train_df = final_df.iloc[:train_size]
    test_df = final_df.iloc[train_size:]
    
    # Fit model on train
    cph_model.fit(train_df, duration_col='Age', event_col='event')
    
    # Compute predicted risk scores on test set
    # Higher risk score => higher hazard
    test_df['risk_score'] = cph_model.predict_partial_hazard(test_df)
    
    # Prepare results list for bootstrap C-indices
    bootstrap_c_indices = []
    
    # For each bootstrap iteration, sample pairs of rows from test_df and compute pairwise concordance
    for _ in tqdm(range(n_bootstrap), desc="Bootstrap C-index pairs"):
        # Randomly sample 2 rows from test_df without replacement
        pair = test_df.sample(n=2, replace=False)
        
        # Extract data
        times = pair['Age'].values
        events = pair['event'].values
        risks = pair['risk_score'].values
        
        # We only consider pairs where one dies before the other (different survival times)
        if times[0] == times[1]:
            # Tie in survival times, skip
            continue
        
        # Check if the higher risk predicts the one who died earlier
        # Concordant if:
        #   risk_i > risk_j and time_i < time_j
        # or risk_j > risk_i and time_j < time_i
        if (risks[0] > risks[1] and times[0] < times[1]) or (risks[1] > risks[0] and times[1] < times[0]):
            concordant = 1
        else:
            concordant = 0
        
        bootstrap_c_indices.append(concordant)
    
    # Final bootstrap C-index is mean concordance across all sampled pairs
    c_index_estimate = np.mean(bootstrap_c_indices)
    c_index_sd = np.std(bootstrap_c_indices)
    print(f"Bootstrap estimated C-index (pairwise): {c_index_estimate:.4f} +/- {c_index_sd:.4f}")
    
    return c_index_estimate, bootstrap_c_indices

# Usage:
cph = CoxPHFitter()
final_df = preprocess_final_samples(df)  # Your existing preprocessing function
final_df = prepare_cox_covariates(final_df)
cph.fit(final_df, duration_col='Age', event_col='event')
cph.plot()
c_index, all_concordances = c_index_bootstrap_pairwise(final_df, cph, n_bootstrap=10000, test_frac = 0.3)


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lifelines import CoxPHFitter

def run_c_index_reduction(input_df, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    final_df = preprocess_final_samples(input_df)  # Your existing preprocessing function
    final_df = prepare_cox_covariates(final_df)
    np.random.seed(random_seed)
    excluded_cols = ['Age', 'event']
    all_covariates = [c for c in final_df.columns if c not in excluded_cols]
    results = []
    current_covariates = all_covariates.copy()

    while len(current_covariates) > 1:
        print(f"\nRunning with {len(current_covariates)} covariates...")
        try:
            bootstrap_c_indices = []

            for _ in range(n_bootstrap):
                df_shuffled = final_df.sample(frac=1, random_state=np.random.randint(0, 1e6)).reset_index(drop=True)
                train_size = int(len(df_shuffled) * (1 - test_frac))
                train_df = df_shuffled.iloc[:train_size]
                test_df = df_shuffled.iloc[train_size:]

                cph = CoxPHFitter()
                cph.fit(train_df[['Age', 'event'] + current_covariates],
                        duration_col='Age', event_col='event')

                test_df = test_df[['Age', 'event'] + current_covariates].copy()
                test_df['risk_score'] = cph.predict_partial_hazard(test_df)

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

            mean_c = np.mean(bootstrap_c_indices)
            sd_c = np.std(bootstrap_c_indices)
            print(f"{len(current_covariates)} vars: C-index = {mean_c:.4f} ± {sd_c:.4f}")
            results.append((len(current_covariates), mean_c, sd_c))

        except Exception:
            break

        drop_var = np.random.choice(current_covariates)
        print(f'Dropping {drop_var}')
        current_covariates.remove(drop_var)

    return np.array(results)


def c_index_boot_plot_compare(df, z_df, n_bootstrap=1000, test_frac=0.2, random_seed=42):
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
    #plt.gca().tick_params(axis='x', which='both', top=True, bottom=True)

    plt.xlabel("Number of covariates")
    plt.ylabel("Bootstrap C-index")
    plt.title("C-index vs Number of Covariates: Raw vs Z-transformed")
    plt.gca().invert_xaxis()
    plt.grid(False)
    plt.legend()
    plt.show()

    return results_df, results_zdf

c_index_boot_plot_compare(df, z_df, n_bootstrap=1000, test_frac=0.2, random_seed=42)




def prepare_dual_covariates(df, z_df, duration_col='Age', event_col='event', corr_thresh=0.95):
    # Keep only numeric data
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    z_df_numeric = z_df.select_dtypes(include=[np.number]).copy()
    
    df_numeric = df_numeric.merge(z_df_numeric, on=['Age', 'Sex'], how='left')
    
    # Remove duration and event from covariates
    covariates = df_numeric.drop(columns=[duration_col, event_col, 'AnimalID', 'Species', 'Age', 'Sex', 'Age_norm', 'ones'], errors='ignore')

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

cph = CoxPHFitter()
final_df, final_z_df = preprocess_final_samples(df), preprocess_final_samples(z_df, type_df = 'z')  # Your existing preprocessing function
final_df = prepare_dual_covariates(final_df, final_z_df)
cph.fit(final_df, duration_col='Age', event_col='event')
plot_cox_results(cph, sorter='coef_minus_se')
