import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm

def preprocess_final_samples(df):
    # Sort and grab latest sample for each dolphin
    final_df = df.sort_values(by=["dog_id", "Age"]).groupby("dog_id").tail(1).copy()

    # Drop any that are missing Age or biomarker data
    final_df = final_df.dropna(subset=["Age"])

    final_df = final_df.drop(columns=['ones', 'Age_norm'])

    return final_df

def bootstrap_survival_dataset(df, avg_lifespan=32.5, annual_death_rate=0.027, n_bootstraps=1000, seed=42):
    np.random.seed(seed)
    boot_datasets = []

    unique_ids = df['dog_id'].unique()

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping pseudo-events"):
        #np.random.seed(i)
        # Sample dolphins WITH replacement
        sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
        boot_df = df[df['dog_id'].isin(sampled_ids)].copy()

        death_probs = 1 - np.exp(-annual_death_rate * boot_df['Age'])
        # Reassign pseudo-events randomly
        boot_df['event'] = np.random.binomial(1, death_probs, size=len(boot_df))

        # Age is used as survival time (event or censoring)
        boot_df['duration'] = boot_df['Age']

        boot_datasets.append(boot_df)
      
    return boot_datasets


def fit_cox_models(boot_datasets, duration_col="duration", event_col="event"):
    summaries = []
    cph = CoxPHFitter()

    for df in tqdm(boot_datasets, desc="Fitting Cox models"):
        try:
            df = df.dropna(axis=1)  # Drop any leftover NA columns
            cph.fit(df, duration_col=duration_col, event_col=event_col)
            summary = cph.summary.reset_index()
            summary['biomarker'] = summary['covariate']
            summaries.append(summary[['biomarker', 'coef', 'se(coef)', 'p']])
        except Exception as e:
            continue  # Skip failed fits

    return pd.concat(summaries, ignore_index=True)

def summarize_bootstrap_cox(results_df):
    # Aggregate across bootstraps
    return results_df.groupby('biomarker').agg(
        mean_coef=('coef', 'mean'),
        std_coef=('coef', 'std'),
        mean_p=('p', 'mean')
    ).sort_values(by='mean_coef', ascending=False)





# 1. Take only last sample per dolphin
final_df = preprocess_final_samples(df)

# 2. Create bootstrap pseudo-event datasets
boot_data = bootstrap_survival_dataset(final_df, avg_lifespan=11.23, annual_death_rate=0.079, n_bootstraps=1000)

# 3. Fit Cox models
results_df = fit_cox_models(boot_data)

# 4. Summarize effects
summary_df = summarize_bootstrap_cox(results_df)
print(summary_df)

import seaborn as sns
import matplotlib.pyplot as plt

def plot_hazard_summary(summary_df):
    summary_df = summary_df.sort_values(by="mean_coef", ascending=False)

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(summary_df))

    plt.barh(
        y=y_pos,
        width=summary_df["mean_coef"],
        xerr=summary_df["std_coef"],
        align='center',
        color='skyblue',
        ecolor='black',
        capsize=4
    )
    plt.axvline(0, color='grey', linestyle='--')
    plt.yticks(y_pos, summary_df.index)
    plt.xlabel("Average Cox Coefficient")
    plt.ylabel("Biomarker")
    plt.title("Estimated Hazard Influence of Biomarkers (Bootstrapped)")
    plt.tight_layout()
    plt.show()

plot_hazard_summary(summary_df)
