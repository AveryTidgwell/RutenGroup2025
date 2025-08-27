import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_dolphin_counts_by_age(df_valid, bin_width=0.5, smooth_frac=0.2):
    # Round AnimalID to the nearest integer
    df = df_valid.copy()
    df['AnimalID'] = df['AnimalID'].round().astype(int)

    # Create 0.5-year bins for Age
    df['AgeBin'] = (df['Age'] // bin_width) * bin_width

    # Count unique AnimalIDs per AgeBin
    dolphin_counts = df.groupby('AgeBin')['AnimalID'].nunique().reset_index()
    dolphin_counts.rename(columns={'AnimalID': 'UniqueDolphins'}, inplace=True)

    # LOWESS smoothing
    smoothed = lowess(dolphin_counts['UniqueDolphins'], dolphin_counts['AgeBin'], frac=smooth_frac)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.scatter(dolphin_counts['AgeBin'], dolphin_counts['UniqueDolphins'], color='blue', alpha=0.6, label='Raw Counts')
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='lightseagreen', linewidth=2, label='LOWESS Smooth')

    plt.xlabel('Age (years)')
    plt.ylabel('Number of Unique Dolphins')
    plt.title('Number of Unique Dolphins Sampled by Age')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_dolphin_counts_by_age(df_valid, bin_width=0.5, smooth_frac=0.2)
