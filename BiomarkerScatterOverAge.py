import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# === Folder path ===
output_dir = os.path.expanduser('~/Downloads/dolphins-master/results/biomarker_plots(2-98%)')

# === Create folder if missing ===
os.makedirs(output_dir, exist_ok=True)

# Select biomarker columns (columns 8 to 51)
biomarker_columns = df.columns[8:52]  # Adjust as needed

#Optional slope sorting
neg_slope = list()
pos_slope = list()

for biomarker in biomarker_columns:
    # Drop NaNs
    df_valid = df[['Age', biomarker]].dropna()

    if df_valid.shape[0] < 3:
        print(f"Skipping {biomarker} (insufficient data)")
        continue

    # --- Percentile Clipping (2%-98%) ---
    lower_percentile = df_valid[biomarker].quantile(0.02)
    upper_percentile = df_valid[biomarker].quantile(0.98)

    df_filtered = df_valid[
        (df_valid[biomarker] >= lower_percentile) &
        (df_valid[biomarker] <= upper_percentile)
    ]

    if df_filtered.shape[0] < 3:
        print(f"Skipping {biomarker} (insufficient data after clipping)")
        continue

    # Extract filtered data
    x_all = df_filtered['Age'].values
    y_all = df_filtered[biomarker].values
    one_x = df_filtered['Age'].values if df_filtered['AnimalID'] == 1
    one_y = df_filtered[biomarker].values if df_filtered['AnimalID'] == 1
    
    # Linear regression
    slope, intercept = np.polyfit(x_all, y_all, 1)
    x_fit = np.linspace(x_all.min(), x_all.max(), 100)
    y_fit_linear = slope * x_fit + intercept
    
    #Slope sorting
    if slope < 0:
      neg_slope.append(biomarker)
    elif slope > 0:
      pos_slope.append(biomarker)
    else:
      break
    
    # LOWESS smoothing
    lowess_result = sm.nonparametric.lowess(y_all, x_all, frac=0.3, return_sorted=True)
    x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 12))

    
    ax.scatter(x_all, y_all, alpha=0.5, s=10, label='Data (clipped 2%-98%)')
    ax.plot(x_fit, y_fit_linear, color='red', linestyle='--', linewidth=2, label=f'Linear Fit: y = {slope:.5f}x + {intercept:.5f}')
    ax.plot(x_lowess, y_lowess, color='green', linestyle='-', linewidth=2.5, label='LOWESS Smoother')
    ax.plot(one_x, one_y, alpha = 0.9, linewidth=1.5, color='magenta')
    
    ax.set_xlabel('Age')
    ax.set_ylabel(biomarker)
    ax.set_title(f'{biomarker} over Age (Clipped Extremes, Linear & LOWESS)')
    ax.legend()

    plt.tight_layout()
    plt.show()
    # === Save Plot ===
    #filename = os.path.join(output_dir, f"{biomarker}_over_age.png")
    #plt.savefig(filename, dpi=300)
    #print(f"Saved plot to {filename}")

    plt.close(fig)
print(neg_slope)
print(pos_slope)

import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

output_dir = os.path.expanduser('~/Downloads/dolphins-master/results/biomarker_plots(2-98%)')
os.makedirs(output_dir, exist_ok=True)

biomarker_columns = df.columns[8:52]

neg_slope = []
pos_slope = []

for biomarker in biomarker_columns:
    # Drop NaNs
    df_valid = df[['Age', 'AnimalID', biomarker]].dropna()

    if df_valid.shape[0] < 3:
        print(f"Skipping {biomarker} (insufficient data)")
        continue

    # --- Percentile Clipping (2%-98%) ---
    lower_percentile = df_valid[biomarker].quantile(0.02)
    upper_percentile = df_valid[biomarker].quantile(0.98)

    df_filtered = df_valid[
        (df_valid[biomarker] >= lower_percentile) &
        (df_valid[biomarker] <= upper_percentile)
    ]

    if df_filtered.shape[0] < 3:
        print(f"Skipping {biomarker} (insufficient data after clipping)")
        continue

    # Extract filtered data
    x_all = df_filtered['Age'].values
    y_all = df_filtered[biomarker].values

    # Filter for AnimalID == 1
    df_one = df_filtered[df_filtered['AnimalID'] == 8]
    one_x = df_one['Age'].values
    one_y = df_one[biomarker].values

    # Linear regression
    slope, intercept = np.polyfit(x_all, y_all, 1)
    x_fit = np.linspace(x_all.min(), x_all.max(), 100)
    y_fit_linear = slope * x_fit + intercept

    # Slope sorting
    if slope < 0:
        neg_slope.append(biomarker)
    elif slope > 0:
        pos_slope.append(biomarker)

    # LOWESS smoothing
    lowess_result = sm.nonparametric.lowess(y_all, x_all, frac=0.3, return_sorted=True)
    x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 12))

    ax.scatter(x_all, y_all, alpha=0.5, s=10, label='Data (clipped 2%-98%)')
    ax.plot(x_fit, y_fit_linear, color='red', linestyle='--', linewidth=2, label=f'Linear Fit: y = {slope:.5f}x + {intercept:.5f}')
    ax.plot(x_lowess, y_lowess, color='green', linestyle='-', linewidth=2.5, label='LOWESS Smoother')

    if len(one_x) > 0:
        ax.plot(one_x, one_y, alpha=0.7, linewidth=2.5, color='magenta', label='AnimalID = 8')

    ax.set_xlabel('Age')
    ax.set_ylabel(biomarker)
    ax.set_title(f'{biomarker} over Age (Clipped Extremes, Linear & LOWESS)')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Save plot if needed
    #filename = os.path.join(output_dir, f"{biomarker}_over_age.png")
    #plt.savefig(filename, dpi=300)
    #print(f"Saved plot to {filename}")

    plt.close(fig)
plt.figure(figsize = (12,12))
sns.heatmap(W, cmap = 'inferno', xticklabels = biomarker_columns, yticklabels = biomarker_columns)
plt.xlabel('To')
plt.ylabel('From')
plt.title('W Interaction Matrix')
plt.show()
plt.clf()
