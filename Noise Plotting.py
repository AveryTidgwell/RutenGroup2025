y_cur = data['y_cur']          # y_t
y_next = data['y_next']        # y_{t+1}
delta_t = data['delta_t']      # Δt
mu = mu                        # assumed to be aligned with y_cur
W = W                          # 43 x 43 interaction matrix

# Ensure y_cur and mu have same column indices
y_cur.columns = list(range(len(y_cur.columns)))
mu.columns = list(range(len(mu.columns)))

# Compute (y_t - mu)
y_mu = y_cur - mu

# Compute the deterministic update
deterministic = delta_t.values[:, None] * y_mu.dot(W.T)

# Compute noise
epsilon = y_next.values - y_cur.values - deterministic


# Compute RMSE per biomarker
rmse_per_biomarker = np.sqrt((epsilon ** 2).mean(axis=0))
rmse_per_biomarker_df = pd.DataFrame({
    'Biomarker': biomarker_columns,
    'RMSE': rmse_per_biomarker
})

# Add residuals to df_valid for grouping
residuals_df = epsilon.copy()
residuals_df.columns = biomarker_columns
residuals_df['Sex'] = df_valid['Sex'].values

# Group by Sex and calculate mean residuals per biomarker
grouped = residuals_df.groupby('Sex')
mean_residuals_m = grouped.get_group(1)[biomarker_columns].mean()
mean_residuals_f = grouped.get_group(0)[biomarker_columns].mean()

# Combine into one DataFrame
grouped_mean_residuals = pd.DataFrame({
    'Biomarker': biomarker_columns,
    'Mean Residual (M)': mean_residuals_m.values,
    'Mean Residual (F)': mean_residuals_f.values
})

# Compute difference between male and female
grouped_mean_residuals['Sex_Difference'] = (
    grouped_mean_residuals['Mean Residual (M)'] - grouped_mean_residuals['Mean Residual (F)']
)

# Sort the DataFrame by RMSE (ascending)
rmse_sorted = rmse_per_biomarker_df.sort_values('RMSE', ascending=True)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
sns.barplot(data=rmse_sorted, x='Biomarker', y='RMSE', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('RMSE per Biomarker')
plt.tight_layout()
plt.show()
plt.clf()


# Usage example:
plot_sex_difference_barplot(grouped_mean_residuals)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import TwoSlopeNorm

def plot_sex_difference_barplot(df):
    # Sort by difference for clearer visualization
    df_sorted = df.sort_values('Sex_Difference', ascending=True)

    # Set diverging color map: pink for negative, blue for positive
    cmap = sns.diverging_palette(330, 210, s=90, l=60, as_cmap=True)

    # Normalize around zero to map colors properly
    norm = TwoSlopeNorm(vmin=df_sorted['Sex_Difference'].min(), vcenter=0, vmax=df_sorted['Sex_Difference'].max())

    # Map values to colors
    colors = cmap(norm(df_sorted['Sex_Difference'].values))

    # Plot
    plt.figure(figsize=(8, 8))
    bars = plt.barh(df_sorted['Biomarker'], df_sorted['Sex_Difference'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

    plt.xlabel('Mean Noise Difference (M - F)')
    plt.title('Sex Differences in Noise by Biomarker')

    # Add Female > Male and Male > Female labels
    xlim = plt.xlim()
    ylim = plt.ylim()
    offset = (xlim[1] - xlim[0]) * 0.01  # small offset from edge

    plt.text(xlim[0] + offset, ylim[0] - 1.5, 'Female > Male', ha='left', va='top', fontsize=10, color='#C71585')  # pink
    plt.text(xlim[1] - offset, ylim[0] - 1.5, 'Female < Male', ha='right', va='top', fontsize=10, color='#0000CD')  # blue

    plt.tight_layout()
    plt.show()
    plt.clf()














import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Loop through each biomarker column (0 to 43)
for biomarker in epsilon.columns:
    # Skip the 'Age' column if it has already been added
    if biomarker == 'Age':
        continue

    # Create DataFrame with Age and residual for this biomarker
    df_plot = pd.DataFrame({
        'Age': df_valid['Age'].values,
        'residual': epsilon[biomarker].values
    })

    # Compute absolute residual as noise
    df_plot['abs_residual'] = df_plot['residual'].abs()

    # Plot LOWESS line
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=df_plot,
        x='Age',
        y='abs_residual',
        lowess=True,
        scatter_kws={'s': 10, 'alpha': 0.3},
        line_kws={'color': 'darkblue'}
    )
    
    name = biomarker_columns[biomarker]
    plt.title(f'{name} Noise Over Age')
    plt.xlabel('Age')
    plt.ylabel('Absolute Residual (Noise)')
    plt.tight_layout()
    plt.show()
    plt.clf()






import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

for biomarker in epsilon.columns:
    # Skip the 'Age' column if it has already been added
    if biomarker == 'Age':
        continue

    # Create DataFrame with Age and residual for this biomarker
    df_plot = pd.DataFrame({
        'Age': df_valid['Age'].values,
        'residual': epsilon[biomarker].values
    })

    # Compute absolute residual as noise
    df_plot['abs_residual'] = df_plot['residual'].abs()

    # Fit linear model
    X = df_plot[['Age']].values
    y = df_plot['abs_residual'].values
    model = LinearRegression().fit(X, y)
    m = model.coef_[0]
    b = model.intercept_

    # Format regression equation
    equation = f"y = {m:.5f}x + {b:.5f}"

    # Plot LOWESS line
    plt.figure(figsize=(6, 4))
    
    sns.scatterplot(
        data=df_plot,
        x='Age',
        y='abs_residual',
        color='grey',
        alpha=0.3
    )
    

    # Plot linear fit
    sns.regplot(
        data=df_plot,
        x='Age',
        y='abs_residual',
        lowess=False,
        scatter=False,
        line_kws={'color': 'red', 'linestyle': '--', 'label': 'Linear Fit'}
    )

    # Run LOWESS with custom frac
    smoothed = lowess(
        endog=df_plot['abs_residual'],
        exog=df_plot['Age'],
        frac=0.3  # sensitivity: lower = more sensitive to noise, higher = smoother
    )

    # Plot manually
    plt.plot(smoothed[:, 0], smoothed[:, 1], color='blue', label='Smooth Fit')

    # Add regression equation text
    plt.text(
        0.05, 0.95, equation,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5)
    )

    # Set plot title and labels
    name = biomarker_columns[biomarker]
    plt.title(f'{name} Noise Over Age')
    plt.xlabel('Age (yrs)')
    plt.ylabel('Absolute Residual (Noise)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    # === Save Plot ===
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'Noise_Over_Age')
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{name}_noise_over_age.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved plot to {filename}")

