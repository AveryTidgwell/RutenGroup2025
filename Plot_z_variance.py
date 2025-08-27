import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_z_variance(z_df):
    for z_col in z_df.columns[0:10]:
        plt.figure(figsize=(8, 6))

        # Optional scatter plot of raw data
        #plt.scatter(z_df['Age'], z_df[z_col], alpha=0.2, c='red', label='Raw data')

        # Bin ages to nearest 0.5 years for smoother trend
        z_df['binned_age'] = (z_df['Age'] * 2).round() / 2

        # Use seaborn to plot line + 95% CI directly from raw data
        sns.lineplot(
            data=z_df,
            x='binned_age',
            y=z_col,
            ci=95,
            estimator='mean',
            color='blue',
            label='Mean ± 95% CI'
        )
        
        #Plot mu
        t = z_df['Age']
        col_idx = z_df.columns.get_loc(z_col)  # get position of z_col
        mu_vals = z_mu_df.iloc[:, col_idx]     # access same-position column in z_mu_df

        #Get linear regression of mu 
        slope, intercept = np.polyfit(t, mu_vals, 1)
        x_fit = np.linspace(t.min(), t.max(), 100)
        y_fit_linear = slope * x_fit + intercept    
       
        plt.plot(x_fit, y_fit_linear, linestyle='-', linewidth = 3, c = 'black',alpha=1, label = f'mu(t) = {slope:.5f}*t + {intercept:.5f}')
       
        # Labels and formatting
        plt.xlabel('Age (yrs)')
        plt.ylabel(f'Natural Variable: {z_col}')
        plt.title(f'{z_col} Over Time (Smoothed)')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'results')
        os.makedirs(save_dir, exist_ok=True)

        plotname = f'{z_col} variance.png'
        save_path = os.path.join(save_dir, plotname)
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

        plt.show()
        plt.clf()


plot_z_variance(z_df)

