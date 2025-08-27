import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_bio(df, biomarker_columns = biomarker_columns, pdf_path='dog_biomarker_plots.pdf'):
    with PdfPages(pdf_path) as pdf:
        for biomarker in biomarker_columns:
            plt.figure(figsize=(10, 6))

            # Scatter all points in blue
            plt.scatter(df['Age'], df[biomarker], color='blue', label='Values')

            # Highlight the row(s) with the max value
            max_val = df[biomarker].max()
            max_points = df[df[biomarker] == max_val]
            plt.scatter(max_points['Age'], max_points[biomarker], color='red', label='Max Value', zorder=5)

            plt.title(f'{biomarker} vs Age')
            plt.xlabel('Age')
            plt.ylabel(biomarker)
            plt.legend()
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()

    print(f"Saved all biomarker plots to {pdf_path}")

plot_bio(df, biomarker_columns, pdf_path = 'dog_biomarker_plots.pdf')




from scipy.stats import norm
import pandas as pd

from scipy.stats import norm
import pandas as pd

def get_outlier_pvals_with_top5(df, biomarker_columns, id_col='dog_id', alpha=0.001):
    outlier_summary = {}

    for biomarker in biomarker_columns:
        values = df[biomarker].dropna()
        mean = values.mean()
        std = values.std()

        if std == 0:
            continue

        # Compute z-scores and p-values
        z_scores = (df[biomarker] - mean) / std
        p_values = 2 * norm.sf(abs(z_scores))

        df[f'{biomarker}_z'] = z_scores
        df[f'{biomarker}_pval'] = p_values

        # Find the 5 farthest points (largest |z|)
        top5 = df.loc[z_scores.abs().nlargest(5).index, [id_col, biomarker, f'{biomarker}_z', f'{biomarker}_pval']]
        print(f"\nTop 5 outliers for biomarker: {biomarker}")
        print(top5.to_string(index=False))

        # Identify true statistical outliers
        n_outliers = (p_values < alpha).sum()
        outlier_summary[biomarker] = {
            'num_outliers': n_outliers,
            'percent_outliers': 100 * n_outliers / df.shape[0]
        }

    return df, outlier_summary




df_with_pvals, outlier_stats = get_outlier_pvals_with_top5(df, biomarker_columns, id_col = 'dog_id', alpha=0.001)

# View outlier counts per biomarker
for biomarker, stats in outlier_stats.items():
    print(f"{biomarker}: {stats['num_outliers']} outliers ({stats['percent_outliers']:.2f}%)")
