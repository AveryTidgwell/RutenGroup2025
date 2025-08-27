from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import os
import matplotlib.pyplot as plt

def plot_drift_group(
    z_df,
    z_var_list,
    output_pdf='drift_group.pdf',
    linear_title='Linear Drift Plot',
    lowess_title='LOWESS Drift Plot'
):
    colors = sns.color_palette("Set1", len(z_var_list))  # use distinct colors
    t = z_df['Age']

    # Set up saving location
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'Drift_Groups')
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, output_pdf)

    with PdfPages(pdf_path) as pdf:
        # -------- Plot 1: Linear Fits --------
        plt.figure(figsize=(10, 8))
        for i, z_col in enumerate(z_var_list):
            x_vals = t
            y_vals = z_df[z_col].to_numpy()

            x_pred, mean, lower, upper = bootstrap_regression(x_vals, y_vals)
            plt.plot(x_pred, mean, color=colors[i], label=z_col)
            plt.fill_between(x_pred, lower, upper, color=colors[i], alpha=0.2)

        plt.ylim(-5, 5)
        plt.xlabel('Age (yrs)')
        plt.ylabel('Natural Variable (z)')
        plt.title(linear_title)
        plt.legend(title="Natural Variable")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # -------- Plot 2: LOWESS Fits --------
        plt.figure(figsize=(10, 8))
        for i, z_col in enumerate(z_var_list):
            x_vals = t
            y_vals = z_df[z_col].to_numpy()

            x_eval, mean_lowess, lower_ci, upper_ci = bootstrap_lowess_ci(x_vals, y_vals)
            plt.plot(x_eval, mean_lowess, color=colors[i], label=z_col)
            plt.fill_between(x_eval, lower_ci, upper_ci, color=colors[i], alpha=0.2)

        plt.ylim(-5, 5)
        plt.xlabel('Age (yrs)')
        plt.ylabel('Natural Variable (z)')
        plt.title(lowess_title)
        plt.legend(title="Natural Variable")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"Drift plots saved to {pdf_path}")


# Minimal drift group
plot_drift_group(
    z_df,
    ['z_43', 'z_35', 'z_23', 'z_22', 'z_38'],
    output_pdf='minimal_drift.pdf',
    linear_title='Minimal Drift - Linear Fit',
    lowess_title='Minimal Drift - LOWESS Fit'
)

# Moderate drift group
plot_drift_group(
    z_df,
    ['z_30', 'z_26', 'z_10', 'z_1', 'z_5'],
    output_pdf='moderate_drift.pdf',
    linear_title='Moderate Drift - Linear Fit',
    lowess_title='Moderate Drift - LOWESS Fit'
)

# High drift group
plot_drift_group(
    z_df,
    ['z_4', 'z_12', 'z_13', 'z_15', 'z_17'],
    output_pdf='high_drift.pdf',
    linear_title='High Drift - Linear Fit',
    lowess_title='High Drift - LOWESS Fit'
)

from PyPDF2 import PdfMerger
import os

# Define your filenames (adjust if needed)
pdf_files = [
    'minimal_drift.pdf',
    'moderate_drift.pdf',
    'high_drift.pdf'
]

# Path to your folder
base_dir = os.path.join(os.getcwd(), 'Downloads', 'dolphins-master', 'results', 'Drift_Groups')

# Full paths
full_paths = [os.path.join(base_dir, fname) for fname in pdf_files]

# Output file
merged_pdf_path = os.path.join(base_dir, 'combined_drift_plots.pdf')

# Merge PDFs
merger = PdfMerger()
for path in full_paths:
    merger.append(path)

merger.write(merged_pdf_path)
merger.close()

print(f"Merged PDF saved to: {merged_pdf_path}")

