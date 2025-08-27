import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
import os


def smooth_and_diff(age, biomarker_vals, frac=0.3):
    """
    LOWESS smoothing and numerical derivative estimation
    """
    valid = ~np.isnan(age) & ~np.isnan(biomarker_vals)
    if valid.sum() < 3:
        return None, None, None

    age_clean = age[valid]
    biomarker_clean = biomarker_vals[valid]

    smoothed = lowess(biomarker_clean, age_clean, frac=frac, return_sorted=True)
    x_smooth, y_smooth = smoothed[:, 0], smoothed[:, 1]
    dy_dx = np.gradient(y_smooth, x_smooth)
    return x_smooth, y_smooth, dy_dx


def plot_phase_diagrams(df, var_x, var_y, id_col='AnimalID', age_col='Age',
                        frac=0.3, output_pdf='phase_diagrams.pdf'):
    """
    Generate phase diagrams (Δvar_y vs Δvar_x) for each subject and save to multipage PDF.
    """
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    ids = df[id_col].dropna().unique()
    ids.sort()

    with PdfPages(output_pdf) as pdf:
        for subj_id in ids:
            df_sub = df[df[id_col] == subj_id].sort_values(age_col)
            age = df_sub[age_col].to_numpy()
            x_vals = df_sub[var_x].to_numpy()
            y_vals = df_sub[var_y].to_numpy()

            # Smooth and differentiate
            age_x, _, dx = smooth_and_diff(age, x_vals, frac=frac)
            age_y, _, dy = smooth_and_diff(age, y_vals, frac=frac)

            # Check we have matching valid arrays
            if dx is None or dy is None:
                continue

            # Align if needed (assuming age_x ≈ age_y)
            min_len = min(len(dx), len(dy))
            dx = dx[:min_len]
            dy = dy[:min_len]

            # Plot phase diagram
            plt.figure(figsize=(8, 6))
            plt.plot(dx, dy, '-o', color='tab:blue', alpha=0.7, label='Trajectory')

            # Optionally add arrows
            for i in range(1, len(dx)):
                plt.arrow(dx[i-1], dy[i-1], dx[i] - dx[i-1], dy[i] - dy[i-1],
                          shape='full', lw=0, length_includes_head=True,
                          head_width=0.01, head_length=0.01, alpha=0.4, color='gray')

            plt.xlabel(f'Δ{var_x}')
            plt.ylabel(f'Δ{var_y}')
            plt.title(f'Phase Diagram: {var_y} vs {var_x} (ID: {subj_id})')
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Phase diagrams saved to: {output_pdf}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import UnivariateSpline
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def smooth_and_diff(df, variable, age_col='Age'):
    df = df.sort_values(age_col)
    x = df[age_col].values
    y = df[variable].values

    valid = ~np.isnan(x) & ~np.isnan(y)
    if valid.sum() < 4:
        return None, None

    x = x[valid]
    y = y[valid]

    spline = UnivariateSpline(x, y, s=1)
    dy = spline.derivative()(x)
    return x, dy


def plot_phase_diagrams(df, var_x, var_y, id_col='AnimalID', age_col='Age', output_pdf='phase_diagrams.pdf'):
    pdf_pages = PdfPages(output_pdf)
    ids = df[id_col].unique()

    for subj_id in ids:
        df_sub = df[df[id_col] == subj_id]

        x_time, dx = smooth_and_diff(df_sub, var_x, age_col)
        y_time, dy = smooth_and_diff(df_sub, var_y, age_col)

        if dx is None or dy is None:
            continue

        # Match time and derivatives
        common_len = min(len(dx), len(dy), len(x_time), len(y_time))
        dx = dx[:common_len]
        dy = dy[:common_len]
        t = x_time[:common_len]

        if len(t) < 2:
            continue

        # Normalize time for colormap
        norm = mcolors.Normalize(vmin=t.min(), vmax=t.max())
        cmap = cm.get_cmap('viridis')

        fig, ax = plt.subplots(figsize=(6, 6))

        for i in range(len(t) - 1):
            ax.plot(
                [dx[i], dx[i + 1]], [dy[i], dy[i + 1]],
                color=cmap(norm(t[i])),
                linewidth=2
            )

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Age')

        ax.set_title(f'Phase Diagram: {var_y} vs {var_x} | ID: {subj_id}')
        ax.set_xlabel(f'd({var_x})/dt')
        ax.set_ylabel(f'd({var_y})/dt')
        ax.grid(False)
        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()

    pdf_pages.close()
    print(f"Saved all phase diagrams to {output_pdf}")















# Example usage
plot_phase_diagrams(
    df=df_valid,  # your cleaned dolphin dataframe
    var_x='RBC',
    var_y='SED60',
    output_pdf='results/phase_diagrams_RBC_SED60.pdf'
)
