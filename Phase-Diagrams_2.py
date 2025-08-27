import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
    Generate phase diagrams (Δvar_y vs Δvar_x) for each subject with time-based colormap.
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

            if dx is None or dy is None:
                continue

            # Align lengths
            min_len = min(len(dx), len(dy))
            dx = dx[:min_len]
            dy = dy[:min_len]

            # Build segments for line coloring
            points = np.array([dx, dy]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize time for color mapping
            norm = mcolors.Normalize(vmin=0, vmax=len(segments))
            cmap = cm.get_cmap('viridis')
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.arange(len(segments)))
            lc.set_linewidth(2)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlabel(f'Δ{var_x}')
            ax.set_ylabel(f'Δ{var_y}')
            ax.set_title(f'Phase Diagram: {var_y} vs {var_x} (ID: {subj_id})')
            ax.grid(False)

            # Add colorbar as time legend
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('Time Progression (relative)')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Phase diagrams saved to: {output_pdf}")




plot_phase_diagrams(df_valid, var_x='Iron', var_y='MCH', output_pdf='results/iron_mch_phase.pdf')











import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib import cm
import os

def smooth_and_diff_common_age(age, vals, frac=0.3, common_age_grid=None):
    valid = ~np.isnan(age) & ~np.isnan(vals)
    if valid.sum() < 3:
        return None, None, None

    age_clean = age[valid]
    vals_clean = vals[valid]

    smoothed = lowess(vals_clean, age_clean, frac=frac, return_sorted=True)
    age_smooth, val_smooth = smoothed[:, 0], smoothed[:, 1]

    if common_age_grid is not None:
        val_interp = np.interp(common_age_grid, age_smooth, val_smooth)
        dval = np.gradient(val_interp, common_age_grid)
        return common_age_grid, val_interp, dval
    else:
        dval = np.gradient(val_smooth, age_smooth)
        return age_smooth, val_smooth, dval

def plot_phase_diagrams(df, var_x, var_y, id_col='AnimalID', age_col='Age',
                        frac=0.3, output_pdf='phase_diagrams.pdf', num_interp_points=200):
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    ids = df[id_col].dropna().unique()
    ids.sort()

    global_age_min = df[age_col].min()
    global_age_max = df[age_col].max()
    global_age_grid = np.linspace(global_age_min, global_age_max, num_interp_points)

    cmap = cm.viridis
    norm = plt.Normalize(global_age_min, global_age_max)

    with PdfPages(output_pdf) as pdf:
        for subj_id in ids:
            df_sub = df[df[id_col] == subj_id].sort_values(age_col)
            age = df_sub[age_col].to_numpy()
            x_vals = df_sub[var_x].to_numpy()
            y_vals = df_sub[var_y].to_numpy()

            _, _, dx = smooth_and_diff_common_age(age, x_vals, frac, common_age_grid=global_age_grid)
            _, _, dy = smooth_and_diff_common_age(age, y_vals, frac, common_age_grid=global_age_grid)

            if dx is None or dy is None:
                continue

            # Drop any segments that include NaNs
            points = np.column_stack([dx, dy])
            valid = np.all(np.isfinite(points), axis=1)

            # Keep only consecutive valid segments
            points_valid = points[valid]
            age_valid = global_age_grid[valid]

            if len(points_valid) < 2:
                continue  # not enough to plot

            # Form segments for LineCollection (no wraparound)
            # Append a NaN row to break the final connection
            points_valid = np.vstack([points_valid, [np.nan, np.nan]])
            age_valid = np.append(age_valid, np.nan)

            # Build segments, skipping any that include NaNs
            segments = []
            segment_ages = []
            for i in range(len(points_valid) - 1):
                seg = np.vstack([points_valid[i], points_valid[i + 1]])
                if np.isnan(seg).any():
                    continue  # skip segments with NaN
                segments.append(seg)
                segment_ages.append(age_valid[i])
            segments = np.array(segments)
            segment_ages = np.array(segment_ages)

            segment_ages = age_valid[:-1]  # for coloring

            fig, ax = plt.subplots(figsize=(8, 6))
            lc = LineCollection(segments, cmap=cmap, norm=norm,
                                array=segment_ages, linewidth=2.5)
            ax.add_collection(lc)

            ax.set_xlim(np.nanmin(dx) - 0.1, np.nanmax(dx) + 0.1)
            ax.set_ylim(np.nanmin(dy) - 0.1, np.nanmax(dy) + 0.1)
            ax.set_xlabel(f'd{var_x}/dt')
            ax.set_ylabel(f'd{var_y}/dt')
            ax.set_title(f'Phase Diagram: {var_y} vs {var_x} (ID: {subj_id})')
            ax.grid(False)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Age (yrs)')

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Phase diagrams saved to: {output_pdf}")
