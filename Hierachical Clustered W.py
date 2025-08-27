import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

def hierarchical_block_clustering(W, biomarker_columns, title='Hierarchical Clustering of W', save_path=None):
    # Symmetrize W
    W_sym = (W + W.T) / 2

    # Compute pairwise distance (1 - correlation)
    distance_matrix = 1 - np.corrcoef(W_sym)
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    ordered_indices = leaves_list(linkage_matrix)
    W_ordered = W_sym[ordered_indices][:, ordered_indices]

    # Reorder biomarker names
    biomarker_labels = [biomarker_columns[i] for i in ordered_indices]

    # Mask diagonal
    mask = np.eye(W_ordered.shape[0], dtype=bool)
    off_diag_vals = W_ordered[~mask]
    max_offdiag = np.max(np.abs(off_diag_vals))

    # Set up the figure with axes for dendrogram and heatmap
    fig = plt.figure(figsize=(14, 12))
    grid = plt.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 20], hspace=0.05, wspace=0.05)
    

    # Dendrogram (top)
    ax_dendro = fig.add_subplot(grid[0, 0])
    dendro = dendrogram(
        linkage_matrix,
        orientation='right',
        no_labels=True,
        ax=ax_dendro,
        color_threshold=None,
    )
    ax_dendro.axis('off')

    # Heatmap (bottom left)
    ax_heatmap = fig.add_subplot(grid[1, 0])
    ax = sns.heatmap(
        W_ordered,
        mask=mask,
        cmap='bwr',
        center=0,
        vmin=-max_offdiag,
        vmax=max_offdiag,
        xticklabels=biomarker_labels,
        yticklabels=biomarker_labels,
        ax=ax_heatmap,
        cbar=False  # We'll add a custom colorbar
    )
    ax_heatmap.set_title(title)
    ax.set_aspect('equal')
    ax_heatmap.tick_params(labelsize=8)
    plt.setp(ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)

    # Custom colorbar in top-right corner
    ax_cbar = fig.add_axes([0.91, 0.7, 0.015, 0.12])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-max_offdiag, vmax=max_offdiag))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Interaction Strength', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustered heatmap saved to {save_path}")
    else:
        plt.show()

    return ordered_indices, W_ordered


ordered_indices, W_ordered = hierarchical_block_clustering(W, biomarker_columns, title='Clustered Interaction Matrix')
