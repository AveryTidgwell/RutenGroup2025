import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster, maxdists
from scipy.spatial.distance import squareform
from collections import defaultdict

def hierarchical_block_clustering(W, biomarker_columns, title='Hierarchical Clustering of W', num_clusters=4, draw_boxes = True, save_path=None):
    # Symmetrize W
    W_sym = (W + W.T) / 2
    W_sym = W_sym / np.abs(W_sym).max()

    # Compute pairwise distance (1 - correlation)
    distance_matrix = 1 - np.corrcoef(W_sym)
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    # Calculate color threshold to match num_clusters
    dists = linkage_matrix[:, 2]
    dists_sorted = np.sort(dists)
    if num_clusters - 1 < len(dists_sorted):
        color_threshold = dists_sorted[-num_clusters + 1]
    else:
        color_threshold = 0.0
    ordered_indices = leaves_list(linkage_matrix)
    W_ordered = W_sym[ordered_indices][:, ordered_indices]
    biomarker_labels = [biomarker_columns[i] for i in ordered_indices]

    # Get cluster assignments and color mapping
    cluster_ids = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    cluster_ids_ordered = cluster_ids[ordered_indices]

    # Mask diagonal for color scaling, but restore it with dark blue afterward
    #mask = np.eye(W_ordered.shape[0], dtype=bool)
    #off_diag_vals = W_ordered[~mask]
    #max_offdiag = np.max(np.abs(off_diag_vals))

    max_offdiag = 1


    # Prepare colormap and restore diagonal with a strong color (e.g., dark blue)
    W_visual = W_ordered.copy()
    #np.fill_diagonal(W_visual, -max_offdiag)  # Dark blue by using the min value in 'bwr'
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 14))  # Match height and width
    grid = GridSpec(1, 2, width_ratios=[20, 2], wspace=0.02)
    
    # Heatmap (left)
    ax_heatmap = fig.add_subplot(grid[0])
    sns.heatmap(
        W_visual,
        cmap='bwr',
        center=0,
        vmin=-max_offdiag,
        vmax=max_offdiag,
        xticklabels=biomarker_labels,
        yticklabels=biomarker_labels,
        mask=False,
        cbar=False,
        ax=ax_heatmap
    )
    ax_heatmap.set_title(title, fontsize=14, pad=20)
    ax_heatmap.tick_params(labelsize=7)
    plt.setp(ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)
    ax_heatmap.set_aspect('equal')
    
    # Manually create a smaller dendrogram axis
    ax_dendro = fig.add_axes([0.82, 0.26, 0.12, 0.47]) 

    
    # Dendrogram with color_threshold
    dendro = dendrogram(
        linkage_matrix,
        orientation='right',
        labels=[biomarker_columns[i] for i in ordered_indices],
        ax=ax_dendro,
        no_labels=True,
        color_threshold=color_threshold
    )

    
    # Invert dendrogram to put leaves at top
    ax_dendro.invert_yaxis()

    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.set_title('Dendrogram', fontsize=12)



    
    # Group contiguous leaves by dendrogram branch color
    leaf_order = dendro['leaves']
    leaf_colors = dendro['leaves_color_list']

    cluster_blocks = []
    current_color = leaf_colors[0]
    start_idx = 0

    for i in range(1, len(leaf_colors)):
        if leaf_colors[i] != current_color:
            cluster_blocks.append((start_idx, i - 1, current_color))
            start_idx = i
            current_color = leaf_colors[i]
    cluster_blocks.append((start_idx, len(leaf_colors) - 1, current_color))  # Add last block
    if draw_boxes:
        # Draw rectangles around each contiguous block with the same color
        for start, end, color in cluster_blocks:
            if end - start < 1:
                continue  # Skip singleton points
            ax_heatmap.add_patch(plt.Rectangle(
                (start, start),
                end - start + 1,
                end - start + 1,
                fill=False,
                edgecolor=color,
                linewidth=2
            ))
    
    ax_cbar = fig.add_axes([0.85, 0.8, 0.015, 0.15])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=-max_offdiag, vmax=max_offdiag))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax_cbar)
    cbar.set_label('Interaction Strength', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustered heatmap saved to {save_path}")
        plt.show()
    else:
        plt.show()
    plt.clf()
    # Apply the ordered biomarker column order to a different dataframe (e.g., z_mu_df or z_df)
    ordered_biomarkers = [biomarker_columns[i] for i in ordered_indices]
    reordered_weights = biomarker_weights[ordered_biomarkers]
    reordered_weights = reordered_weights.T
    return ordered_indices, W_ordered, reordered_weights



ordered_indices, W_ordered, reordered_weights = hierarchical_block_clustering(
    W, biomarker_columns,
    title='Clustered Interaction Matrix',
    num_clusters=4,
    draw_boxes = False
)

print(W.min())

