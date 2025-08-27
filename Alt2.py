def hierarchical_block_clustering(W, biomarker_columns, title='Hierarchical Clustering of W', num_clusters=4, save_path=None):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    
    # Symmetrize
    W_sym = (W + W.T) / 2

    # Distance: 1 - correlation matrix
    distance_matrix = 1 - np.corrcoef(W_sym)
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward')
    ordered_indices = leaves_list(linkage_matrix)
    W_ordered = W_sym[ordered_indices][:, ordered_indices]
    biomarker_labels = [biomarker_columns[i] for i in ordered_indices]

    # Compute distance threshold for desired clusters
    last_distances = linkage_matrix[-(num_clusters - 1):, 2]
    threshold = np.min(last_distances)

    # Cluster assignment using distance threshold
    cluster_ids = fcluster(linkage_matrix, t=threshold, criterion='distance')
    cluster_ids_ordered = cluster_ids[ordered_indices]

    # Setup heatmap values
    mask = np.eye(W_ordered.shape[0], dtype=bool)
    off_diag_vals = W_ordered[~mask]
    max_offdiag = np.max(np.abs(off_diag_vals))
    W_visual = W_ordered.copy()
    np.fill_diagonal(W_visual, -max_offdiag)  # make diagonal distinct color on heatmap

    # Plotting setup
    fig = plt.figure(figsize=(14, 12))
    grid = plt.GridSpec(1, 2, width_ratios=[20, 1.5], wspace=0.02)

    # Heatmap axis
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
    ax_heatmap.tick_params(labelsize=8)
    plt.setp(ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)

    # Dendrogram axis
    ax_dendro = fig.add_subplot(grid[1])

    # Dendrogram with color_threshold for coloring branches
    dendro = dendrogram(
        linkage_matrix,
        orientation='right',
        labels=biomarker_labels,
        color_threshold=threshold,
        above_threshold_color='grey',
        ax=ax_dendro,
        no_labels=False
    )
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    # Map dendrogram leaf label -> color directly
    leaf_color_map = dict(zip(dendro['ivl'], dendro['color_list']))


    # For heatmap cluster boxes: group indices by leaf_color_map colors
    unique_colors = list(dict.fromkeys(leaf_color_map.values()))  # preserve order
    print("biomarker_labels:", biomarker_labels)
    print("leaf_color_map keys:", leaf_color_map.keys())
    print("dendrogram labels:", dendro['ivl'])

    for ccolor in unique_colors:
        indices = [i for i, label in enumerate(biomarker_labels) if label in leaf_color_map and leaf_color_map[label] == ccolor]
        if len(indices) < 2:
            continue
        i_min, i_max = min(indices), max(indices)
        ax_heatmap.add_patch(plt.Rectangle(
            (i_min, i_min),
            i_max - i_min + 1,
            i_max - i_min + 1,
            fill=False,
            edgecolor=ccolor,
            linewidth=2
        ))

    # Color dendrogram leaf labels to match branch colors
    for txt in ax_dendro.texts:
        label_text = txt.get_text()
        if label_text in leaf_color_map:
            txt.set_color(leaf_color_map[label_text])
            txt.set_fontsize(8)

    ax_dendro.set_title('Dendrogram', fontsize=12)

    # Colorbar
    ax_cbar = fig.add_axes([0.91, 0.7, 0.015, 0.15])
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

    return ordered_indices, W_ordered



ordered_indices, W_ordered = hierarchical_block_clustering(
    W, biomarker_columns,
    title='Clustered Interaction Matrix',
    num_clusters=7
)


