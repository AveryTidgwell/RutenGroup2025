import os
import matplotlib.pyplot as plt

def get_z_variables(W, mu, df, biomarker_columns, plotname = None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    print(np.real(eigenvalues))
    P_inv = np.linalg.inv(eigenvectors)
    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T

    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(len(biomarker_columns))]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(len(biomarker_columns))]
    lambda_names = [f'lambda_{i+1}' for i in range(len(biomarker_columns))]
    
    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_mu_df  = pd.DataFrame(z_mu.real, columns=natural_mu_names)


    z_df = pd.concat([z_bio_df,z_mu_df],axis = 1)
    z_df[['AnimalID','Sex','Species','Age']] = df[['AnimalID','Sex','Species','Age']].copy()

    if plotname is not None:
        # Sort eigenvalues and corresponding biomarkers
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        sorted_biomarkers = [lambda_names[i] for i in np.argsort(eigenvalues)[::-1]]

        # Plot the sorted eigenvalues with biomarker names on the x-axis
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_biomarkers, y=sorted_eigenvalues, palette="viridis")
        plt.xlabel("Name")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W")
        plt.xticks(rotation=90)  # Rotate x labels if needed for better readability
        plt.tight_layout()
        plt.show()
        #plt.savefig(os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300, bbox_inches='tight')

    return z_df, z_mu_df

z_df, z_mu_df = get_z_variables(W, mu, data['df_valid'], biomarker_columns, plotname = 'get_z_variables')
z_df = imputation(z_df, imputation_type = 'mean')

def plot_all_z(z_df, z_mu_df):
    # Identify natural variable columns
    z_cols = [col for col in z_df.columns if col.startswith("z_") and not col.startswith("mu_")]
    for z_col in z_cols:
        
        plt.figure(figsize=(10,8))
        for dolphin_id in z_df['AnimalID'].unique():
            dolphin_data = z_df.sort_values(['AnimalID', 'Age'])[z_df['AnimalID'] == dolphin_id]
            if dolphin_data.empty:
                continue
            # Plot actual values
            t = dolphin_data['Age']
            z_vals = dolphin_data[z_col]
            plt.scatter(t, z_vals, linestyle='-', alpha=0.8)
        
        t = z_df['Age']
        col_idx = z_df.columns.get_loc(z_col)  # get position of z_col
        mu_vals = z_mu_df.iloc[:, col_idx]     # access same-position column in z_mu_df

        #Get linear regression of mu 
        slope, intercept = np.polyfit(t, mu_vals, 1)
        x_fit = np.linspace(t.min(), t.max(), 100)
        y_fit_linear = slope * x_fit + intercept    
        plt.plot(x_fit, y_fit_linear, linestyle='-', linewidth = 3, c = 'black',alpha=1, label = f'mu(t) = {slope:.5f}*t + {intercept:.5f}')
       
        eigvals, _ = np.linalg.eig(W)
        eigenvalue = np.real(eigvals[col_idx])
        recovery_time = -1 / eigenvalue  # time between markers

        # Generate marker positions along the x-axis (time)
        t_start, t_end = t.min(), t.max()
        marker_positions = np.arange(t_start, t_end, recovery_time)

        # Plot horizontal line with markers at the computed positions
        y_level = z_df[z_col].min() - 1

        plt.plot(
            marker_positions,                             # x positions of markers
            [y_level] * len(marker_positions),            # constant y values
            linestyle='None',
            marker='x',
            color='black',
            label=f'Auto-Correlation Time: {recovery_time:.1f} yrs'
        )
        plt.grid()
        plt.xlabel('Age (yrs)')
        plt.ylabel(f'Natural Variable: {z_col}')
        plt.title(f'{z_col} for all dolphins')
        plt.tight_layout()

        #Optional: turn off legend for cleaner plot
        plt.legend()

        #Save the plot
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'z_natural_variables')
        os.makedirs(save_dir, exist_ok=True)

        plotname = f'{z_col} for all dolphins.png'
        save_path = os.path.join(save_dir, plotname)
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()
        plt.clf()

plot_all_z(z_df, z_mu_df)


def plot_zdev_lambda(z_df, z_mu_df, W, tol=1e-5):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # Identify z_ columns
    z_cols = [col for col in z_df.columns if col.startswith("z_") and not col.startswith("mu_")]
    eigvals, _ = np.linalg.eig(W)

    # Step 1: Collect all (x, y, label) points
    points = []

    for z_col in z_cols:
        col_idx = z_df.columns.get_loc(z_col)
        z_vals = z_df[z_col]
        mu_vals = z_mu_df.iloc[:, col_idx]

        mean_deviation = (z_vals - mu_vals).mean()
        stdev = z_vals.std()
        y = mean_deviation / stdev
        x = -np.real(eigvals[col_idx])

        points.append((x, y, z_col))

    # Step 2: Group nearby points by rounded coordinates
    grouped_points = defaultdict(list)

    for x, y, label in points:
        # Round coordinates to specified tolerance
        key = (round(x / tol) * tol, round(y / tol) * tol)
        grouped_points[key].append(label)
       
    # Step 3: Plot each unique point with combined label
    plt.figure(figsize=(8, 8))

    for (x, y), labels in grouped_points.items():
        label_text = ' + '.join(labels)
        plt.scatter(x, y, color='blue', marker='+')
        plt.text(x, y, label_text, fontsize=9, ha='right', va='bottom')
        
    # Decorate plot
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Recovery Rate (−λ)')
    plt.ylabel(r'$\langle z_n - \mu_n \rangle / \mathrm{SD}(z_n)$')
    plt.title('Natural Variable Deviation vs. Recovery Rate')
    plt.tight_layout()

    # Save
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', 'z_natural_variables')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'z_deviation_vs_lambda_grouped.png')
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.show()
    plt.clf()

plot_zdev_lambda(z_df, z_mu_df, W, 1e-5)


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_biomarker_circle(W, biomarker_names, top_n = False):
    G = nx.DiGraph()
    n = len(biomarker_names)

    # Step 1: Extract all edges and weights
    all_edges = []
    for i in range(n):
        for j in range(n):
            #if i != j:  # exclude self-loops
            all_edges.append((biomarker_names[i], biomarker_names[j], W[i, j]))

    # Step 2: Sort by absolute weight and keep top_n
    if top_n != False:
        top_edges = sorted(all_edges, key=lambda x: abs(x[2]), reverse=True)[:top_n]
        title = top_n
    else:
        top_edges = all_edges
        title = ''
    # Step 3: Build graph with top edges
    G.add_nodes_from(biomarker_names)

    for src, tgt, weight in top_edges:
        G.add_edge(src, tgt, weight=weight)

    # Step 4: Draw circular layout
    pos = nx.circular_layout(G)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges with scaled color, width, and arrow size
    for u, v, d in G.edges(data=True):
        weight = d['weight']
        color = 'red' if weight > 0 else 'blue'
        alpha = min(1, max(0.3, abs(weight)))
        width = abs(weight) * 2

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=color,
            width=width,
            alpha=alpha,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.2'
        )

    plt.title(f'Top {title} Biomarker Interactions')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.clf()

plot_biomarker_circle(W, biomarker_columns)


