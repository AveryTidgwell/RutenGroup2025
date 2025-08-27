biomarker_data = df_valid[biomarker_columns].copy()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(biomarker_data)

from sklearn.decomposition import PCA

pca = PCA(n_components=len(biomarker_columns))  
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

pca_df['Age'] = df_valid['Age'].values
pca_df['Sex'] = df_valid['Sex'].values
pca_df['AnimalID'] = df_valid['AnimalID'].round().astype(int)

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Age', palette='bwr', alpha=0.7)
plt.title('PCA: PC1 vs PC2 colored by Age')
plt.tight_layout()
plt.show()


explained_var = pca.explained_variance_ratio_
print(f"Explained variance by PC1–PC5: {explained_var[:5]}")
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_var), marker='o')
plt.axhline(y=0.8, c='r', alpha = 0.4, linestyle = '--', label = '80% Variance')
plt.axhline(y=0.95, c='r', linestyle = '--', label = '95% Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()


import numpy as np

# Eigenvalues from PCA (not just variance ratio)
eigenvalues = pca.explained_variance_

# Make a table of PCs with their eigenvalues
eigen_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues,
    'Explained Variance Ratio': explained_var
})

# Filter PCs with eigenvalue > 1
eigen_gt1_df = eigen_df[eigen_df['Eigenvalue'] > 1]

print(f"Number of PCs with eigenvalue > 1: {len(eigen_gt1_df)}")
print(eigen_gt1_df)




biomarker_data = df_valid[biomarker_columns].copy()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(biomarker_data)

from sklearn.decomposition import PCA

pca = PCA(n_components=len(biomarker_columns))  
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

pca_df['Age'] = df_valid['Age'].values
pca_df['Sex'] = df_valid['Sex'].values
pca_df['AnimalID'] = df_valid['AnimalID'].round().astype(int)

plt.figure(figsize=(9, 7))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Sex', palette='Set1', alpha=0.7)
plt.title('PCA: PC1 vs PC2 Coloured by Sex')
plt.grid(True)
plt.tight_layout()
plt.show()








import matplotlib.animation as animation

# Assume you already have pca_df with PC1, PC2 and Age or Species
fig, ax = plt.subplots(figsize=(8, 6))

def animate(i):
    ax.clear()
    if i < 10:
        # gradually show more data
        data = pca_df.iloc[:int((i+1) * len(pca_df) / 10)]
    else:
        data = pca_df

    sns.scatterplot(data=data, x='PC1', y='PC2', hue='Age', palette='viridis', alpha=0.7, ax=ax)
    ax.set_title(f'PCA Projection: Frame {i+1}')
    ax.set_xlim(pca_df['PC1'].min(), pca_df['PC1'].max())
    ax.set_ylim(pca_df['PC2'].min(), pca_df['PC2'].max())
    ax.legend(loc='best', fontsize='small')

ani = animation.FuncAnimation(fig, animate, frames=15, interval=500)
ani.save('pca_projection.gif', writer='pillow')


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from matplotlib.patches import Patch

# Step 1: Standardize biomarker data
X = StandardScaler().fit_transform(df_valid[biomarker_columns])

# Step 2: Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 3: Prepare labels
sex_labels = df_valid['Sex'].astype(int)

# Step 4: Define bin size
bin_size = 0.2

# Round coordinates to bin centers
x_bins = np.round(X_pca[:, 0] / bin_size) * bin_size
y_bins = np.round(X_pca[:, 1] / bin_size) * bin_size
points = np.vstack([x_bins, y_bins, sex_labels]).T

# Map bin to set of sexes
bin_sex_map = defaultdict(set)
for x, y, sex in points:
    bin_sex_map[(x, y)].add(sex)

# Determine color per bin
bin_color_map = {}
for bin_key, sexes in bin_sex_map.items():
    if sexes == {0}:
        bin_color_map[bin_key] = 'deeppink'
    elif sexes == {1}:
        bin_color_map[bin_key] = 'blue'
    else:
        bin_color_map[bin_key] = 'darkorchid'

# Step 5: Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot square markers
for (x, y), color in bin_color_map.items():
    ax.scatter(x, y, color=color, marker='s', s=46, edgecolor='none')

# Set axis limits tight to the data
all_x = np.array([x for x, y in bin_color_map])
all_y = np.array([y for x, y in bin_color_map])
ax.set_xlim(all_x.min() - bin_size, all_x.max() + bin_size)
ax.set_ylim(all_y.min() - bin_size, all_y.max() + bin_size)

# Add legend
legend_elements = [
    Patch(facecolor='deeppink', label='Female-only region'),
    Patch(facecolor='blue', label='Male-only region'),
    Patch(facecolor='darkorchid', label='Overlap region')
]
ax.legend(handles=legend_elements)

# Final touches
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA: PC1 vs PC2 with Sex Overlap (Square Grid)')
ax.grid(False)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
plt.clf()



# ------------------------------------------ #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def animate_expanding_sex_overlap(df_valid, biomarker_columns, bin_size=0.2, age_bin_width=0.5, save_path='expanding_pca_sex_overlap.gif'):
    # Step 1: PCA
    X = StandardScaler().fit_transform(df_valid[biomarker_columns])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Store PCA in df
    df_valid = df_valid.copy()
    df_valid["PC1"] = X_pca[:, 0]
    df_valid["PC2"] = X_pca[:, 1]

    # Define age bins
    age_min, age_max = df_valid["Age"].min(), df_valid["Age"].max()
    age_bins = np.arange(age_min, age_max + age_bin_width, age_bin_width)

    # Compute fixed axis limits
    buffer = bin_size
    pc1_min, pc1_max = df_valid["PC1"].min() - buffer, df_valid["PC1"].max() + buffer
    pc2_min, pc2_max = df_valid["PC2"].min() - buffer, df_valid["PC2"].max() + buffer

    # Set up plot
    fig, ax = plt.subplots(figsize=(6, 6))
    legend_elements = [
        Patch(facecolor='deeppink', label='Female-only region'),
        Patch(facecolor='blue', label='Male-only region'),
        Patch(facecolor='darkorchid', label='Overlap region')
    ]
    ax.set_xlim(pc1_min, pc1_max)
    ax.set_ylim(pc2_min, pc2_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(False)
    plt.tight_layout()

    # Accumulate data bins
    cumulative_bin_map = defaultdict(set)
    scatter_objs = []

    def update(frame):
        age_cutoff = age_bins[frame]
        df_bin = df_valid[df_valid["Age"] <= age_cutoff]

        # Clear existing plotted markers
        for s in scatter_objs:
            s.remove()
        scatter_objs.clear()

        # Round to bin centers
        x_bins = np.round(df_bin["PC1"] / bin_size) * bin_size
        y_bins = np.round(df_bin["PC2"] / bin_size) * bin_size
        sexes = df_bin["Sex"].astype(int)

        # Update cumulative map
        for x, y, sex in zip(x_bins, y_bins, sexes):
            cumulative_bin_map[(x, y)].add(sex)

        # Color bins based on cumulative sex content
        for (x, y), sex_set in cumulative_bin_map.items():
            if sex_set == {0}:
                color = 'deeppink'
            elif sex_set == {1}:
                color = 'blue'
            else:
                color = 'darkorchid'
            s = ax.scatter(x, y, color=color, marker='s', s=46, edgecolor='none')
            scatter_objs.append(s)

        ax.set_title(f'Cumulative PCA Region: Age ≤ {age_cutoff:.1f}')

    # Run animation
    ani = FuncAnimation(fig, update, frames=len(age_bins), interval=500, blit=False)
    ani.save(save_path, writer='pillow', fps=2)

animate_expanding_sex_overlap(df_valid, biomarker_columns, save_path='expanding_overlap.gif')
