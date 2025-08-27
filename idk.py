from statistics import mean, stdev

for column in epsilon.columns:
    abs_noise = [abs(x) for x in epsilon[column]]
    threshold = mean(abs_noise) + stdev(abs_noise)
    
    count = sum(1 for x in abs_noise if x > threshold)
    print(f"{column}: {count} values above mean+std")
    
    threshold2 = threshold + stdev(abs_noise)
    count2 = sum(1 for x in abs_noise if x > threshold2)
    print(f"{column}: {count2} values above mean+2std")
    
    threshold3 = threshold2 + stdev(abs_noise)
    count3 = sum(1 for x in abs_noise if x > threshold3)
    print(f"{column}: {count3} values above mean+3std")
    
    threshold4 = threshold3 + stdev(abs_noise)
    count4 = sum(1 for x in abs_noise if x > threshold4)
    print(f"{column}: {count4} values above mean+4std")
    
    threshold5 = threshold4 + stdev(abs_noise)
    count5 = sum(1 for x in abs_noise if x > threshold5)
    print(f"{column}: {count5} values above mean+5std")



# Store results for each threshold
threshold_counts = {i: {} for i in range(1, 11)}

# Step 1: Compute counts above thresholds for each biomarker column
for column in epsilon.columns:
    abs_noise = [abs(x) for x in epsilon[column]]
    mu = mean(abs_noise)
    sigma = stdev(abs_noise)

    for i in range(1, 11):
        threshold = mu + i * sigma
        count = sum(1 for x in abs_noise if x > threshold)
        biomarker_name = biomarker_columns[column]  # Map column index to name
        threshold_counts[i][biomarker_name] = count

# Step 2: Print top 5 biomarker names at each threshold
for i in range(1, 11):
    print(f"\n--- Top 5 Biomarkers above mean + {i}·std ---")
    sorted_items = sorted(threshold_counts[i].items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_items[:5]:
        print(f"{name}: {count} values above threshold")
        
        
        
        
        
        
        
        
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Prepare data
data = []
for i in range(1, 6):
    for biomarker, count in threshold_counts[i].items():
        data.append({
            'Biomarker': biomarker,
            'Threshold': f'mean+{i}·std',
            'Count': count
        })

df_plot = pd.DataFrame(data)

# Step 2: Filter to top 10 biomarkers
top_biomarkers = (
    df_plot.groupby('Biomarker')['Count']
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .index
)
df_top = df_plot[df_plot['Biomarker'].isin(top_biomarkers)]

# Step 3: Plot with count labels
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=df_top,
    x='Biomarker',
    y='Count',
    hue='Threshold',
    palette='coolwarm'
)
plt.title('Top 10 Biomarkers by High-Noise Count at Increasing Thresholds')
plt.ylabel('Number of High-Noise Values')
plt.xticks(rotation=45)
plt.tight_layout()

# Step 4: Add labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=2)

plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean, stdev

# Step 1: Extend threshold_counts up to mean + 9·std
threshold_counts = {i: {} for i in range(1, 10)}

for col in epsilon.columns:
    abs_noise = [abs(x) for x in epsilon[col]]
    mu = mean(abs_noise)
    sigma = stdev(abs_noise)
    biomarker = biomarker_columns[col]

    for i in range(1, 10):
        threshold = mu + i * sigma
        count = sum(1 for x in abs_noise if x > threshold)
        threshold_counts[i][biomarker] = count

# Step 2: Plot five graphs, each sorted by threshold n
for n in range(1, 6):
    print(f"Creating plot for threshold mean + {n}·std...")

    # Thresholds to include in this plot: n through n+4
    thresholds_to_plot = list(range(n, n + 5))
    threshold_labels = [f'mean+{i}·std' for i in thresholds_to_plot]

    # Get counts for all thresholds and biomarkers
    data = []
    for t in thresholds_to_plot:
        for biomarker, count in threshold_counts[t].items():
            data.append({
                'Biomarker': biomarker,
                'Threshold': f'mean+{t}·std',
                'Count': count
            })
    df_plot = pd.DataFrame(data)

    # Sort biomarkers by threshold n
    top_biomarkers = (
        df_plot[df_plot['Threshold'] == f'mean+{n}·std']
        .sort_values(by='Count', ascending=False)
        .head(10)['Biomarker']
        .tolist()
    )

    df_top = df_plot[df_plot['Biomarker'].isin(top_biomarkers)]
    df_top['Biomarker'] = pd.Categorical(df_top['Biomarker'], categories=top_biomarkers, ordered=True)

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_top,
        x='Biomarker',
        y='Count',
        hue='Threshold',
        hue_order=threshold_labels,
        palette='coolwarm'
    )

    # Bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=2)

    # Labels and formatting
    plt.title(f'Top 10 Biomarkers by High-Noise Count (Sorted by mean+{n}·std)')
    plt.ylabel('Number of High-Noise Values')
    plt.xlabel('Biomarker')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




