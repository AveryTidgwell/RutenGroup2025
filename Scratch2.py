import pandas as pd
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

#---------------------------------------------------------------
#load data
current_dir = os.getcwd()
file_path = os.path.abspath(os.path.join(current_dir, 'Downloads', 'dolphins-master','data', 'dolphin_data.csv'))
df = pd.read_csv(file_path, index_col=None, header = 4)

print(df.columns)


def plot_mean_bio(df, biomarker, plotname):
    df = data['df_valid']

    for age in range(int(min(df['Age'])), int(max(df['Age'])):
        estimate_mu(L, [])
    mu = estimate_mu(L)
    # Filter data by sex
    male = df[df['Sex'] == 'M']
    female = df[df['Sex'] == 'F']

    # Define age bins
    bins = np.arange(0, 100, 5)

    # Compute the mean biomarker value for each age bin
    male_means = male.groupby(pd.cut(male['Age'], bins))[[biomarker]].mean()
    female_means = female.groupby(pd.cut(female['Age'], bins))[[biomarker]].mean()

    plt.figure(figsize=(8, 6))
    plt.plot(male_means.index.astype(str), male_means[biomarker], color='blue', label='Male', marker='o')
    plt.plot(female_means.index.astype(str), female_means[biomarker], color='red', label='Female', marker='o')
    plt.xlabel('Age')
    plt.ylabel(biomarker)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, '..', 'figures', plotname), dpi=300, bbox_inches='tight')
    plt.show()
    return

plot_mean_bio(df, 'Cholesterol',  plotname = 'cholesterol.png')

