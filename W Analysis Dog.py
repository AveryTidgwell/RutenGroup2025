import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
current_dir = os.getcwd()

def get_z_variables(W, mu, df, biomarker_columns, plotname = None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    print(np.real(eigenvalues))
    P_inv = np.linalg.inv(eigenvectors)
    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T

    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=biomarker_columns)
    z_mu_df = pd.DataFrame( z_mu.real, columns="mu_" + biomarker_columns)

    z_df = pd.concat([z_bio_df,z_mu_df],axis = 1)
    z_df[['dog_id','Sex','Fixed','Age']] = df[['dog_id','Sex','Fixed','Age']].copy()
    
    if plotname is not None:
        # Sort eigenvalues and corresponding biomarkers
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        sorted_biomarkers = [biomarker_columns[i] for i in np.argsort(eigenvalues)[::-1]]

        # Plot the sorted eigenvalues with biomarker names on the x-axis
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_biomarkers, y=sorted_eigenvalues, palette="viridis")
        plt.xlabel("Biomarker Name")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W by Biomarker")
        plt.xticks(rotation=90)  # Rotate x labels if needed for better readability
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300, bbox_inches='tight')

    return z_df, z_mu_df



def plot_z(z_df, dolphin_id, biomarker):
    print(z_df.head(2))
    t = z_df[z_df['dog_id'] == dolphin_id]['Age']
    z_bio = z_df[z_df['dog_id'] == dolphin_id][biomarker]
    z_mu = z_df[z_df['dog_id'] == dolphin_id]["mu_"+biomarker]
    plt.figure(figsize=(8, 6))
    plt.plot(t,z_bio,marker='o')
    plt.plot(t,z_mu)

    plt.xlabel('Age')
    plt.ylabel('z_variable ' + biomarker )
    plotname = 'Z of ' + biomarker + ' for dolphin # ' + str(dolphin_id)
    plt.savefig(os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    return

def plot_mean_bio(data, L, bio_columns, bio_name, cov_col, plotname):
    df = data['df_valid']

    # Prepare L coefficients
    L = pd.DataFrame(L.T)
    L.columns = df[bio_columns].columns

    # Normalize age
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()

    # Create age range
    age_range = np.arange(df['Age'].min(), df['Age'].max(), 0.1)
    age_range_norm = (age_range-mean_age)/std_age

    # Extract coefficients
    mu0 = L[bio_name][3]
    mu_sex = L[bio_name][0]
    mu_age = L[bio_name][2]
    mu_Fixed = L[bio_name][1]

    # Group data and compute means by bin
    bins = np.arange(0, 100, 5)
    df['Age_bin'] = pd.cut(df['Age'], bins)



    # Compute trendlines in standardized space
    trendlineF = [mu0 + mu_sex * 0 + mu_age * age for age in age_range_norm]
    trendlineM = [mu0 + mu_sex * 1 + mu_age * age for age in age_range_norm]



    male = df[df['Sex'] == 1]
    female = df[df['Sex'] == 0]

    male_means = pd.DataFrame()
    male_means['Age'] = male.groupby('Age_bin')['Age'].mean()
    male_means[bio_name] = male.groupby('Age_bin')[bio_name].mean().values

    female_means = pd.DataFrame()
    female_means['Age'] = female.groupby('Age_bin')['Age'].mean().values
    female_means[bio_name] = female.groupby('Age_bin')[bio_name].mean().values

    # Plot using numeric age for proper alignment
    plt.figure(figsize=(10, 6))
    plt.plot(male_means['Age'], male_means[bio_name], 'bo-', label='Male (Avg)')
    plt.plot(female_means['Age'], female_means[bio_name], 'ro-', label='Female (Avg)')
    plt.plot(age_range, trendlineM, 'b--', label='Male Trendline')
    plt.plot(age_range, trendlineF, 'r--', label='Female Trendline')

    plt.xlabel('Age')
    plt.ylabel(bio_name)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(current_dir, 'Downloads', 'dolphins-master', 'results', plotname), dpi=300, bbox_inches='tight')
    plt.close()
    return

