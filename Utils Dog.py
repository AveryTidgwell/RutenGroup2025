import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import matplotlib.colors as mcolors

current_dir = os.getcwd()

def z_score_for_percentile(p):
    return abs(norm.ppf(p))
  
def log_transform(dx, colnames=None):
    for col in colnames:
        if col in dx.columns:
            # Add the smallest positive non-zero value as a pseudocount
            min_val = dx[col][dx[col] > 0].min()
            dx[col] = np.log(dx[col] + min_val)
    return dx

def remove_outliers(df, col, outlier_percentile):
    z_threshold = z_score_for_percentile(outlier_percentile)
    mean = df[col].mean()
    std = df[col].std()
    z_scores = abs(df[col] - mean)/std
    return df[z_scores <= z_threshold]

def normalize_dolphin_paper(x,biomarker_columns, below_age=5, outlier_percentile=0.01):
    #Inputs: x - biomarker value, t - time
    #Goal: to remove linear drift and normalize

    original_df = df.copy()

    #remove dolphins below certain age
    df = df[df['Age'] >= below_age]

    #biomarkers start here
    for col in biomarker_columns:
        df_col = df[['Age',col]]
        df_wo = remove_outliers(df_col, col, outlier_percentile)

        #remove linear drift
        #if df_wo.shape[0]>0:
            # Fit: x = a + b*t
           # reg = LinearRegression(fit_intercept=True)
           # reg.fit(df_wo[['Age']], df_wo[col])
            #b = reg.coef_   #slope
            #a = reg.intercept_  #y-intercept

            #predicted_x = reg.predict(original_df[['Age']])
            #original_df[col] = original_df[col] - predicted_x  # shape (n_samples, d)

        #normalize to mean 0 and variance 1
        mean = original_df[col].mean()
        std  = original_df[col].std()
        original_df[col] =  (original_df[col] - mean)/std

    return original_df

def normalize(df,col):
    mean = df[col].mean()
    std = df[col].std()
    return  (df[col] - mean) / std

def normalize_biomarkers(df, biomarker_columns):
    for col in biomarker_columns:
        # normalize to mean 0 and variance 1
        df[col] = normalize(df, col)
    return df

def plot_biomarker_drift(df,col, plotname):
    df['dt'] = df.groupby('dog_id')['Age'].diff(periods=-1)
    df['sqrt-dt']=np.sqrt(df['dt'])
    df['dV'] = df.groupby('dog_id')[col].diff(periods=-1)
    #df= log_transform(df,colnames=['dV'])
    df_clean = df.dropna(subset=['sqrt-dt', 'dV'])
    plt.figure(figsize=(6,6))

    plt.plot(df_clean['sqrt-dt'],df_clean['dV'],'.k')

    max_sqrt_time = 1
    min_sqrt_time = 0
    interval = max_sqrt_time/12
    for i in range(0,12):
        start_t = min_sqrt_time + interval*i
        end_t = start_t + interval
        middle_t = (start_t + end_t)/2
        mean = df_clean[(df_clean['sqrt-dt'] >= start_t) & (df_clean['sqrt-dt'] < end_t)]['dV'].mean()
        std = df_clean[(df_clean['sqrt-dt'] >= start_t) & (df_clean['sqrt-dt'] < end_t)]['dV'].std()
        plt.plot(middle_t,mean,'ro')
        plt.plot(middle_t,mean+std,'yo')
        plt.plot(middle_t,mean-std,'yo')

    plt.xlim(0,0.8)
    plt.ylim(-2,2)
    plt.xlabel('sqrt(dt)')
    plt.ylabel('dV')
    plt.title(plotname)
    plt.savefig(os.path.join(current_dir, '..', 'figures', plotname + '.png'), dpi=300, bbox_inches='tight')
    return

def count_appearances(df,plotname="Frequency histogram", plot_histogram=False, filename = 'appearance_frequency.png' , id_col = 'dog_id'):
    # Count number of appearances for each DolphinID
    counts = df[id_col].value_counts()
    if plot_histogram:
        # Plot histogram
        plt.figure(figsize=(10, 6))
        counts.hist(bins=30)
        plt.xlabel('Number of Appearances')
        plt.ylabel('Number of individuals')
        plt.title(plotname)
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(current_dir, '..', 'figures', filename), dpi=300, bbox_inches='tight')

    # Keep only IDs with 2 or more appearances
    valid_ids = counts[counts >= 2].index
    # Filter the original dataframe
    df = df[df[id_col].isin(valid_ids)]
    return df

def encode_categorical(df):
    #not a nice way to do it, but ok for now
    # Dictionary to map
    sex_map = {'F': 0, 'M': 1}
    fixed_map = {'fixed': 0, 'intact': 1}

    df['Sex'] = df['Sex'].str.strip() #has some whitespaces
    df['Fixed'] = df['Fixed'].str.strip() #has some whitespaces

    df['Sex'] = df['Sex'].map(sex_map)
    df['Fixed'] = df['Fixed'].map(fixed_map)
    return df


def print_predicted_biomarker(y_pred, y_next_true, age, plotname='Biomarker Prediction',
                              filename='biomarker_plot.png'):
    plt.figure(figsize=(10, 6))

    # Plot true vs predicted biomarker values
    plt.plot(age, y_next_true, marker='o', label='True')
    plt.plot(age, y_pred, marker='s', label='Predicted')

    plt.xlabel('Age')
    plt.ylabel(f'Biomarker Value')
    plt.title(plotname)
    plt.legend()
    plt.grid(True)

    # Save figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(current_dir, '..', 'figures', filename), dpi=300, bbox_inches='tight')
    plt.show()
    return

def plot_biomarker_heatmap(W, biomarker_names, plotname ='Heatmap Biomarkers' ):
    plt.figure(figsize=(6, 6))
    cmap = 'bwr'

    # Normalize so that 0 is in the center
    norm = mcolors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())
    
    ax = sns.heatmap(W, annot=False, cmap=cmap, norm=norm, xticklabels=biomarker_names, yticklabels=biomarker_names)
    ax.set(xlabel='',ylabel='')
    ax.set_aspect('equal')
    plt.title(plotname)
    plt.xlabel('Biomarkers')
    plt.ylabel('Biomarkers')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(current_dir, '..', 'figures', plotname + '.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    del ax
    return



from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_normal_heatmap(W, biomarker_names, plotname='Heatmap Biomarkers'):
    plt.figure(figsize=(6, 6))
    cmap = 'bwr'

    # Normalize W to [-1, 1]
    W = W / np.abs(W).max()

    # Normalize colors to center at 0
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # Create main axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create divider to host colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Draw heatmap
    sns.heatmap(W, ax=ax, cbar=True, cbar_ax=cax, cmap=cmap, norm=norm,
                xticklabels=biomarker_names, yticklabels=biomarker_names)

    # Formatting
    ax.set(xlabel='', ylabel='')
    ax.set_aspect('equal')
    ax.set_title(plotname)
    ax.set_xlabel('Biomarkers')
    ax.set_ylabel('Biomarkers')
    plt.tight_layout()
    plt.show()

    # Save and cleanup
    #plt.savefig(os.path.join(current_dir, '..', 'figures', plotname + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_prediction(data, W, L, biomarker_columns, id=4, biomarker = 2, filename='pred.png'):
    y_pred = pd.DataFrame(
        prediction(data['y_cur'].to_numpy(), W, data['delta_t'].to_numpy(), estimate_mu(L, data['x_cov'].to_numpy())))

    # Assuming df, y_pred, y_next, delta_t, etc. are already defined
    df = data['df_valid']
    index = df[df['dog_id'] == id].index
    bio_index = biomarker
    #print(y_pred.loc[index, bio_index].head(10))
    #print(data['y_next'].loc[index, biomarker_columns[bio_index]].head(10))
    print_predicted_biomarker(
        y_pred=y_pred.loc[index, bio_index],  # The predicted biomarker values (from first column)
        y_next_true=data['y_next'].loc[index, biomarker_columns[bio_index]],  # The true biomarker values
        age=df.loc[index, 'Age'].shift(-1),  # Age values for the specific animal
        filename = filename
    )
    return

def remove_nan_data(df, threshold):
    rm_col = check_nan(df, threshold)
    df = df.drop(columns=rm_col) #drop any columns that have more than 40% nan
    return df

def check_nan(df, threshold=None):
    # Check if any NaN values exist
    nan_check = df.isna().sum().sum()

    # Check if there are any empty strings (assuming empty strings are considered "empty")
    empty_check = (df == '').sum().sum()

    # Print results
    if nan_check > 0:
        print(f"DataFrame contains {nan_check} NaN values.")
    else:
        print("No NaN values in the DataFrame.")

    if empty_check > 0:
        print(f"DataFrame contains {empty_check} empty string values.")
    else:
        print("No empty string values in the DataFrame.")

    # Check for NaN values in each column
    if not threshold:
        return
    else:
        nan_counts = df.isna().sum()/df.shape[0]
        columns_with_40plus_nans = nan_counts[nan_counts > threshold].index.tolist()
        return columns_with_40plus_nans
