import pandas as pd
import numpy as np

def imputation(df, imputation_type):
    if imputation_type == "mean":
        # Fill numeric columns with their own mean
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        if df.isnull().sum().sum() !=0: # See if any NaNs remain
            print('There are still NA values in the loaded dataset')
        return df
    else:
        print("Imputation is implemented for mean only")
        return np.nan
      
import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
import pandas as pd

#calculate prediction
def prediction(y_cur, W, delta_t, mu, nsteps=1):
    '''
    :param y_cur: N x D (number of dolphins x number of biomarkers)
    :param W: D x D
    :param delta_t: N x 1
    :param mu: N x D (number of dolphins x number of biomarkers)
    :return:  y_pred N x D
    '''
    if delta_t.ndim == 1:
        delta_t = delta_t.reshape(-1, 1)

    if  nsteps==1:
        diff = (y_cur - mu)*delta_t #N x D, by broadcasting, element wise multiplication
        y_pred = y_cur + np.matmul(W, diff.T).T #N x D
    else:
        small_t = delta_t/nsteps
        for i in range(nsteps):
            diff = (y_cur - mu)*small_t #N x D, by broadcasting, element wise multiplication
            y_cur = y_cur + np.matmul(W, diff.T).T #N x D
        y_pred = y_cur
    return y_pred

#define the cost function
def loss_function(residual):
    '''
    :param residual: y_pred - y_true
    :return: Mean Squared Error
    '''
    return np.mean(residual ** 2)

#define partial derivatives (Jacobian)
def jacobian_dw(residual, y_cur, mu, delta_t):
    '''
    :param residual: N x D
    :param y_cur: N x D
    :param mu: N x D
    :param delta_t: N x 1
    :return:
    '''
    N, D = y_cur.shape
    diff = (y_cur - mu)*delta_t #N x D, by broadcasting, element wise multiplication
    J = 2*np.matmul(diff.T, residual)/(N*D)
    return J.T

def jacobian_lambda(residual, W, x_cov, delta_t):
    '''
    :param residual: N x D
    :param W: D x D
    :param x_cov: N x C
    :param delta_t: N x 1
    :return: Jacobian N x C
    '''
    N, C = x_cov.shape
    D, _ = W.shape
    x_cov_scaled = x_cov * delta_t
    rw = residual @ W #N x D
    J = rw.T @ x_cov_scaled              # (D x C)
    return -2*J/(N*D)

def check_jacobian_dl(N,D,C):
    np.random.seed(0)
    y_cur = np.random.normal(size=(N, D))
    y_next_true = np.random.normal(size=(N, D))
    delta_t = np.random.normal(size=(N, 1))
    #W = np.random.normal(size=(D, D))
    W = -np.eye(D) * 0.5 + 0.01 * np.random.randn(D, D)
    L = np.random.normal(size=(D, C))
    x_cov = np.random.random(size = (N,C))

    residual = prediction(y_cur, W, delta_t, estimate_mu(L, x_cov))- y_next_true
    # Analytical Jacobian
    an_gradL = jacobian_lambda(residual, W, x_cov, delta_t)  # shape D x D

    # Numerical Jacobian
    eps = 1e-5
    num_gradL = np.zeros_like(L)
    for i in range(D):
        for j in range(C):
            L_pos = L.copy()
            L_neg = L.copy()
            L_pos[i, j] += eps
            L_neg[i, j] -= eps

            loss_pos = loss_function(prediction(y_cur, W, delta_t, estimate_mu(L_pos, x_cov))- y_next_true)
            loss_neg = loss_function(prediction(y_cur, W, delta_t, estimate_mu(L_neg, x_cov))- y_next_true)

            num_gradL[i, j] = (loss_pos - loss_neg) / (2 * eps)

    diff = np.abs(an_gradL - num_gradL)
    print("Max abs diff:", np.max(diff))
    print("Mean abs diff:", np.mean(diff))
    return

def check_jacobian_dw(N,D):
    np.random.seed(0)
    y_cur = np.random.normal(size=(N, D))
    y_next_true = np.random.normal(size=(N, D))
    delta_t = np.random.normal(size=(N, 1))
    mu = np.random.normal(size=(N, D))
    W = np.random.normal(size=(D, D))

    # Analytical Jacobian
    residual = prediction(y_cur, W, delta_t, mu) - y_next_true
    an_gradW = jacobian_dw(residual, y_cur, mu, delta_t)  # shape D x D

    # Numerical Jacobian
    eps = 1e-5
    num_gradW = np.zeros_like(W)
    for i in range(D):
        for j in range(D):
            W_pos = W.copy()
            W_neg = W.copy()
            W_pos[i, j] += eps
            W_neg[i, j] -= eps

            loss_pos = loss_function(prediction(y_cur, W_pos, delta_t, mu)- y_next_true)
            loss_neg = loss_function(prediction(y_cur, W_neg, delta_t, mu)- y_next_true)

            num_gradW[i, j] = (loss_pos - loss_neg) / (2 * eps)

    diff = np.abs(an_gradW - num_gradW)
    print("Max abs diff:", np.max(diff))
    print("Mean abs diff:", np.mean(diff))
    return
  
  
def estimate_mu(Lambda, x_cov):
    '''
    :param Lambda: D x C
    :param x_cov: N x C
    :return:
    '''
    return  np.matmul(Lambda,x_cov.T).T

def gradient_descent(data, learning_rate=1e-2, num_iter=500, seed=0, nsteps = 1, tolerance=1e-9):

    y_next=data['y_next'].to_numpy()
    y_cur=data['y_cur'].to_numpy()
    x_cov=data['x_cov'].to_numpy()
    delta_t=data['delta_t'].to_numpy()

    if not seed:
        np.random.seed(seed)

    N,D = y_cur.shape
    N,C = x_cov.shape
    W = -np.eye(D) * 0.5 + 0.01 * np.random.randn(D, D)
    L = 0.01*np.random.normal(size = (D,C))
    history = []

    delta_t = delta_t.reshape(-1, 1)
    for i in range(num_iter):
        mu = estimate_mu(L, x_cov)
        residual = prediction(y_cur, W, delta_t, mu, nsteps) - y_next
        loss = loss_function(residual)
        gradW = jacobian_dw(residual, y_cur, mu, delta_t)
        gradL =jacobian_lambda(residual, W, x_cov, delta_t)
        W = W - learning_rate*gradW
        L = L - learning_rate*gradL
        #learning_rate /= (1 + i * 1e-4)
        history.append(loss)
        if i %10000 == 0:
            print(f"Iteration {i}, Cost: {loss}")
            learning_rate = learning_rate/ (1 + i * 1e-5)
            #print(np.mean(np.abs(gradW)))
            #print(np.mean(np.abs(gradL)))

        if len(history) > 1 and abs(history[-1] - history[-2]) < tolerance:
            print(f"Converged at iteration {i}")
            break

    return W, L, history

def test_jacobians():
    check_jacobian_dw(100,5)
    check_jacobian_dl(100,5,4)
    return

def linear_regression(data):
    y_next=data['y_next'].to_numpy()
    y_cur=data['y_cur'].to_numpy()
    x_cov=data['x_cov'].to_numpy()
    delta_t=data['delta_t'].to_numpy().reshape(-1, 1)

    delta_y = y_next-y_cur
    z = np.concatenate([y_cur, x_cov],axis=1)*delta_t
    model = LinearRegression(fit_intercept=False)
    model.fit(z, delta_y)

    theta = model.coef_
    nbio = y_next.shape[1]
    W = theta[:, 0:nbio]
    A = theta[:,nbio:]
    L = inv(W) @ A
    return W, -L


def linear_regression_col_separate(data):
    y_next=data['y_next'].to_numpy()
    y_cur=data['y_cur'].to_numpy()
    x_cov=data['x_cov'].to_numpy()
    delta_t=data['delta_t'].to_numpy().reshape(-1, 1)

    N,D = y_next.shape
    N,C = x_cov.shape
    theta = np.zeros([D, D+C])

    delta_y = y_next-y_cur
    model = LinearRegression(fit_intercept=False)

    for biomarker in range(D):
        z = np.concatenate([y_cur, x_cov],axis=1)*delta_t
        model.fit(z, delta_y[:, biomarker].reshape(-1, 1))  # Fit the model
        theta[biomarker,:]  =  model.coef_

    W = theta[:, 0:D]
    A = theta[:,D:]
    L = -inv(W) @ A
    return W, L


def linear_regression_col_separate2(data):
    y_next = data['y_next'].to_numpy()
    y_cur = data['y_cur'].to_numpy()
    x_cov = data['x_cov'].to_numpy()
    delta_t = data['delta_t'].to_numpy().reshape(-1, 1)
    N, D = y_next.shape
    N, C = x_cov.shape

    W = np.zeros((D, D))
    A = np.zeros((D, C))
    weights = 1 / np.abs(delta_t.flatten())

    for j in range(D):
        # Calculate dyn dynamically for this column (like in R)
        delta_y = (y_next[:, j] - y_cur[:, j]).reshape(-1, 1)

        # Construct the predictors: y (lagged) and x (contemporaneous)
        predictors_y = y_cur * delta_t
        predictors_x = x_cov * delta_t
        predictors = np.hstack([predictors_y, predictors_x])

        # Weighted regression
        model = LinearRegression(fit_intercept=False)
        model.fit(predictors, delta_y, sample_weight=weights)

        # Extract coefficients
        coef = model.coef_.flatten()
        W[j, :] = coef[:D]
        A[j, :] = coef[D:]

        # Debugging: Print intermediate values
        #print(f"\n--- Column {j + 1} ---")
        #print("Delta Y:", delta_y.flatten()[:5])
        #print("Predictors (Y):", predictors_y[:5, :])
        #print("Predictors (X):", predictors_x[:5, :])
        #print("Coefficients (Y):", W[j, :])
        #print("Coefficients (X):", A[j, :])

    # Calculate lambda (L in Python)
    try:
        L = np.linalg.inv(W) @ A
        #print("\nW Matrix:\n", W)
        #print("\nA Matrix:\n", A)
        #print("\nL (Lambda) Matrix:\n", L)
    except np.linalg.LinAlgError:
        #print("\nW is singular, cannot invert.")
        L = np.full((D, C), np.nan)

    return W, -L

#def plot_biomarker_drift2(df, mu0, mu_t):
def calculate_loss(data, W, L):
    pred = pd.DataFrame(prediction(data['y_cur'].to_numpy(), W, data['delta_t'].to_numpy(), estimate_mu(L, data['x_cov'].to_numpy())))
    residual = pred - data['y_next'].to_numpy()
    print(f"Loss: {loss_function(residual)}")
    return
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
    z_df[['AnimalID','Sex','Species','Age']] = df[['AnimalID','Sex','Species','Age']].copy()

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

    return z_df



def plot_z(z_df, dolphin_id, biomarker):
    print(z_df.head(2))
    t = z_df[z_df['AnimalID'] == dolphin_id]['Age']
    z_bio = z_df[z_df['AnimalID'] == dolphin_id][biomarker]
    z_mu = z_df[z_df['AnimalID'] == dolphin_id]["mu_"+biomarker]
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
    mu_species = L[bio_name][1]

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
    if not colnames:
        colnames = ['WBC', 'MCV', 'RDW', 'NRBC', 'ACNeutrophils', 'Lymphs', 'ACLymphocytes', 'Monocytes', 'EOS', 'ACEosinophils',
                'Glucose', 'BUN', 'Creatinine', 'UricAcid', 'Potassium','Protein', 'Calcium', 'AlkPhos', 'LDH', 'AST',
                'ALT', 'GGT', 'Bilirubin', 'Cholesterol', 'Triglyceride', 'Iron', 'CPK', 'SED60', 'GFR']
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
    df['dt'] = df.groupby('AnimalID')['Age'].diff(periods=-1)
    df['sqrt-dt']=np.sqrt(df['dt'])
    df['dV'] = df.groupby('AnimalID')[col].diff(periods=-1)
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

def count_appearances(df,plotname="Frequency histogram", plot_histogram=False, filename = 'appearance_frequency.png' , id_col = 'AnimalID'):
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
    species_map = {'TT': 0, 'TG': 1, 'TTTG':2}

    df['Sex'] = df['Sex'].str.strip() #has some whitespaces
    df['Species'] = df['Species'].str.strip() #has some whitespaces

    df['Sex'] = df['Sex'].map(sex_map)
    df['Species'] = df['Species'].map(species_map)
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
    plt.figure(figsize=(8, 6))
    custom_cmap = ['#add8e6', '#ffffff', '#00008b']  # light blue → white → dark blue
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", custom_cmap)

    # Normalize so that 0 is in the center
    norm = mcolors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())
    
    ax = sns.heatmap(W, annot=False, cmap=cmap, norm=norm, xticklabels=biomarker_names, yticklabels=biomarker_names)
    ax.set(xlabel='',ylabel='')
    plt.title(plotname)
    plt.xlabel('Biomarkers')
    plt.ylabel('Biomarkers')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(current_dir, '..', 'figures', plotname + '.png'), dpi=300, bbox_inches='tight')
    plt.clf()
    del ax
    return


def print_prediction(data, W, L, biomarker_columns, id=4, biomarker = 2, filename='pred.png'):
    y_pred = pd.DataFrame(
        prediction(data['y_cur'].to_numpy(), W, data['delta_t'].to_numpy(), estimate_mu(L, data['x_cov'].to_numpy())))

    # Assuming df, y_pred, y_next, delta_t, etc. are already defined
    df = data['df_valid']
    index = df[df['AnimalID'] == id].index
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
    #df = df.drop(columns=rm_col) #drop any columns that have more than 40% nan
    df = df.drop(['Reason', 'Fasting', 'LabCode', 'Mg'], axis = 1)
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
def get_z_variables(W, mu, df, biomarker_columns, plotname=None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    abs_eigenvalue_order = np.argsort(np.abs(eigenvalues))  # least to most stable

    sorted_eigenvalues = eigenvalues[abs_eigenvalue_order]
    sorted_eigenvectors = eigenvectors[:, abs_eigenvalue_order]
    P_inv = np.linalg.inv(sorted_eigenvectors)

    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(P_inv.shape[0])]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(P_inv.shape[0])]
    lambda_names = [f'lambda_{i+1}' for i in range(P_inv.shape[0])]

    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_bio_df = imputation(z_bio_df, imputation_type='mean')
    z_mu_df = pd.DataFrame(z_mu.real, columns=natural_mu_names)

    z_df = pd.concat([z_bio_df, z_mu_df], axis=1)
    z_df[['AnimalID', 'Sex', 'Species', 'Age']] = df[['AnimalID', 'Sex', 'Species', 'Age']].copy()

    # Rank map: for each stability rank i, what original z_col was it?
    # If abs_eigenvalue_order[0] == 3, then z_4 is least stable and should be plotted first
    z_col_rank_map = [f'z_{i+1}' for i in abs_eigenvalue_order]
    mu_col_rank_map = [f'mu_z_{i+1}' for i in abs_eigenvalue_order]

    biomarker_weights = pd.DataFrame(
        P_inv.real,
        columns=biomarker_columns,
        index=natural_var_names
    )

    if plotname is not None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=lambda_names, y=sorted_eigenvalues.real, marker='o', color="lightseagreen")
        plt.xlabel("Natural Variable")
        plt.ylabel("Eigenvalue")
        plt.title("Sorted Eigenvalues of W (by stability)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'Downloads', 'dolphins-master', 'results', plotname + '.png'), dpi=300)
        plt.show()

    return z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map

def process_data(df, kalman = False):

    df = df[df['Age'] > 5] #remove baby dolphins
    df = remove_nan_data(df, 0.4) #remove columns that contain more than 40% missing data
    
    df = df.sort_values(by=['AnimalID', 'Age']).reset_index(level=0, drop=True)
    biomarker_columns = df.columns[4:]

    #count how many times each dolphin appears, drop ones with 1 appearances, plot histogram
    df = count_appearances(df, plot_histogram=False)

    #fill missing values with mean
    #df = imputation(df,imputation_type="mean")

    #check_nan(df)

    for id in df['AnimalID'].unique():
        df.loc[df['AnimalID'] == id, biomarker_columns] = (
            df.loc[df['AnimalID'] == id, biomarker_columns].interpolate(method='linear', limit_direction='both'))

    df = imputation(df, imputation_type = "mean")
    #check_nan(df)

    # Create a tuple of identifying fields
    df['AnimalID'] = list(zip(df['AnimalID'], df['Sex'], df['Species']))
    # Convert each unique tuple to a unique number
    df['AnimalID'] = df.groupby('AnimalID').ngroup()+1

    #do log transformation (original dolphin paper)
    df = log_transform(df) #to convert left skewed data to one that resembles normal distribution

    #normalize to mean 0 and std 1
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df) #convert sex to [0,1] , species to [0,1,2] and normalize age

    return df, biomarker_columns

def prepare_model_data(df, kalman = False):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman)
    
    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['AnimalID', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('AnimalID')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('AnimalID')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'Species', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index
  
    # Filter outliers
    #for biomarker in biomarker_columns:
        #lower_percentile = df[biomarker].quantile(0.02)
        #upper_percentile = df[biomarker].quantile(0.98)

        #df_filtered = df[
            #(df[biomarker] >= lower_percentile) &
            #(df[biomarker] <= upper_percentile)
        #]


    
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, df
    
    
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
n_bootstraps = 300
quantile_lower = 0.02
quantile_upper = 0.98
kalman = False
output_pdf = f'eighth_dolphins.pdf'
data_path = '/Users/summer/Downloads/dolphins-master/data/dolphin_data.csv'

# ========== LOAD & CLEAN DATA ==========
df = pd.read_csv(data_path, index_col=None, header=4)
biomarker_columns = [col for col in df.columns if col not in ['AnimalID', 'Sex', 'Species', 'Age', 'Reason', 'Fasting', 'LabCode', 'Mg']]

for biomarker in biomarker_columns:
    q_low = df[biomarker].quantile(quantile_lower)
    q_high = df[biomarker].quantile(quantile_upper)
    df.loc[(df[biomarker] < q_low) | (df[biomarker] > q_high), biomarker] = np.nan

# ========== BASELINE TRANSFORMATION ==========
data0, df0 = prepare_model_data(df, kalman=kalman)
W0, L0 = linear_regression(data0)
mu0 = estimate_mu(L0, data0['x_cov'])

eigenvalues0, eigenvectors0 = np.linalg.eig(W0)
abs_order = np.argsort(np.abs(eigenvalues0))
V_sorted = eigenvectors0[:, abs_order]
P_inv0 = np.linalg.inv(V_sorted)
sorted_eigenvalues = eigenvalues0[abs_order]

n_z = P_inv0.shape[0]
z_col_rank_map = [f'z_{i+1}' for i in range(n_z)]
mu_col_rank_map = [f'mu_z_{i+1}' for i in range(n_z)]

def reproject_to_W0_basis(df_valid, mu, P_inv, biomarker_columns):
    z_data = np.matmul(P_inv, df_valid[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    z_df = pd.DataFrame(z_data.real, columns=[f'z_{i+1}' for i in range(P_inv.shape[0])])
    z_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

    z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
    z_mu_df[['Age', 'AnimalID']] = df_valid[['Age', 'AnimalID']].values

    return z_df, z_mu_df

z_df0, z_mu_df0 = reproject_to_W0_basis(data0['df_valid'], mu0, P_inv0, biomarker_columns)
z_df0 = imputation(z_df0, imputation_type='mean')

z_df_list = []
z_mu_df_list = []
eigen_boot = []
boot_df_lengths = []

print("Running bootstraps by AnimalID...")
unique_ids = df['AnimalID'].unique()
n_half = len(unique_ids) // 8

for i in tqdm(range(n_bootstraps), desc="Half-population subsampling"):
    np.random.seed(i)
    sampled_ids = np.random.choice(unique_ids, size=n_half, replace=False)
    df_half = df[df['AnimalID'].isin(sampled_ids)].copy().reset_index(drop=True)
    data, df_half_cleaned = prepare_model_data(df_half, kalman=kalman)
    W, L = linear_regression(data)
    mu = estimate_mu(L, data['x_cov'])

    try:
        eigvals, _ = np.linalg.eig(W)
        eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
        eigen_boot.append(np.real(eigvals_sorted))

        z_df_j, z_mu_df_j = reproject_to_W0_basis(data['df_valid'], mu, P_inv0, biomarker_columns)
        z_df_list.append(z_df_j)
        z_mu_df_list.append(z_mu_df_j)
    except:
        continue

# ========== PLOTTING ==========
def plot_all_z_with_bootstrap(z_df0, z_mu_df0, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map,
                              z_df_list, z_mu_df_list, eigen_boot, output_pdf, W0):
    os.makedirs('results/Smoother_z', exist_ok=True)
    pdf_path = os.path.join('results/Smoother_z', output_pdf)

    with PdfPages(pdf_path) as pdf:
        for i, (z_col, mu_col) in enumerate(zip(z_col_rank_map, mu_col_rank_map)):
            plt.figure(figsize=(9, 8))
            t = z_df0['Age']
            z_vals = z_df0[z_col]
            mu_vals = z_mu_df0[mu_col]
            x_pred = np.linspace(t.min(), t.max(), 200)

            df_lowess0 = pd.DataFrame({'Age': t, 'z': z_vals}).dropna()
            lowess0 = lowess(df_lowess0['z'], df_lowess0['Age'], frac=0.2, return_sorted=True)
            x_lowess0, y_lowess0 = lowess0[:, 0], lowess0[:, 1]
            plt.plot(x_lowess0, y_lowess0, color='magenta', lw=2.5, label='Original LOWESS')

            df_mu0 = pd.DataFrame({'Age': t, 'mu': mu_vals}).dropna()
            if len(df_mu0) >= 2:
                slope, intercept = np.polyfit(df_mu0['Age'], df_mu0['mu'], deg=1)
                y_mu0 = slope * x_pred + intercept
                plt.plot(x_pred, y_mu0, '--', lw=2, color='red', label='Original Linear Fit')

            mu_boot_stack = []
            for df_boot in z_mu_df_list:
                try:
                    df_b = df_boot[['Age', mu_col]].dropna().sort_values('Age')
                    if len(df_b) >= 2:
                        slope_j, intercept_j = np.polyfit(df_b['Age'], df_b[mu_col], deg=1)
                        yj = slope_j * x_pred + intercept_j
                        mu_boot_stack.append(yj)
                except:
                    continue
            if mu_boot_stack:
                mu_boot_stack = np.vstack(mu_boot_stack)
                y_mu_std = mu_boot_stack.std(axis=0)
                plt.fill_between(x_pred, y_mu0 - y_mu_std, y_mu0 + y_mu_std,
                                 color='red', alpha=0.3, label='±1 SD (Linear Fit)')

            lowess_stack = []
            for z_df in z_df_list:
                try:
                    df_b = z_df[['Age', z_col]].dropna().sort_values('Age')
                    if len(df_b) >= 2:
                        lowess_j = lowess(df_b[z_col], df_b['Age'], frac=0.2, return_sorted=True)
                        yj_interp = np.interp(x_lowess0, lowess_j[:, 0], lowess_j[:, 1])
                        lowess_stack.append(yj_interp)
                except:
                    continue
            if lowess_stack:
                lowess_stack = np.vstack(lowess_stack)
                y_lowess_std = lowess_stack.std(axis=0)
                plt.fill_between(x_lowess0, y_lowess0 - y_lowess_std, y_lowess0 + y_lowess_std,
                                 color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

            plt.scatter(t, z_vals, color='gray', alpha=0.1, s=10, label='All Data')
            plt.ylim(-5, 5)
            plt.xlabel('Age (yrs)')
            plt.ylabel(f'Natural Variable: z_{i+1}')
            plt.title(f'z_{i+1}')
            plt.legend(loc='best', fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        eigen_boot = np.vstack(eigen_boot)
        mean_eigs = np.mean(eigen_boot, axis=0)
        std_eigs = np.std(eigen_boot, axis=0)
        eigvals0 = np.linalg.eigvals(W0)
        eigvals0_sorted = eigvals0[np.argsort(np.abs(eigvals0))]

        x = np.arange(1, len(eigvals0_sorted) + 1)

        plt.figure(figsize=(9, 6))
        plt.errorbar(x, eigvals0_sorted.real, yerr=std_eigs, fmt='o', capsize=3, color='black', label='Baseline ±1 SD')
        plt.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o', capsize=3, color='red', label='Bootstrap Mean ±1 SD')
        plt.axhline(0, color='gray', linestyle='--', lw=1)
        plt.xlabel('Eigenvalue Rank')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues of W with ±1 SD Error from Bootstrap')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"All plots (z and eigenvalues) saved to {pdf_path}")


plot_all_z_with_bootstrap(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    z_df_list=z_df_list,
    z_mu_df_list=z_mu_df_list,
    eigen_boot=eigen_boot,
    output_pdf=output_pdf,
    W0=W0
)

end_time = datetime.now()
elapsed = end_time - start_time
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")



