import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm  # Progress bar

file_path = os.path.abspath(os.path.join('/Volumes/Health Files/', 'TransplantDataRepo-PhysicsValidation_DATA_LABELS_2024-11-13_1619.csv'))


def remove_nan_data(df):
    # Drop columns with all NA values
    all_na_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=all_na_columns)

    # Safely drop optional columns if they exist
    cols_to_drop = [
        'Recipient province:   Move all province data to this field if empty',
        'Preemptive status', 'Access', 'Dialysis vintage', 'Dialysis Type',
        'UA Blood', 'UA Nitrite', 'UA RBC', 'UA Protein', 'CMV IgG Ab EIA',
        'Ven Potassium', 'U MALB Creat Ratio', 'C-Peptide', 'Art Potassium',
        'Art PCO2', 'Repeat Instrument', 'Repeat Instance',
        'Recipient date of birth:', 'DM', 'Smoker:', 'BMI', 'Body weight',
        'Height', 'Admission date', 'Discharge date', 'HLA mismatch',
        'Lab time:', 'Accession number:', 'Complete?', 'date of check',
        'Transplant date:', 'Glucose Random',
        # These are the newly failing ones:
        'LVEF', 'Cause of ESRD', 'Cause of ESRD: Description (from FORT data set):',
        'ESRD #2', 'Cause of ESRD 2: Description (from FORT data set):',
        'ESRD #3', 'Cause of ESRD 3: Description (from FORT data set):'
    ]

    # Only drop columns that exist
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    return df

df = pd.read_csv(file_path, index_col=None)
df = remove_nan_data(df)
biomarker_columns = df.columns[5:14]

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")

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
        colnames = biomarker_columns
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
    df['dt'] = df.groupby('Record ID')['Age'].diff(periods=-1)
    df['sqrt-dt']=np.sqrt(df['dt'])
    df['dV'] = df.groupby('Record ID')[col].diff(periods=-1)
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

def count_appearances(df,plotname="Frequency histogram", plot_histogram=False, filename = 'appearance_frequency.png' , id_col = 'Record ID'):

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
    # Clean and map DM Type
    df['DM Type'] = df['DM Type'].astype(str).str.replace(' ', '').str.lower()

    def map_dm_type(val):
        if '1' in val:
            return 1
        elif '2' in val:
            return 2
        else:
            return 0  # default for unknown/missing
    df['DM Type'] = df['DM Type'].apply(map_dm_type)

    # Clean and map Sex
    def map_sex(val):
        val = str(val).strip().lower()
        if val == 'female':
            return 0
        elif val == 'male':
            return 1
        else:
            return np.nan  # missing or unrecognized

    df['Sex'] = df['Sex'].apply(map_sex)

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

    # Normalize so that 0 is in the center
    norm = mcolors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())
    
    ax = sns.heatmap(W, annot=False, cmap='bwr', norm=norm, xticklabels=biomarker_names, yticklabels=biomarker_names)
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


def print_prediction(data, W, L, biomarker_columns, id=4, biomarker = 2, filename='pred.png'):
    y_pred = pd.DataFrame(
        prediction(data['y_cur'].to_numpy(), W, data['delta_t'].to_numpy(), estimate_mu(L, data['x_cov'].to_numpy())))

    # Assuming df, y_pred, y_next, delta_t, etc. are already defined
    df = data['df_valid']
    index = df[df['Record ID'] == id].index
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

def remove_nan_data(df):
    # Drop columns with all NA values
    all_na_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=all_na_columns)

    # Safely drop optional columns if they exist
    cols_to_drop = [
        'Recipient province:   Move all province data to this field if empty',
        'Preemptive status', 'Access', 'Dialysis vintage', 'Dialysis Type',
        'UA Blood', 'UA Nitrite', 'UA RBC', 'UA Protein', 'CMV IgG Ab EIA',
        'Ven Potassium', 'U MALB Creat Ratio', 'C-Peptide', 'Art Potassium',
        'Art PCO2', 'Repeat Instrument', 'Repeat Instance',
        'Recipient date of birth:', 'DM', 'Smoker:', 'BMI', 'Body weight',
        'Height', 'Admission date', 'Discharge date', 'HLA mismatch',
        'Lab time:', 'Accession number:', 'Complete?', 'date of check',
        'Transplant date:', 'Glucose Random',
        # These are the newly failing ones:
        'LVEF', 'Cause of ESRD', 'Cause of ESRD: Description (from FORT data set):',
        'ESRD #2', 'Cause of ESRD 2: Description (from FORT data set):',
        'ESRD #3', 'Cause of ESRD 3: Description (from FORT data set):'
    ]

    # Only drop columns that exist
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

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


def process_data(df, kalman = False):
    df, biomarker_columns = prepreprocessing(
    df,
    remove_missing_dialysis_start=True,
    save_path="/Volumes/Health Files/cleaned_transplant_data_with_times.csv"
    )
    biomarker_columns = df.columns[5:14]
    for col in biomarker_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for id in df['Record ID'].unique():
        df.loc[df['Record ID'] == id, biomarker_columns] = (
            df.loc[df['Record ID'] == id, biomarker_columns].interpolate(method='linear', limit_direction='both'))

    #quantile_lower = 0.02
    #quantile_upper = 0.98
    
    #for biomarker in biomarker_columns:
        #q_low = df[biomarker].quantile(quantile_lower)
        #q_high = df[biomarker].quantile(quantile_upper)
        #df.loc[(df[biomarker] < q_low) | (df[biomarker] > q_high), biomarker] = np.nan


    df = imputation(df, imputation_type = "mean")

    # Create a tuple of identifying fields
    df['Record ID'] = list(zip(df['Record ID'], df['Sex'], df['DM Type']))
    # Convert each unique tuple to a unique number
    df['Record ID'] = df.groupby('Record ID').ngroup()+1

    df = log_transform(df) #to convert left skewed data to one that resembles normal distribution

    #normalize to mean 0 and std 1
    df = normalize_biomarkers(df, biomarker_columns)
    
    return df, biomarker_columns

def prepare_model_data(df, kalman = False, save_path = None):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman)
    
    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['Record ID', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('Record ID')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('Record ID')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'DM Type', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index

    # Now filter all arrays using cleaned valid_rows
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to: {save_path}")
    
    
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, df, biomarker_columns

def prepare_spaced_data(df, kalman = False, save_path = None):
    df = df.copy()
    df, biomarker_columns = process_data(df, kalman)
    
    df = thin_by_fixed_spacing(df)
    
    # Create 'ones' column for bias term (after processing)
    df['ones'] = 1
    df['Age_norm'] = normalize(df,'Age')
    df = df.sort_values(by=['Record ID', 'Age'], ascending=[True, True]).reset_index(drop=True)
    # Calculate delta_t per animal (age difference to next row)
    delta_t = df.groupby('Record ID')['Age'].diff(periods=-1).reset_index(level=0, drop=True)
    y_cur = df[biomarker_columns].reset_index(level=0, drop=True)
    y_next = df.groupby('Record ID')[biomarker_columns].shift(-1).reset_index(level=0, drop=True)

    # Covariates (repeat Age_norm if needed for L matrix)
    x_cov = df[['Sex', 'DM Type', 'Age_norm', 'ones']]

    # Identify valid rows (where biomarker change is defined)
    valid_rows = y_next.dropna().index

    # Now filter all arrays using cleaned valid_rows
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to: {save_path}")
    
    
    # Filter all arrays by valid rows
    return {
        'y_next': y_next.loc[valid_rows],
        'y_cur': y_cur.loc[valid_rows],
        'x_cov': x_cov.loc[valid_rows],
        'delta_t': -delta_t.loc[valid_rows],
        'df_valid': df.loc[valid_rows]
    }, df, biomarker_columns

def prepreprocessing(df, remove_missing_dialysis_start=True, save_path=None):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert date columns
    df['Lab date:'] = pd.to_datetime(df['Lab date:'], errors='coerce')
    df['Dialysis start Date'] = pd.to_datetime(df['Dialysis start Date'], errors='coerce')

    # Remove people with no dialysis start date
    if remove_missing_dialysis_start:
        valid_ids = df.groupby('Record ID')['Dialysis start Date'].transform('first').notna()
        df = df[valid_ids]

    # Remove people who have no Lab dates at all
    lab_valid = df.groupby('Record ID')['Lab date:'].transform('count') > 0
    df = df[lab_valid]

    print(f"Unique patients after initial filtering: {df['Record ID'].nunique()}")
    print(f"Remaining rows: {df.shape[0]}")

    df = df.sort_values(by=['Record ID', 'Lab date:'])

    # Interpolate missing Lab dates
    df['Lab date:'] = (
        df.groupby('Record ID')['Lab date:']
        .transform(lambda x: pd.to_datetime(x.astype('int64').interpolate(method='linear')))
    )

    # Assign first dialysis start date per ID
    df['Dialysis start Date'] = df.groupby('Record ID')['Dialysis start Date'].transform('first')

    # Calculate Age using true dialysis start date
    df['Age'] = (df['Lab date:'] - df['Dialysis start Date']).dt.total_seconds() / (365.25 * 24 * 3600)

    # Exclude IDs with Age at 2nd sample ≥ 0.25 years (3 months)
    second_sample_age = df.groupby('Record ID').nth(1)['Age']
    bad_ids = second_sample_age[second_sample_age >= 0.25].index
    df = df[~df['Record ID'].isin(bad_ids)]

    # Normalize 'Sex' before aggregation
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.strip().str.lower().replace({'nan': np.nan})
        df['Sex'] = df.groupby('Record ID')['Sex'].transform(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)

    # Normalize 'DM Type' before aggregation
    if 'DM Type' in df.columns:
        df['DM Type'] = df['DM Type'].astype(str).str.strip().replace({'nan': np.nan})
        df['DM Type'] = df.groupby('Record ID')['DM Type'].transform(lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan)

    # Keep only samples within [0.25, 5] years
    df = df[(df['Age'] >= 0.25) & (df['Age'] <= 5.0)]


    # Encode categorical fields
    df = encode_categorical(df)
    
    # Remove duplicate Lab dates per ID by keeping the more complete row
    def keep_most_complete(group):
        if len(group) <= 1:
            return group
        nan_counts = group.isna().sum(axis=1)
        return group.loc[[nan_counts.idxmin()]]

    df = (
        df.groupby(['Record ID', 'Lab date:'], group_keys=False)
          .apply(keep_most_complete)
          .reset_index(drop=True)
    )

    df = remove_nan_data(df)
    report_na_distribution(df)  # Or whatever your cleaned DataFrame is
    
    na_percent = df.isna().mean() * 100
    print(na_percent.sort_values(ascending=False))

    print(f"Remaining unique patients: {df['Record ID'].nunique()}")
    print(f"Remaining rows: {df.shape[0]}")
    
    df = remove_rows_with_most_nas(df, percent=10)  # removes top 5% most NA rows
    report_na_distribution(df)  # Or whatever your cleaned DataFrame is
    # Show NA summary
    na_percent = df.isna().mean() * 100
    print(na_percent.sort_values(ascending=False))

    print(f"Remaining unique patients: {df['Record ID'].nunique()}")
    print(f"Remaining rows: {df.shape[0]}")
    
    biomarker_columns = df.columns[5:14]
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved cleaned data to: {save_path}")

    return df, biomarker_columns

def report_na_distribution(df):
        # Count number of NAs per row
        na_counts = df.isna().sum(axis=1)

        # Count how many rows have 0, 1, 2, ... NAs
        distribution = na_counts.value_counts().sort_index()

        # Convert to percentage
        percentage = (distribution / len(df)) * 100

        # Print the results
        print("NA count per row distribution:")
        for na_count, pct in percentage.items():
            print(f"{na_count} NA(s): {pct:.1f}%")

def remove_rows_with_most_nas(df, percent=5):
        """
        Removes the top `percent` of rows that have the most missing (NA) values.
    
        Parameters:
        - df: pandas DataFrame
        - percent: float, percentage of rows to remove (e.g., 5 for 5%)
    
        Returns:
        - Cleaned DataFrame
        """
        n_remove = int(len(df) * percent / 100)
    
        # Count NA per row
        na_counts = df.isna().sum(axis=1)
    
        # Get indices of rows with most NAs
        worst_rows = na_counts.nlargest(n_remove).index
    
        # Drop those rows
        df_cleaned = df.drop(index=worst_rows)
        
        os.makedirs(os.path.dirname('/Volumes/Health Files/df_with_dropped_NAs.csv'), exist_ok=True)
        df_cleaned.to_csv('/Volumes/Health Files/df_with_dropped_NAs.csv', index=False)
        
        print(f"Removed {len(worst_rows)} rows with the most NA values.")
        return df_cleaned
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tqdm import tqdm  # Progress bar

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
n_bootstraps = 200
kalman = False
sample_limit = 32
output_pdf = 'Alt_32_sample_cap_kidney.pdf'
data_path = '/Volumes/Health Files/'
file_path = os.path.abspath(os.path.join('/Volumes/Health Files/', 'TransplantDataRepo-PhysicsValidation_DATA_LABELS_2024-11-13_1619.csv'))

# ========== LOAD & CLEAN DATA ==========
df = pd.read_csv(file_path, index_col=None)

# ========== BASELINE TRANSFORMATION ==========
data0, df0, biomarker_columns = prepare_model_data(df, kalman=kalman)
W0, L0 = linear_regression(data0)
mu0 = estimate_mu(L0, data0['x_cov'])

eigenvalues0, eigenvectors0 = np.linalg.eig(W0)
abs_order = np.argsort(-eigenvalues0.real)
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
    z_df[['Age', 'Record ID']] = df_valid[['Age', 'Record ID']].values
    z_mu_df = pd.DataFrame(z_mu.real, columns=[f'mu_z_{i+1}' for i in range(P_inv.shape[0])])
    z_mu_df[['Age', 'Record ID']] = df_valid[['Age', 'Record ID']].values
    return z_df, z_mu_df

z_df0, z_mu_df0 = reproject_to_W0_basis(data0['df_valid'], mu0, P_inv0, biomarker_columns)
z_df0 = imputation(z_df0, imputation_type='mean')

print(f"Type of z_col_rank_map: {type(z_col_rank_map)}")
print(f"First few z_col_rank_map entries: {z_col_rank_map[:3]}")
print(f"First few mu_col_rank_map entries: {mu_col_rank_map[:3]}")
print(f"Length z_col_rank_map: {len(z_col_rank_map)}; mu_col_rank_map: {len(mu_col_rank_map)}")
for entry in zip(z_col_rank_map, mu_col_rank_map):
    print("Entry:", entry)
    print("Length:", len(entry))
    break  # Only print one

# ========== BOOTSTRAP: IN-MEMORY STORAGE ONLY ==========
z_df_list = []
z_mu_df_list = []
eigen_boot = []

print("Running bootstraps by Record ID with per-dolphin sample cap...")
unique_ids = df['Record ID'].unique()

for i in tqdm(range(n_bootstraps), desc="ID Bootstrap w/ Sample Cap"):
    np.random.seed(i)

    # 1. Bootstrap dolphins (with replacement)
    sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)

    # 2. Cap samples per dolphin
    sampled_df_list = []
    for aid in sampled_ids:
        group = df[df['Record ID'] == aid]
        sampled_rows = group.sample(n=sample_limit, replace=True, random_state=i)
        sampled_df_list.append(sampled_rows)

    # 3. Combine
    df_capped = pd.concat(sampled_df_list, ignore_index=True)

    # 4. Continue with transformation and projection
    data, df_capped_cleaned, biomarker_columns = prepare_model_data(df_capped, kalman=kalman)
    try:
        W, L = linear_regression(data)
    except ValueError as e:
        continue

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
                              z_df_list, z_mu_df_list, eigen_boot, P_inv0, biomarker_columns,
                              output_pdf, W0):
    os.makedirs('Volumes/Health Files/censoring_results', exist_ok=True)
    pdf_path = os.path.join('Volumes/Health Files/censoring_results', output_pdf)

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
            lowess_stack = []
            for z_df_j, z_mu_df_j in zip(z_df_list, z_mu_df_list):
                try:
                    df_mu = z_mu_df_j[['Age', mu_col]].dropna().sort_values('Age')
                    if len(df_mu) >= 2:
                        slope_j, intercept_j = np.polyfit(df_mu['Age'], df_mu[mu_col], deg=1)
                        yj = slope_j * x_pred + intercept_j
                        mu_boot_stack.append(yj)

                    df_z = z_df_j[['Age', z_col]].dropna().sort_values('Age')
                    if len(df_z) >= 2:
                        lowess_j = lowess(df_z[z_col], df_z['Age'], frac=0.2, return_sorted=True)
                        yj_interp = np.interp(x_lowess0, lowess_j[:, 0], lowess_j[:, 1])
                        lowess_stack.append(yj_interp)
                except:
                    continue

            if mu_boot_stack:
                mu_boot_stack = np.vstack(mu_boot_stack)
                y_mu_std = mu_boot_stack.std(axis=0)
                plt.fill_between(x_pred, y_mu0 - y_mu_std, y_mu0 + y_mu_std, color='red', alpha=0.3, label='±1 SD (Linear Fit)')

            if lowess_stack:
                lowess_stack = np.vstack(lowess_stack)
                y_lowess_std = lowess_stack.std(axis=0)
                plt.fill_between(x_lowess0, y_lowess0 - y_lowess_std, y_lowess0 + y_lowess_std, color='magenta', alpha=0.2, label='±1 SD (LOWESS)')

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

        # ========== Final eigenvalue plot ==========
        if eigen_boot:
            eigen_boot = np.vstack(eigen_boot)
        else:
            print("No eigenvalues were collected during bootstrapping!")
            return

        mean_eigs = np.mean(eigen_boot, axis=0)
        std_eigs = np.std(eigen_boot, axis=0)
        eigvals0 = np.linalg.eigvals(W0)
        eigvals0_sorted = eigvals0[np.argsort(-eigvals0.real)]

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

# Call the plotting function
plot_all_z_with_bootstrap(
    z_df0=z_df0,
    z_mu_df0=z_mu_df0,
    sorted_eigenvalues=sorted_eigenvalues,
    z_col_rank_map=z_col_rank_map,
    mu_col_rank_map=mu_col_rank_map,
    z_df_list=z_df_list,
    z_mu_df_list=z_mu_df_list,
    eigen_boot=eigen_boot,
    P_inv0=P_inv0,
    biomarker_columns=biomarker_columns,
    output_pdf=output_pdf,
    W0=W0
)

end_time = datetime.now()
elapsed = end_time - start_time
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {elapsed.total_seconds():.2f} seconds")
