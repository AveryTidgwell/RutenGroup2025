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
