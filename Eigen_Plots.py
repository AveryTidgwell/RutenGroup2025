import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler

def plot_normalized_mean_eigenvalues_smooth(mean_eigs):
    # Rank and reverse
    n = len(mean_eigs)
    x = np.arange(n)[::-1]  # reversed rank: 43, 42, ..., 0

    # Normalize x and y
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    y_norm = scaler.fit_transform(mean_eigs.reshape(-1, 1)).flatten()

    # Fit LOWESS
    lowess_result = lowess(y_norm, x_norm, frac=0.3, return_sorted=True)
    x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]

    # Fit degree-3 polynomial
    coeffs = np.polyfit(x_norm, y_norm, deg=3)
    poly_fit = np.poly1d(coeffs)
    y_poly = poly_fit(x_norm)

    # Plot
    plt.figure(figsize=(8, 6))
    #plt.plot(x_norm, y_norm, 'o', color='red', label='Normalized Mean Eigenvalues')
    #plt.plot(x_lowess, y_lowess, color='blue', label='LOWESS Fit', lw=2)
    plt.plot(x_norm, y_poly, color='green', label='Degree-3 Polynomial Fit', lw=2)

    plt.xlabel('Normalized Eigenvalue Rank')
    plt.ylabel('Normalized Mean Eigenvalue')
    plt.title('Normalized Mean Eigenvalues with LOWESS and Polynomial Fit')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()






def plot_bootstrap_eigenvalues(W0, bootstrap_dir, n_bootstraps=250):
    eigen_boot = []
    for j in range(n_bootstraps):
        try:
            with open(os.path.join(bootstrap_dir, f'W_{j:04d}.pkl'), 'rb') as f:
                W_j = pickle.load(f)
            eigvals = np.linalg.eigvals(W_j)
            eigvals_sorted = eigvals[np.argsort(np.abs(eigvals))]  # sort by abs value
            eigen_boot.append(np.real(eigvals_sorted))
        except Exception:
            continue

    if not eigen_boot:
        print("No valid bootstrap eigenvalues found.")
        return

    # Stack and summarize
    eigen_boot = np.vstack(eigen_boot)
    mean_eigs = np.mean(eigen_boot, axis=0)
    std_eigs = np.std(eigen_boot, axis=0)

    eigvals0 = np.linalg.eigvals(W0)
    eigvals0_sorted = np.real(eigvals0[np.argsort(np.abs(eigvals0))])
    x = np.arange(1, len(eigvals0_sorted) + 1)

    # Plot
    plt.figure(figsize=(8, 6))

    # Baseline W0 (black dots + SD)
    plt.errorbar(x, eigvals0_sorted, yerr=std_eigs, fmt='o', capsize=3, color='black', label='Baseline Population Values')

    # Bootstrap mean eigenvalues (red dots + SD)
    plt.errorbar(x, mean_eigs, yerr=std_eigs, fmt='o', capsize=3, color='red', label='Bootstrap Mean')

    plt.ylim(-2,0)
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalues of $W$ with ±1 SD Error Bars')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()




plot_bootstrap_eigenvalues(W0=W0, bootstrap_dir=save_dir, n_bootstraps=n_bootstraps)
