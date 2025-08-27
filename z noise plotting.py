def z_noise(z_df, W, z_mu_df):
    # Select relevant z columns
    z = z_df[[col for col in z_df.columns if col.startswith("z_") and not col.startswith("mu_")]].copy()

    # Compute Δz = z(t+1) - z(t)
    delta_z = z.diff(-1)

    # Time differences
    t = z_df['Age']
    delta_t = t.diff(-1)  # shape (n_samples,)

    # Get eigenvalues from W (43 x 43)
    eigenvalues, _ = np.linalg.eig(W)
    eigenvalues = np.real(eigenvalues)  # shape (43,)

    # Ensure mu values are aligned
    mu = z_mu_df.values  # shape (43,)

    # (z - mu): shape (n_samples, 43)
    z_minus_mu = z - mu  # pandas will align along columns

    # Reshape delta_t and eigenvalues for broadcasting
    delta_t = delta_t.values[:, np.newaxis]       # shape (n_samples, 1)
    eigenvalues = eigenvalues[np.newaxis, :]      # shape (1, 43)

    # Compute deterministic part
    deterministic = eigenvalues * delta_t * z_minus_mu  # shape (n_samples, 43)

    # Compute noise
    noise = delta_z - deterministic

    return noise, eigenvalues.flatten()

noise, eigenvalues = z_noise(z_df, W, z_mu_df)

z_noise_var = noise.var(skipna=True)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_noise_variance_summary(sigma_squared, output_pdf='z_noise_summary.pdf'):
    plt.figure(figsize=(10, 6))
    z_labels = sigma_squared.index  # e.g., ['z_1', 'z_2', ..., 'z_43']
    sigma_vals = sigma_squared.values

    sns.barplot(x=z_labels, y=sigma_vals, palette='magma')
    plt.xticks(rotation=90)
    plt.ylabel(r"$\sigma_i^2$ (Noise Variance)")
    plt.title("Noise Variance by z-variable")
    plt.tight_layout()
    
    # Save to PDF
    pdf_path = os.path.join('Downloads', 'dolphins-master', 'results', 'Smoother_z', output_pdf)
    plt.savefig(pdf_path)
    plt.show()
    plt.close()
    print(f"Noise variance summary saved to {pdf_path}")

plot_noise_variance_summary(z_noise_var)












