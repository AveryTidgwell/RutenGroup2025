from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

z_matrix = z_df0[z_cols].values
pca = PCA(n_components=1)
health_scores = pca.fit_transform(StandardScaler().fit_transform(z_matrix))

z_df0['health'] = -health_scores[:, 0]  # negate if higher PC1 = worse health

def hazard_function(health, lambda0=0.027, beta=1.0):
    return lambda0 * np.exp(beta * health)

def simulate_death_age(df_i, lambda0=0.027, beta=1.0):
    age_seq = df_i['Age'].values
    health_seq = df_i['health'].values
    dt = np.diff(age_seq, prepend=age_seq[0])

    for age, h, delta in zip(age_seq, health_seq, dt):
        hazard = hazard_function(h, lambda0=lambda0, beta=beta)
        death_prob = 1 - np.exp(-hazard * delta)
        if np.random.rand() < death_prob:
            return age
    return age_seq[-1]  # Survives through final time

death_ages = []
for aid, df_i in z_df0.groupby('AnimalID'):
    death_age = simulate_death_age(df_i, lambda0=0.01, beta=1.0)
    death_ages.append((aid, death_age))

# After collecting death_ages
ages = [age for aid, age in death_ages]
avg_lifespan = np.mean(ages)
simulated_annual_mortality = 1 / avg_lifespan  # crude estimate
print(simulated_annual_mortality)


from scipy.optimize import minimize

def loss(params):
    lambda0, beta = params
    if lambda0 <= 0 or beta < 0:
        return np.inf

    death_ages = []
    for aid, df_i in z_df0.groupby('AnimalID'):
        death_age = simulate_death_age(df_i, lambda0=lambda0, beta=beta)
        death_ages.append(death_age)
    
    avg_life = np.mean(death_ages)
    mortality_rate = 1 / avg_life
    return (mortality_rate - 0.027)**2  # squared error

# Initial guess
init_params = [0.01, 1.0]
res = minimize(loss, init_params, bounds=[(1e-4, 0.1), (0.0, 5.0)])
lambda0_opt, beta_opt = res.x
print(f"Optimal λ₀: {lambda0_opt:.5f}, β: {beta_opt:.3f}")

