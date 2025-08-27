# Old version which drops a variable after each layer where it is never used again

def run_c_index_reduction(input_df, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    final_df = preprocess_final_samples(input_df)  # Your existing preprocessing function
    final_df = prepare_cox_covariates(final_df)
    np.random.seed(random_seed)
    excluded_cols = ['Age', 'event']
    all_covariates = [c for c in final_df.columns if c not in excluded_cols]
    results = []
    current_covariates = all_covariates.copy()

    while len(current_covariates) > 1:
        print(f"\nRunning with {len(current_covariates)} covariates...")
        try:
            bootstrap_c_indices = []

            for _ in range(n_bootstrap):
                df_shuffled = final_df.sample(frac=1, random_state=np.random.randint(0, 1e6)).reset_index(drop=True)
                train_size = int(len(df_shuffled) * (1 - test_frac))
                train_df = df_shuffled.iloc[:train_size]
                test_df = df_shuffled.iloc[train_size:]

                cph = CoxPHFitter()
                cph.fit(train_df[['Age', 'event'] + current_covariates],
                        duration_col='Age', event_col='event')

                test_df = test_df[['Age', 'event'] + current_covariates].copy()
                test_df['risk_score'] = cph.predict_partial_hazard(test_df)

                concordant_list = []
                for _ in range(200):
                    pair = test_df.sample(n=2, replace=False)
                    times = pair['Age'].values
                    risks = pair['risk_score'].values
                    if times[0] == times[1]:
                        continue
                    if (risks[0] > risks[1] and times[0] < times[1]) or \
                       (risks[1] > risks[0] and times[1] < times[0]):
                        concordant_list.append(1)
                    else:
                        concordant_list.append(0)

                if concordant_list:
                    bootstrap_c_indices.append(np.mean(concordant_list))

            mean_c = np.mean(bootstrap_c_indices)
            sd_c = np.std(bootstrap_c_indices)
            print(f"{len(current_covariates)} vars: C-index = {mean_c:.4f} ± {sd_c:.4f}")
            results.append((len(current_covariates), mean_c, sd_c))

        except Exception:
            break

        drop_var = np.random.choice(current_covariates)
        current_covariates.remove(drop_var)

    return np.array(results)
  
  
  
  
# New version that drops n random variables before each bootstrap depending on the layer
  
def run_c_index_reduction(input_df, n_bootstrap=1000, test_frac=0.2, random_seed=42):
    final_df = preprocess_final_samples(input_df)  # Your existing preprocessing function
    final_df = prepare_cox_covariates(final_df)
    np.random.seed(random_seed)
    
    excluded_cols = ['Age', 'event']
    all_covariates = [c for c in final_df.columns if c not in excluded_cols]
    results = []

    # Number of variables to drop in this layer will increase each iteration
    num_drops = 0  

    while len(all_covariates) > num_drops + 1:
        print(f"\nRunning with {len(all_covariates) - num_drops} covariates per bootstrap...")
        try:
            bootstrap_c_indices = []

            for _ in range(n_bootstrap):
                # Randomly drop `num_drops` covariates for this bootstrap run
                dropped_vars = np.random.choice(all_covariates, size=num_drops, replace=False)
                current_covariates = [c for c in all_covariates if c not in dropped_vars]

                # Shuffle and split
                df_shuffled = final_df.sample(frac=1, random_state=np.random.randint(0, 1e6)).reset_index(drop=True)
                train_size = int(len(df_shuffled) * (1 - test_frac))
                train_df = df_shuffled.iloc[:train_size]
                test_df = df_shuffled.iloc[train_size:]

                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(train_df[['Age', 'event'] + current_covariates],
                        duration_col='Age', event_col='event')

                # Risk prediction
                test_df = test_df[['Age', 'event'] + current_covariates].copy()
                test_df['risk_score'] = cph.predict_partial_hazard(test_df)

                # Calculate bootstrap C-index manually
                concordant_list = []
                for _ in range(200):
                    pair = test_df.sample(n=2, replace=False)
                    times = pair['Age'].values
                    risks = pair['risk_score'].values
                    if times[0] == times[1]:
                        continue
                    if (risks[0] > risks[1] and times[0] < times[1]) or \
                       (risks[1] > risks[0] and times[1] < times[0]):
                        concordant_list.append(1)
                    else:
                        concordant_list.append(0)

                if concordant_list:
                    bootstrap_c_indices.append(np.mean(concordant_list))

            # Summary for this number of dropped variables
            mean_c = np.mean(bootstrap_c_indices)
            sd_c = np.std(bootstrap_c_indices)
            print(f"Dropped {num_drops} vars: C-index = {mean_c:.4f} ± {sd_c:.4f}")
            results.append((len(all_covariates) - num_drops, mean_c, sd_c))

        except Exception as e:
            print("Error:", e)
            break

        # Increase number of drops for next layer
        num_drops += 1

    return np.array(results)  
  
  
