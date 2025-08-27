def plot_individual_survivals(cph, df, id_col="dog_id", duration_col="duration", event_col="event", n_ids=3):
    """
    Plots survival curves for n_ids individuals with an event=1 outcome,
    along with vertical lines marking their actual death time.
    
    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        A fitted Cox proportional hazards model.
    df : pd.DataFrame
        Dataset containing covariates, id_col, duration_col, event_col.
    id_col : str
        Column name for individual IDs.
    duration_col : str
        Column name for the survival time / age.
    event_col : str
        Column name for the event indicator (1 = death, 0 = censored).
    n_ids : int
        Number of individuals to plot.
    """
    
    # Filter to individuals who died
    died_df = df[df[event_col] == 1]
    unique_ids = died_df[id_col].unique()
    
    if len(unique_ids) < n_ids:
        raise ValueError(f"Not enough death events to sample {n_ids} individuals.")
    
    # Randomly pick individuals
    chosen_ids = random.sample(list(unique_ids), n_ids)
    
    plt.figure(figsize=(10, 6))
    
    for i, cid in enumerate(chosen_ids):
        indiv_data = df[df[id_col] == cid]
        death_age = indiv_data[duration_col].max()
        
        # Get last observation (covariates just before/at death)
        last_obs = indiv_data[indiv_data[duration_col] == death_age]
        
        # Drop survival-specific columns before prediction
        covariates = last_obs.drop(columns=[id_col, duration_col, event_col])
        
        # Predict survival function
        surv_func = cph.predict_survival_function(covariates)
        
        # Plot survival curve
        plt.plot(surv_func.index, surv_func.iloc[:, 0], label=f"{cid}")
        
        # Add vertical line at death age
        plt.axvline(death_age, color=plt.gca().lines[-1].get_color(),
                    linestyle="--", alpha=0.7)
    
    plt.xlabel("Time / Age")
    plt.ylabel("Predicted Survival Probability")
    plt.title("Individual Survival Predictions vs Actual Death")
    plt.legend(title=id_col)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_individual_survivals(cph, train_df, id_col="dog_id", duration_col="death_age", event_col="event", n_ids=3)
