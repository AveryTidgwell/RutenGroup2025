def preprocessing():
    # Retrieve .csv files
    CBC_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/CBC Dog.csv'))
    EOLS_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/EOLS Dog.csv'))
    meta_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/Metadata Dog.csv'))
    ov_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/Overview Dog.csv'))
    
    df = pd.read_csv(CBC_path)
    EOLS_df = pd.read_csv(EOLS_path)
    meta_df = pd.read_csv(meta_path)
    ov_df = pd.read_csv(ov_path)
    
    
    # Remove excess columns
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(['sample_year', 'dap_sample_id', 'krt_cbc_abs_metamyelocytes', 'krt_cbc_rel_metamyelocytes', 'krt_cbc_abs_other_cells', 'krt_cbc_abs_bands', 'krt_cbc_abs_basophils', 'krt_cbc_abs_other_cells', 'krt_cbc_rel_metamyelocytes', 'krt_cbc_rel_bands', 'krt_cbc_rel_basophils', 'krt_cbc_rel_other_cells', 'krt_cbc_pcv', 'krt_cbc_plt_quant', 'krt_cbc_plt_morph', 'krt_cbc_plt_morph_2', 'krt_cbc_plt_morph_num', 'krt_cbc_plt_morph_num_2', 'krt_cbc_nucleated_rbcs'], axis = 1)
    df = df.iloc[:, :22]

    # Remove dogs with only 1 sample
    id_counts = df['dog_id'].value_counts()
    df = df[df['dog_id'].isin(id_counts[id_counts > 1].index)]
    
    # Append date of birth
    ov_df.columns = ov_df.columns.str.strip().str.lower()
    
    df = df.merge(ov_df[['dog_id', 'estimated_dob']],
                          on='dog_id',
                          how='left')
                          
    # Append sample dates
    meta_df.columns = meta_df.columns.str.strip().str.lower()

    meta_cbc = meta_df[meta_df['sample_type'] == 'CBC']
    meta_cbc = meta_cbc[['dog_id', 'sample_collection_datetime']]

    df = df.merge(meta_cbc, on='dog_id', how='left')
    df = df.rename(columns={'sample_collection_datetime': 'sample_date'})
    
    # Append mortality data
    df = df.merge(EOLS_df[['dog_id', 'eol_death_date']], on='dog_id', how='left')
    
    # Determine age at sample
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    df['estimated_dob'] = pd.to_datetime(df['estimated_dob'], errors='coerce')
    df['Age'] = (df['sample_date'] - df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Determine age at death (where applicable)
    df['eol_death_date'] = pd.to_datetime(df['eol_death_date'], errors='coerce')
    df['death_age'] = (df['eol_death_date'] - df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Drop redundant columns
    df = df.drop(['sample_date', 'estimated_dob', 'eol_death_date'], axis = 1)
    biomarker_columns = df.columns[1:22]
    
    # Append covariate information
    col = 'sex_class_at_hles'
    ov_df[col] = ov_df[col].astype(str).str.lower().str.strip()

    ov_df['Sex'] = ov_df[col].apply(
        lambda x: 'M' if 'm' in x else ('F' if 'f' in x else np.nan)
    )

    ov_df['Fixed'] = ov_df[col].apply(
        lambda x: 'intact' if 'intact' in x else ('fixed' if 'spayed' in x or 'neutered' in x else np.nan)
    )

    df = df.merge(ov_df[['dog_id', 'Sex', 'Fixed']], on='dog_id', how='left')
    
    # Drop ageless rows
    df = df.dropna(subset=['Age'])

    return df, biomarker_columns

  
df, biomarker_columns = preprocessing()





def get_z_variables(W, mu, df, biomarker_columns, plotname=None):
    eigenvalues, eigenvectors = np.linalg.eig(W)
    real_eigenvalue_order = np.argsort(-eigenvalues.real)

    sorted_eigenvalues = eigenvalues[real_eigenvalue_order]
    sorted_eigenvectors = eigenvectors[:, real_eigenvalue_order]
    P_inv = np.linalg.inv(sorted_eigenvectors)

    z_biomarkers = np.matmul(P_inv, df[biomarker_columns].T.to_numpy()).T
    z_mu = np.matmul(P_inv, mu.T.to_numpy()).T

    natural_var_names = [f'z_{i+1}' for i in range(P_inv.shape[0])]
    natural_mu_names  = [f'mu_z_{i+1}' for i in range(P_inv.shape[0])]
    lambda_names = [f'lambda_{i+1}' for i in range(P_inv.shape[0])]

    z_bio_df = pd.DataFrame(z_biomarkers.real, columns=natural_var_names)
    z_bio_df = z_imputation(z_bio_df, imputation_type='mean')
    z_mu_df = pd.DataFrame(z_mu.real, columns=natural_mu_names)

    z_df = pd.concat([z_bio_df, z_mu_df], axis=1)
    z_df[['dog_id', 'Sex', 'Fixed', 'Age']] = df[['dog_id', 'Sex', 'Fixed', 'Age']].copy()

    # Rank map: for each stability rank i, what original z_col was it?
    # If real_eigenvalue_order[0] == 3, then z_4 is least stable and should be plotted first
    z_col_rank_map = [f'z_{i+1}' for i in real_eigenvalue_order]
    mu_col_rank_map = [f'mu_z_{i+1}' for i in real_eigenvalue_order]

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
    
        # Ensure the directory exists
        save_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(save_dir, exist_ok=True)  # <-- create folder if needed
    
        plt.savefig(os.path.join(save_dir, plotname + '.png'), dpi=300)
        plt.show()

    return z_df, z_mu_df, biomarker_weights, sorted_eigenvalues, z_col_rank_map, mu_col_rank_map















