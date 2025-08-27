def preprocessing():
    # === Load data ===
    CBC_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/CBC Dog.csv'))
    EOLS_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/EOLS Dog.csv'))
    meta_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/Metadata Dog.csv'))
    ov_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/Overview Dog.csv'))
    chem_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'DOGGIES/Chemistry Dog.csv'))
    
    df = pd.read_csv(CBC_path)
    EOLS_df = pd.read_csv(EOLS_path)
    meta_df = pd.read_csv(meta_path)
    ov_df = pd.read_csv(ov_path)
    chem_df = pd.read_csv(chem_path)
    
    # === Preprocess CBC ===
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(['sample_year', 'dap_sample_id', 'krt_cbc_abs_metamyelocytes', 'krt_cbc_rel_metamyelocytes', 'krt_cbc_abs_other_cells', 'krt_cbc_abs_bands', 'krt_cbc_abs_basophils', 'krt_cbc_abs_other_cells', 'krt_cbc_rel_metamyelocytes', 'krt_cbc_rel_bands', 'krt_cbc_rel_basophils', 'krt_cbc_rel_other_cells', 'krt_cbc_pcv', 'krt_cbc_plt_quant', 'krt_cbc_plt_morph', 'krt_cbc_plt_morph_2', 'krt_cbc_plt_morph_num', 'krt_cbc_plt_morph_num_2', 'krt_cbc_nucleated_rbcs', 'krt_cbc_rel_neutrophils', 'krt_cbc_rel_lymphocytes', 'krt_cbc_rel_monocytes', 'krt_cbc_rel_eosinophils', 'krt_cbc_retic_per'], axis = 1)
    df = df.iloc[:, :17]
    id_counts = df['dog_id'].value_counts()
    df = df[df['dog_id'].isin(id_counts[id_counts > 1].index)]
    ov_df.columns = ov_df.columns.str.strip().str.lower()
    
    df = df.merge(ov_df[['dog_id', 'estimated_dob']], on='dog_id', how='left')
    meta_df.columns = meta_df.columns.str.strip().str.lower()
    meta_cbc = meta_df[meta_df['sample_type'] == 'CBC'][['dog_id', 'sample_collection_datetime']]
    df = df.merge(meta_cbc, on='dog_id', how='left')
    df = df.rename(columns={'sample_collection_datetime': 'sample_date'})
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    df['estimated_dob'] = pd.to_datetime(df['estimated_dob'], errors='coerce')
    df['Age'] = (df['sample_date'] - df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
    df['sample_date'] = df['sample_date'].dt.normalize()
    
    # === Preprocess Chemistry ===
    chem_df.columns = chem_df.columns.str.strip().str.lower()
    chem_df = chem_df.drop(['sample_year', 'dap_sample_id', 'krt_cp_total_protein_modifier', 'krt_cp_albumin_modifier', 'krt_cp_globulins_modifier', 'krt_cp_alb_glob_ratio_value', 'krt_cp_alb_glob_ratio_modifier', 'krt_cp_calcium_modifier', 'krt_cp_phosphorus_modifier', 'krt_cp_magnesium_modifier', 'krt_cp_glucose_modifier', 'krt_cp_bun_modifier', 'krt_cp_creatinine_modifier', 'krt_cp_bilirubin_total_modifier', 'krt_cp_alkp_modifier', 'krt_cp_alt_modifier', 'krt_cp_ggt_modifier', 'krt_cp_amylase_modifier', 'krt_cp_triglycerides_modifier', 'krt_cp_cholesterol_modifier', 'krt_cp_sodium_modifier', 'krt_cp_potassium_modifier', 'krt_cp_chloride_modifier', 'krt_cp_sp_ratio_value', 'krt_cp_sp_ratio_modifier', 'krt_cp_test_comments'], axis = 1)
    meta_chem = meta_df[meta_df['sample_type'] == 'Chemistry Panel'][['dog_id', 'sample_collection_datetime']]
    chem_df = chem_df.merge(ov_df[['dog_id', 'estimated_dob']], on='dog_id', how='left')
    chem_df = chem_df.merge(meta_chem, on='dog_id', how='left')
    chem_df = chem_df.rename(columns={'sample_collection_datetime': 'sample_date'})
    chem_df['sample_date'] = pd.to_datetime(chem_df['sample_date'], errors='coerce')
    chem_df['estimated_dob'] = pd.to_datetime(chem_df['estimated_dob'], errors='coerce')
    chem_df['Age'] = (chem_df['sample_date'] - chem_df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
    chem_df['sample_date'] = chem_df['sample_date'].dt.normalize()

    # === Add source tags to identify origin ===
    df['source'] = 'cbc'
    chem_df['source'] = 'chem'

    # === Standardize column sets for union ===
    all_cols = set(df.columns).union(chem_df.columns)
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
        if col not in chem_df.columns:
            chem_df[col] = np.nan

    # === Concatenate CBC and Chemistry rows ===
    df = pd.concat([df, chem_df], ignore_index=True)

    # === Group by dog_id and sample_date, merge duplicate-day entries ===
    df = df.sort_values(['dog_id', 'sample_date', 'source']).reset_index(drop=True)
    df = df.groupby(['dog_id', 'sample_date'], as_index=False).first()

    # === Merge mortality ===
    EOLS_df.columns = EOLS_df.columns.str.strip().str.lower()
    df = df.merge(EOLS_df[['dog_id', 'eol_death_date']], on='dog_id', how='left')
    df['eol_death_date'] = pd.to_datetime(df['eol_death_date'], errors='coerce')
    df['death_age'] = (df['eol_death_date'] - df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)

    # === Sex & fixed status ===
    col = 'sex_class_at_hles'
    ov_df[col] = ov_df[col].astype(str).str.lower().str.strip()
    ov_df['Sex'] = ov_df[col].apply(lambda x: 'M' if 'm' in x else ('F' if 'f' in x else np.nan))
    ov_df['Fixed'] = ov_df[col].apply(lambda x: 'intact' if 'intact' in x else ('fixed' if 'spayed' in x or 'neutered' in x else np.nan))
    df = df.merge(ov_df[['dog_id', 'Sex', 'Fixed']], on='dog_id', how='left')
  
    # === Clean up ===
    df = df.drop(columns=['sample_date', 'estimated_dob', 'eol_death_date', 'source'])
    df = df.dropna(subset=['Age'])

    # Remove 'Age' and reinsert it at target index
    cols = list(df.columns)
    cols.remove('Age')
    cols.insert(len(df.columns)-4, 'Age')

    # Reorder the DataFrame
    df = df[cols]
    
    # === Get biomarker columns ===
    df = df.rename(columns={'krt_cbc_total_wbcs': 'wbc',
                            'krt_cbc_abs_neutrophils': 'neutrophils',
                            'krt_cbc_abs_lymphocytes': 'lymphocytes',
                            'krt_cbc_abs_monocytes': 'monocytes',
                            'krt_cbc_abs_eosinophils': 'eosinophils',
                            'krt_cbc_rbc': 'rbc',
                            'krt_cbc_hgb': 'hgb',
                            'krt_cbc_hct': 'hct',
                            'krt_cbc_mcv': 'mcv',
                            'krt_cbc_mch': 'mch',
                            'krt_cbc_mchc': 'mchc',
                            'krt_cbc_rdw': 'rdw',
                            'krt_cbc_plt': 'plt',
                            'krt_cbc_mpv': 'mpv',
                            'krt_cbc_pct': 'pct',
                            'krt_cbc_retic_abs': 'reticulocytes',
                            'krt_cp_chloride_value': 'chloride',
                            'krt_cp_triglycerides_value': 'triglycerides',
                            'krt_cp_alt_value': 'alt',
                            'krt_cp_bun_value': 'bun',
                            'krt_cp_alkp_value': 'alkphos',
                            'krt_cp_potassium_value': 'potassium',
                            'krt_cp_phosphorus_value': 'phosphorus',
                            'krt_cp_total_protein_value': 'protein',
                            'krt_cp_globulins_value': 'globulins',
                            'krt_cp_calcium_value': 'calcium',
                            'krt_cp_albumin_value': 'albumin',
                            'krt_cp_sodium_value': 'sodium',
                            'krt_cp_glucose_value': 'glucose',
                            'krt_cp_cholesterol_value': 'cholesterol',
                            'krt_cp_creatinine_value': 'creatinine',
                            'krt_cp_ggt_value': 'ggt',
                            'krt_cp_bilirubin_total_value': 'bilirubin',
                            'krt_cp_magnesium_value': 'magnesium',
                            'krt_cp_amylase_value': 'amylase'
                            })
    biomarker_columns = df.columns[1:36]
    
    return df, biomarker_columns


df, biomarker_columns = preprocessing()
print(biomarker_columns)
