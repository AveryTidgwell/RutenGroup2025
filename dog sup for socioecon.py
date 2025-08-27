import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import math
import matplotlib as plt
from sklearn.preprocessing import MultiLabelBinarizer



# ===> Before running, you might have to change the paths and names of the .csv files















# Functions from Aziza

def imputation(df, biomarker_columns, imputation_type):
    if imputation_type == "mean":
        # Ensure biomarker_columns is a list
        biomarker_cols = list(biomarker_columns)

        for col in biomarker_cols:
            df[col] = df[col].fillna(df[col].mean())

        # Check for any remaining NaNs
        if df[biomarker_cols].isnull().values.any():
            print("There are still NA values in the loaded dataset")
        else:
            print("All missing values imputed.")

        return df
    else:
        print("Imputation is implemented for mean only")
        return np.nan

def log_transform(dx, colnames=None):
    for col in colnames:
        if col in dx.columns:
            # Add the smallest positive non-zero value as a pseudocount
            min_val = dx[col][dx[col] > 0].min()
            dx[col] = np.log(dx[col] + min_val)
    return dx

def normalize(df,col):
    mean = df[col].mean()
    std = df[col].std()
    return  (df[col] - mean) / std

def normalize_biomarkers(df, biomarker_columns):
    for col in biomarker_columns:
        # normalize to mean 0 and variance 1
        df[col] = normalize(df, col)
    return df




# My functions for dog data

def preprocessing():
    # === Load data ===
    CBC_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/CBC Dog.csv'))
    EOLS_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/EOLS Dog.csv'))
    meta_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Metadata Dog.csv'))
    ov_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Overview Dog.csv'))
    chem_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Chemistry Dog.csv'))
    env_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Environment Dog.csv'))
    
    df = pd.read_csv(CBC_path)
    EOLS_df = pd.read_csv(EOLS_path)
    meta_df = pd.read_csv(meta_path)
    ov_df = pd.read_csv(ov_path)
    chem_df = pd.read_csv(chem_path)
    env_df = pd.read_csv(env_path)

    print(df.shape)
    print(EOLS_df.shape)
    print(meta_df.shape)


    # === Preprocess CBC ===
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(['sample_year', 'krt_cbc_abs_metamyelocytes', 'krt_cbc_rel_metamyelocytes',
                  'krt_cbc_abs_other_cells', 'krt_cbc_abs_bands', 'krt_cbc_abs_basophils', 'krt_cbc_abs_other_cells',
                  'krt_cbc_rel_metamyelocytes', 'krt_cbc_rel_bands', 'krt_cbc_rel_basophils', 'krt_cbc_rel_other_cells',
                  'krt_cbc_pcv', 'krt_cbc_plt_quant', 'krt_cbc_plt_morph', 'krt_cbc_plt_morph_2', 'krt_cbc_plt_morph_num',
                  'krt_cbc_plt_morph_num_2', 'krt_cbc_nucleated_rbcs', 'krt_cbc_rel_neutrophils', 'krt_cbc_rel_lymphocytes',
                  'krt_cbc_rel_monocytes', 'krt_cbc_rel_eosinophils', 'krt_cbc_retic_per'], axis=1)
    df = df.iloc[:, :18]
    id_counts = df['dog_id'].value_counts()
    df = df[df['dog_id'].isin(id_counts[id_counts > 1].index)]
    ov_df.columns = ov_df.columns.str.strip().str.lower()

    # === Preprocess Chemistry ===
    chem_df.columns = chem_df.columns.str.strip().str.lower()
    chem_df = chem_df.drop(['sample_year','krt_cp_total_protein_modifier', 'krt_cp_albumin_modifier',
                            'krt_cp_globulins_modifier', 'krt_cp_alb_glob_ratio_value', 'krt_cp_alb_glob_ratio_modifier',
                            'krt_cp_calcium_modifier', 'krt_cp_phosphorus_modifier', 'krt_cp_magnesium_modifier',
                            'krt_cp_glucose_modifier', 'krt_cp_bun_modifier', 'krt_cp_creatinine_modifier',
                            'krt_cp_bilirubin_total_modifier', 'krt_cp_alkp_modifier', 'krt_cp_alt_modifier',
                            'krt_cp_ggt_modifier', 'krt_cp_amylase_modifier', 'krt_cp_triglycerides_modifier',
                            'krt_cp_cholesterol_modifier', 'krt_cp_sodium_modifier', 'krt_cp_potassium_modifier',
                            'krt_cp_chloride_modifier', 'krt_cp_sp_ratio_value', 'krt_cp_sp_ratio_modifier',
                            'krt_cp_test_comments'], axis=1)

    # === Add source tags to identify origin ===
    df['source'] = 'cbc'
    chem_df['source'] = 'chem'

    # === Add estimated DOB and sample_date for CBC ===
    dob_df = ov_df[['dog_id', 'estimated_dob']].drop_duplicates()
    df = df.merge(dob_df, on='dog_id', how='left')

    meta_df.columns = meta_df.columns.str.strip().str.lower()
    meta_cbc = meta_df[meta_df['sample_type'] == 'CBC'][['dog_id', 'sample_collection_datetime', 'dap_sample_id']].drop_duplicates()
    meta_cbc = meta_cbc.rename(columns={'sample_collection_datetime': 'sample_date'})
    df = df.merge(meta_cbc, on=['dog_id', 'dap_sample_id'], how='left')


    # === Add estimated DOB and sample_date for Chem ===
    chem_df = chem_df.merge(dob_df, on='dog_id', how='left')
    meta_chem = meta_df[meta_df['sample_type'] == 'Chemistry Panel'][['dog_id', 'sample_collection_datetime', 'dap_sample_id']].rename(
        columns={'sample_collection_datetime': 'sample_date'})
    chem_df = chem_df.merge(meta_chem, on=['dog_id', 'dap_sample_id'], how='left')

    # === Convert to datetime and calculate Age ===
    for sub_df in [df, chem_df]:
        sub_df['age_date'] = sub_df['sample_date'].copy()
        sub_df['age_date'] = pd.to_datetime(sub_df['age_date'], errors='coerce')
        sub_df['estimated_dob'] = pd.to_datetime(sub_df['estimated_dob'], errors='coerce')
        sub_df['Age'] = (sub_df['age_date'] - sub_df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
        sub_df['age_date'] = sub_df['age_date'].dt.normalize()

    # === Debugging prints before concatenation ===
    print(f"CBC rows before concat: {len(df)}")
    print(f"Chemistry rows before concat: {len(chem_df)}")
    print(f"CBC sample dog_ids: {df['dog_id'].unique()[:5]}")
    print(f"Chemistry sample dog_ids: {chem_df['dog_id'].unique()[:5]}")

    # === Align columns before concatenation ===
    all_cols = set(df.columns).union(set(chem_df.columns))
    for col in all_cols:
        if col not in df.columns:
            df[col] = np.nan
        if col not in chem_df.columns:
            chem_df[col] = np.nan

    # === Concatenate CBC and Chemistry rows ===
    df = pd.concat([df, chem_df], ignore_index=True)

    # === Debugging prints after concatenation ===
    print(f"Combined rows after concat: {len(df)}")
    print(f"Source counts:\n{df['source'].value_counts()}")

    # === Merge mortality ===
    EOLS_df.columns = EOLS_df.columns.str.strip().str.lower()
    df = df.merge(EOLS_df[['dog_id', 'eol_death_date']], on='dog_id', how='left')
    df['eol_death_date'] = pd.to_datetime(df['eol_death_date'], errors='coerce')
    df['death_age'] = (df['eol_death_date'] - df['estimated_dob']).dt.total_seconds() / (365.25 * 24 * 3600)
    df['death_age'] = df.groupby('dog_id')['death_age'].transform(
        lambda x: [v if i == len(x)-1 else np.nan for i, v in enumerate(x)]
    )
    # === Determine sex ===
    col = 'sex_class_at_hles'
    ov_df[col] = ov_df[col].astype(str).str.lower().str.strip()
    ov_df['Sex'] = ov_df[col].apply(lambda x: 'F' if 'f' in x else ('M'))
    df = df.merge(ov_df[['dog_id', 'Sex']], on='dog_id', how='left')

    # === Clean up ===
    df = df.drop(columns=['age_date', 'dap_sample_id', 'estimated_dob', 'eol_death_date', 'source'])
    df = df.dropna(subset=['Age'])

    # Merge environmental factors
    env_df.columns = env_df.columns.str.strip().str.lower()

    # Convert date columns to datetime
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    env_df['address_year'] = pd.to_datetime(env_df['address_year'], format='%Y', errors='coerce').dt.year

    # Get the year from df['sample_date']
    df['sample_year'] = df['sample_date'].dt.year
    
    # --- Pre-aggregate env_df once ---
    env_avg = (
        env_df.groupby(['dog_id', 'address_year'], as_index=False)
        .agg({
            'cv_median_income': 'mean',
            'cv_pct_less_than_ba_degree': 'mean',
            'cv_disadvantage_index': 'mean',
            'cv_pct_us_born': 'mean',
            'cv_population_density': 'mean'
        })
    )

    # Merge once with both columns
    df = df.merge(
        env_avg,
        left_on=['dog_id', 'sample_year'],
        right_on=['dog_id', 'address_year'],
        how='left'
    )

    # Rename columns
    df = df.rename(columns={
        'cv_median_income': 'income',
        'cv_pct_less_than_ba_degree': 'education',
        'cv_disadvantage_index': 'disadv_idx',
        'cv_pct_us_born': 'immigration',
        'cv_population_density': 'pop_density'
    })

    # Drop helper columns
    df = df.drop(columns=['address_year', 'sample_year'])


    # Remove 'Age' and reinsert it near the end (just before last 4 cols)
    cols = list(df.columns)
    cols.remove('Age')
    cols.insert(len(df.columns), 'Age')
    df = df[cols]
    cols = list(df.columns)
    cols.remove('sample_date')
    cols.insert(len(df.columns), 'sample_date')
    df = df[cols]
    
    # === Rename biomarker columns ===
    df = df.rename(columns={
        'krt_cbc_total_wbcs': 'wbc',
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
    
    def merge_same_day_duplicates(df):
        # Identify rows that share the same dog_id and Age
        dup_groups = df.groupby(['dog_id', 'Age'])

        merged_rows = []
        seen_keys = set()

        for (dog_id, age), group in dup_groups:
            if len(group) == 1:
                merged_rows.append(group.iloc[0])
                continue

            # Initialize with the first row
            merged = group.iloc[0].copy()

            for _, row in group.iloc[1:].iterrows():
                for col in df.columns:
                    if pd.isna(merged[col]) and not pd.isna(row[col]):
                        merged[col] = row[col]

            merged_rows.append(merged)

        return pd.DataFrame(merged_rows)

    df = merge_same_day_duplicates(df)

    df = df.sort_values(by=['dog_id', 'Age'], axis=0)
    df = df.reset_index()
    df = df.drop(columns='index')
    biomarker_columns = df.columns[1:36]
    df = df[(df['Age'] >= 1) & (df['Age'] <= 15)]
    id_counts = df['dog_id'].value_counts()
    df = df[df['dog_id'].isin(id_counts[id_counts > 1].index)]
    
    return df, biomarker_columns

def encode_categorical(df):
    # Sex
    sex_map = {'F': 0, 'M': 1}

    df['Sex'] = df['Sex'].str.strip() #has some whitespaces

    df['Sex'] = df['Sex'].map(sex_map)
    
    # Split income into 3 brackets
    low_thresh = df['income'].quantile(1/3)
    high_thresh = df['income'].quantile(2/3)

    print(f"Low income ≤ {low_thresh:.2f}")
    print(f"Middle income > {low_thresh:.2f} and ≤ {high_thresh:.2f}")
    print(f"High income > {high_thresh:.2f}")

    
    def three_group(x):
        if x <= low_thresh:
            return 0
        elif x <= high_thresh:
            return 1
        else:
            return 2

    df['income'] = df['income'].apply(three_group)

    # Check counts per group
    print(df['income'].value_counts())
    
    
    # Split education into 2 brackets
    thresh = df['education'].quantile(1/2)

    print(f"Low university graduation rate ≤ {thresh:.2f}")
    print(f"High university graduation rate > {thresh:.2f}")

    
    def two_group(x):
        if x <= thresh:
            return 0
        else:
            return 1

    df['education'] = df['education'].apply(two_group)

    # Check counts per group
    print(df['education'].value_counts())
    
    
    # Split disadvantage index into 2 brackets
    thresh = df['disadv_idx'].quantile(1/2)

    print(f"Low disadvantage index ≤ {thresh:.2f}")
    print(f"High disadvantage index > {thresh:.2f}")

    df['disadv_idx'] = df['disadv_idx'].apply(two_group)

    # Check counts per group
    print(df['disadv_idx'].value_counts())
    
    
    # Split immigration into 2 brackets
    thresh = 50

    df['immigration'] = df['immigration'].apply(two_group)

    # Check counts per group
    print('0: 50% or more immigrants, 1: More than 50% US born')
    print(df['immigration'].value_counts())
    
    
    # Split population density into 2 brackets
    thresh = df['pop_density'].quantile(1/2)

    print(f"Rural ≤ {thresh:.2f} people/sq mi")
    print(f"Urban > {thresh:.2f} people/sq mi")

    df['pop_density'] = df['pop_density'].apply(two_group)

    # Check counts per group
    print(df['pop_density'].value_counts())
    return df

def process_data(kalman=False, pval_thresh=0.001):
    df, biomarker_columns = preprocessing()

    outlier_counts = {}
    removed_values = []  # store removed rows

    # Identify and remove statistical outliers
    for biomarker in biomarker_columns:
        values = df[biomarker]
        mean = values.mean()
        std = values.std()

        if std == 0:
            continue  # Skip constant columns

        z_scores = (values - mean) / std
        p_values = 2 * norm.sf(abs(z_scores))  # two-tailed

        mask = (p_values < pval_thresh)
        num_removed = mask.sum()
        outlier_counts[biomarker] = num_removed

        if num_removed > 0:
            removed = df.loc[mask, ['dog_id', 'Age', biomarker]].copy()
            removed['biomarker'] = biomarker
            removed_values.append(removed)

            #print(f"[{biomarker}] Removed {num_removed} outlier(s) with p < {pval_thresh}")
            #print(removed)

        df.loc[mask, biomarker] = np.nan

    #print("\n===== Outlier Counts Per Biomarker =====")
    total_removed = 0
    for biomarker, count in outlier_counts.items():
        #print(f"{biomarker}: {count} removed")
        total_removed += count

    #print(f"\n🎯 Total outliers removed across all biomarkers: {total_removed}")

    # Combine all removed rows into one dataframe for plotting
    removed_df = pd.concat(removed_values) if removed_values else pd.DataFrame(columns=['dog_id', 'Age', biomarker, 'biomarker'])

    # Continue preprocessing
    df = df.sort_values(by=['dog_id', 'Age']).reset_index(level=0, drop=True)
    #df = df.sort_values(by=['dog_id', 'Age'])
    df = imputation(df, biomarker_columns, imputation_type="mean")

    #df['dog_id'] = list(zip(df['dog_id'], df['Sex']))
    #df['dog_id'] = df.groupby('dog_id').ngroup() + 1

    df = log_transform(df, list(biomarker_columns))
    df = normalize_biomarkers(df, biomarker_columns)
    df = encode_categorical(df)

    return df, biomarker_columns

def condition_map(df):

    n = "101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 198 | 201 | 202 | 203 | 204 | 205 | 206 | 207 | 208 | 209 | 298 | 301 | 302 | 303 | 304 | 305 | 306 | 307 | 308 | 309 | 310 | 311 | 312 | 313 | 398 | 401 | 402 | 403 | 404 | 405 | 406 | 407 | 408 | 409 | 410 | 411 | 412 | 413 | 414 | 415 | 416 | 417 | 418 | 419 | 420 | 421 | 422 | 423 | 424 | 425 | 426 | 427 | 428 | 429 | 430 | 431 | 432 | 498 | 501 | 502 | 503 | 504 | 505 | 506 | 507 | 508 | 509 | 510 | 511 | 512 | 513 | 514 | 515 | 516 | 518 | 519 | 598 | 601 | 602 | 603 | 604 | 605 | 606 | 607 | 608 | 609 | 610 | 611 | 612 | 698 | 701 | 702 | 703 | 704 | 705 | 706 | 707 | 708 | 709 | 710 | 711 | 712 | 713 | 714 | 715 | 716 | 717 | 718 | 719 | 720 | 721 | 798 | 801 | 802 | 803 | 804 | 805 | 806 | 807 | 808 | 809 | 898 | 901 | 902 | 903 | 904 | 905 | 906 | 907 | 908 | 909 | 910 | 911 | 912 | 913 | 914 | 915 | 916 | 998 | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1010 | 1011 | 1012 | 1013 | 1014 | 1015 | 1016 | 1017 | 1098 | 1101 | 1102 | 1103 | 1104 | 1105 | 1106 | 1107 | 1108 | 1109 | 1110 | 1111 | 1112 | 1113 | 1114 | 1115 | 1116 | 1117 | 1118 | 1119 | 1198 | 1201 | 1202 | 1203 | 1204 | 1205 | 1206 | 1207 | 1208 | 1209 | 1210 | 1211 | 1212 | 1213 | 1214 | 1215 | 1216 | 1217 | 1298 | 1301 | 1302 | 1303 | 1304 | 1305 | 1306 | 1307 | 1308 | 1309 | 1310 | 1311 | 1312 | 1398 | 1401 | 1402 | 1403 | 1404 | 1405 | 1406 | 1407 | 1408 | 1409 | 1410 | 1411 | 1412 | 1413 | 1414 | 1415 | 1416 | 1498 | 1598 | 1601 | 1602 | 1603 | 1604 | 1605 | 1606 | 1607 | 1608 | 1609 | 1610 | 1611 | 1612 | 1613 | 1614 | 1615 | 1616 | 1617 | 1618 | 1619 | 1620 | 1621 | 1622 | 1623 | 1624 | 1625 | 1626 | 1627 | 1628 | 1629 | 1630 | 1631 | 1632 | 1633 | 1634 | 1635 | 1636 | 1637 | 1638 | 1639 | 1640 | 1698 | 1701 | 1702 | 1703 | 1704 | 1705 | 1706 | 1707 | 1708 | 1709 | 1710 | 1798 | 1801 | 1802 | 1803 | 1804 | 1805 | 1806 | 1807 | 1808 | 1809 | 1810 | 1811 | 1812 | 1813 | 1814 | 1815 | 1816 | 1817 | 1818 | 1898 | 1901 | 1902 | 1903 | 1904 | 1905 | 1906 | 1907 | 1908 | 1909 | 1910 | 1911 | 1912 | 1998"
    c = open(os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/conditions.txt')))
    c = c.read()
    
    # Split on '|'
    n = n.split(' | ')
    c = c.split(' | ')
    
    # Create DataFrame
    cond_df = pd.DataFrame({
        "ID": n,
        "Condition": c
    })
    cond_df.loc[0, 'Condition'] = 'Blindness'
    cond_df.loc[329, 'Condition'] = 'Other Immune'
    
    
    # Load health data
    health_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Health Dog.csv'))
    health_df = pd.read_csv(health_path)

    # Clean column names
    health_df.columns = health_df.columns.str.strip().str.lower()
    health_df = health_df.rename(columns={
        'afus_followup_year': 'folup_year',
        'afus_hs_new_condition': 'cond_id'
    })

    # Make sure main df has study_age
    df['sample_date'] = pd.to_datetime(df['sample_date'])
    sample_start = df['sample_date'].min()
    df['study_age'] = (df['sample_date'] - sample_start).dt.days / 365.25

    # Merge conditions into df
    merged = df.merge(
        health_df[['dog_id', 'folup_year', 'cond_id']],
        on="dog_id",
        how="left"
    )

    # Keep only conditions that were present at or before the study age
    merged = merged[merged['folup_year'] - 1 <= merged['study_age']]
    
    # Aggregate conditions into a list per row
    merged_cond = (
        merged.groupby(['dog_id', 'sample_date', 'study_age'])
              .agg({'cond_id': lambda x: list(x.dropna())})
              .reset_index()
    )
    merged_cond['cond_id'] = merged_cond['cond_id'].apply(
        lambda x: np.nan if (len(x) == 0 or all(str(i).strip() == "character(0)" for i in x)) else x
    )
    df = df.merge(merged_cond, on=['dog_id', 'sample_date', 'study_age'], how='left')
    #df = df.drop(columns='study_age')
    
    # Build mapping dictionary from cond_df
    cond_df["ID"] = pd.to_numeric(cond_df["ID"], errors="coerce")
    cond_map = cond_df.set_index("ID")["Condition"].to_dict()

    # Apply mapping to each list in df['cond_id']
    df['cond_name'] = df['cond_id'].apply(
        lambda lst: [cond_map.get(i, f"Unknown_{i}") for i in lst] if isinstance(lst, list) else np.nan
    )
    
    return df

def expand_conditions(df, n=10):
    before = len(df.columns)
    # Count number of unique dogs per condition
    condition_counts = (
        df.explode("cond_name")
          .dropna(subset=["cond_name"])
          .groupby("cond_name")["dog_id"].nunique()
    )
    
    # Keep only conditions seen in >= n dogs
    valid_conditions = condition_counts[condition_counts >= n].index
    
    # For each valid condition, add a True/False column
    for cond in valid_conditions:
        df[cond] = df["cond_name"].apply(
            lambda lst: cond in lst if isinstance(lst, list) else False
        )
    after = len(df.columns)
    print(f"{after - before} condition columns added")
    return df

def merge_cancer(df):
    cancer_path = os.path.abspath(os.path.join('/Volumes/Health Files', 'dog_data_files/Cancer Dog.csv'))
    cancer_df = pd.read_csv(cancer_path)
    cancer_df.columns = cancer_df.columns.str.strip().str.lower()
    cancer_df = cancer_df.rename(columns={
        'afus_hs_new_initial_diagnosis_year': 'diagnosis_year',
        'afus_hs_new_initial_diagnosis_month': 'diagnosis_month'
    })
    # --- Step 1: Compute diagnosis_date ---
    cancer_df = cancer_df[(cancer_df['diagnosis_year'] >= 2019) & (cancer_df['diagnosis_year'] <= 2026)]

    cancer_df['diagnosis_date'] = pd.to_datetime(
        cancer_df['diagnosis_year'].astype(int).astype(str) + '-' +
        cancer_df['diagnosis_month'].astype(int).astype(str) + '-01'
    )

    # --- Step 2: Identify location and type columns ---
    location_cols = [col for col in cancer_df.columns if col.startswith('afus_hs_new_cancer_locations_')]
    type_cols = [col for col in cancer_df.columns if col.startswith('afus_hs_new_cancer_types_')]

    # --- Step 3: Merge df and cancer_df on dog_id ---
    merged = df[['dog_id', 'sample_date']].merge(
        cancer_df[['dog_id', 'diagnosis_date'] + location_cols + type_cols],
        on='dog_id',
        how='left'
    )

    # --- Step 4: Keep only cancers diagnosed at or before sample_date ---
    merged = merged[merged['sample_date'] >= merged['diagnosis_date']]

    # --- Step 5: Collapse location and type columns into lists per dog/sample_date ---
    def extract_columns(row, prefix):
        return [
            col.replace(prefix, '').strip().lower()
            for col, val in row.items()
            if val and col.replace(prefix, '').strip().lower() != "other_description"
        ]
    merged['location'] = merged[location_cols].apply(lambda x: extract_columns(x, 'afus_hs_new_cancer_locations_'), axis=1)
    merged['cancer_type'] = merged[type_cols].apply(lambda x: extract_columns(x, 'afus_hs_new_cancer_types_'), axis=1)



    # --- Step 6: Aggregate per dog/sample_date ---
    agg_cols = ['dog_id', 'sample_date']
    final_cancer = merged.groupby(agg_cols).agg({
        'location': lambda x: list({loc for sublist in x for loc in sublist}) or np.nan,
        'cancer_type': lambda x: list({t for sublist in x for t in sublist}) or np.nan
    }).reset_index()

    # Locations
    mlb_loc = MultiLabelBinarizer()
    loc_encoded = pd.DataFrame(
        mlb_loc.fit_transform(final_cancer['location'].dropna()),
        columns=[f"loc_{val}" for val in mlb_loc.classes_],
        index=final_cancer['location'].dropna().index
    )

    # Cancer types
    mlb_type = MultiLabelBinarizer()
    type_encoded = pd.DataFrame(
        mlb_type.fit_transform(final_cancer['cancer_type'].dropna()),
        columns=[f"type_{val}" for val in mlb_type.classes_],
        index=final_cancer['cancer_type'].dropna().index
    )

    # Reindex so everything matches, filling missing with False
    loc_encoded = loc_encoded.reindex(final_cancer.index, fill_value=0).astype(bool)
    type_encoded = type_encoded.reindex(final_cancer.index, fill_value=0).astype(bool)
    
    # Concatenate back into final_cancer
    final_cancer = pd.concat([final_cancer, loc_encoded, type_encoded], axis=1)


    # --- Step 7: Merge back to df ---
    df = df.merge(final_cancer, on=['dog_id', 'sample_date'], how='left')
    
    # --- Step 8: After merging, force NaNs in indicator cols to False ---
    flag_cols = [c for c in df.columns if c.startswith('loc_') or c.startswith('type_')]
    if flag_cols:
        df[flag_cols] = df[flag_cols].fillna(False).astype(bool)

    return df


def create_base_next_df(df):
    # Sort by dog_id and Age
    df = df.sort_values(by=['dog_id', 'Age']).reset_index(drop=True)

    # List to hold final rows
    new_rows = []

    # Iterate over each dog_id
    for dog_id, group in df.groupby('dog_id'):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            # Skip dogs with only one row
            continue

        for i in range(len(group) - 1):
            base_row = group.iloc[i].copy()
            next_row = group.iloc[i + 1].copy()

            combined = {'dog_id': dog_id}  # keep dog_id as is

            # Add base columns with _base suffix (except dog_id)
            for col in group.columns:
                if col != 'dog_id':
                    combined[f'{col}_base'] = base_row[col]

            # Add next columns with _next suffix (except dog_id)
            for col in group.columns:
                if col != 'dog_id':
                    combined[f'{col}_next'] = next_row[col]

            new_rows.append(combined)

    # Convert list of dicts to DataFrame
    final_df = pd.DataFrame(new_rows)

    # Sort by dog_id and base Age
    final_df = final_df.sort_values(by=['dog_id', 'Age_base']).reset_index(drop=True)

    return final_df

def final_adjustments(df):
    # Determine status
    df['status'] = False
    death_rows = df[df['death_age_next'].isna() == False].index
    
    for row in death_rows:
        df['status'].iloc[row] = True

    return df

def add_death_column(df):
    df = df.copy()
    # make sure df is sorted by dog_id and Age_base
    df = df.sort_values(["dog_id", "Age_base"]).reset_index(drop=True)

    # initialize death column
    df["death"] = np.nan

    # loop over dogs
    for dog_id, group in df.groupby("dog_id"):
        idxs = group.index.to_list()

        for i, idx in enumerate(idxs):
            if i < len(idxs) - 1:
                # has a next row
                age_base = df.loc[idx, "Age_next"]
                age_next = df.loc[idx+1, "Age_next"]
                df.loc[idx, "death"] = age_next - age_base
            else:
                # last row for this dog
                age_base = df.loc[idx, "Age_base"]
                death_age = df.loc[idx, "death_age_next"]

                if pd.notnull(death_age):
                    df.loc[idx, "death"] = death_age - age_base
                else:
                    # censor at 2023-12-31
                    censor_date = pd.to_datetime("2023-12-31")
                    sample_date = df.loc[idx, "sample_date_next"]
                    df.loc[idx, "death"] = (censor_date - sample_date).days / 365.25

    return df

# ===== RUN EVERYTHING WITH ONE FUNCTION ===== #

def socioecon_preprocess():
    # Preprocess
    df, biomarker_columns = process_data(kalman=False, pval_thresh=0.001)
    
    # Optional affects
    df = condition_map(df)
    df = expand_conditions(df, n=10)      # n determines how many dogs must have that condition for it to become a column 
    df = merge_cancer(df)
    
    # More needed functions (reconfigures df into base/next columns, adds status and death columns)
    df = create_base_next_df(df)
    df = final_adjustments(df)
    df = add_death_column(df)
    
    # Reindex IDs if you want
    df['dog_id'] = df.groupby('dog_id').ngroup() + 1
    return df

df = socioecon_preprocess()




