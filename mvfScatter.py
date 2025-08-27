import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# === Folder path ===
output_dir = os.path.expanduser('~/Downloads/dolphins-master/results/mvf_biomarker_plots(2-98%)')

# === Create folder if missing ===
os.makedirs(output_dir, exist_ok=True)



def mvf(Biomarker):
    fig, ax = plt.subplots()

    maleAge, maleBio = [], []
    femaleAge, femaleBio = [], []

    # Filter out extreme outliers
    #lower_percentile = df[Biomarker].quantile(0.02)
    #upper_percentile = df[Biomarker].quantile(0.98)
    
    #df_filtered = df[
        #(df[Biomarker] >= lower_percentile) &
        #(df[Biomarker] <= upper_percentile)
    #]
    df_filtered = df_valid.copy()
    for idx, sex in enumerate(df_filtered['Sex']):
        biomarker_val = df_filtered[Biomarker].iloc[idx]
        age_val = df_filtered['Age'].iloc[idx]

        if np.isnan(biomarker_val) or np.isnan(age_val):
            continue

        if sex == 'F' or sex == 0:
            femaleBio.append(biomarker_val)
            femaleAge.append(age_val)
        else:
            maleBio.append(biomarker_val)
            maleAge.append(age_val)

    # Plot scatter
    ax.scatter(maleAge, maleBio, color='C0', alpha = 0.5, label='Male')
    ax.scatter(femaleAge, femaleBio, color='C6', alpha = 0.5, label='Female')


    # LOWESS smoothing
    m_lowess_result = sm.nonparametric.lowess(maleBio, maleAge, frac=0.7, return_sorted=True)
    m_x_lowess, m_y_lowess = m_lowess_result[:, 0], m_lowess_result[:, 1]
    
    f_lowess_result = sm.nonparametric.lowess(femaleBio, femaleAge, frac=0.7, return_sorted=True)
    f_x_lowess, f_y_lowess = f_lowess_result[:, 0], f_lowess_result[:, 1]
    
    ax.plot(m_x_lowess, m_y_lowess, color='black', linestyle='-', linewidth=2.5, label='LOWESS Smoother (Male)')
    ax.plot(f_x_lowess, f_y_lowess, color='red', linestyle='-', linewidth=2.5, label='LOWESS Smoother (Female)')

    # Male linear fit
    
    if len(maleAge) >= 2:
        m_slope, m_intercept = np.polyfit(maleAge, maleBio, 1)
        m_x_fit = np.linspace(np.min(maleAge), np.max(maleAge), 100)
        m_y_fit_linear = m_slope * m_x_fit + m_intercept
        ax.plot(m_x_fit, m_y_fit_linear, color='black', linestyle='--', linewidth=2,
                label=f'Male Fit: y = {m_slope:.5f}x + {m_intercept:.5f}')
    else:
        print("Not enough male data points for polyfit.")

    # Female linear fit
    if len(femaleAge) >= 2:
        f_slope, f_intercept = np.polyfit(femaleAge, femaleBio, 1)
        f_x_fit = np.linspace(np.min(femaleAge), np.max(femaleAge), 100)
        f_y_fit_linear = f_slope * f_x_fit + f_intercept
        ax.plot(f_x_fit, f_y_fit_linear, color='red', linestyle='--', linewidth=2,
                label=f'Female Fit: y = {f_slope:.5f}x + {f_intercept:.5f}')
    else:
        print("Not enough female data points for polyfit.")
    
    
    plt.title(Biomarker + ' Over Age (2%-98%)')
    plt.xlabel('Age')
    plt.ylabel(Biomarker)
    plt.legend()
    plt.show()
    
    # === Save Plot ===
    #filename = os.path.join(output_dir, f"{biomarker}_mvf_over_age.png")
    #plt.savefig(filename, dpi=300)
    #print(f"Saved plot to {filename}")

    plt.clf()
    del fig

# Run
for biomarker in biomarker_columns[1:2]:
  mvf(Biomarker=biomarker)


