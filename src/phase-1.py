import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Using 3 NHANES L-Cycle Datasets:")
print("1. BAX_L.XPT (Balance)")
print("2. BPXO_L.XPT (Blood Pressure)")
print("3. BMX_L.XPT (Body Measures)")

# Load all three datasets
bax_df = pd.read_sas('BAX_L.xpt')       # Balance
bpx_df = pd.read_sas('BPXO_L.xpt')      # Blood Pressure
bmx_df = pd.read_sas('BMX_L.xpt')       # Body Measures

print(f"Balance dataset shape: {bax_df.shape}")
print(f"Blood Pressure dataset shape: {bpx_df.shape}")
print(f"Body Measures dataset shape: {bmx_df.shape}")

# Merge all three datasets on SEQN
merged_df = bax_df.merge(bpx_df, on='SEQN', how='inner') \
                  .merge(bmx_df, on='SEQN', how='inner')
print(f"Merged dataset shape: {merged_df.shape}")

# Check what variables we have available
print("All columns in merged dataset:")
all_columns = sorted(merged_df.columns.tolist())
for i, col in enumerate(all_columns):
    print(f"{i+1:2d}. {col}")

# Look for essential variables with actual L-cycle names

# Blood Pressure variables (L-cycle uses BPXOSY1, BPXODI1)
bp_variables = [col for col in merged_df.columns if 'BPXO' in col and ('SY' in col or 'DI' in col)]
print(f"Available BP variables: {bp_variables}")

# Demographics - check if any demographic variables are in the merged dataset
demo_candidates = ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3']
available_demo = [col for col in demo_candidates if col in merged_df.columns]
print(f"Available demographic variables: {available_demo}")

# Body measures
body_measures = [col for col in merged_df.columns if 'BMX' in col]
print(f"Available body measure variables: {body_measures}")

# Balance variables
balance_vars = [col for col in merged_df.columns if 'BAX' in col or 'BAQ' in col]
print(f"Available balance variables (sample): {balance_vars[:10]}...")

# Proceed with analysis using actual variable names

# Define hypertension using the correct BP variable names
if 'BPXOSY1' in merged_df.columns and 'BPXODI1' in merged_df.columns:
    print("✅ Found primary BP variables: BPXOSY1, BPXODI1")
    merged_df['HYPERTENSIVE'] = (
        (merged_df['BPXOSY1'] >= 130) |
        (merged_df['BPXODI1'] >= 80)
    ).astype(int)

    print(f"Hypertension prevalence: {merged_df['HYPERTENSIVE'].mean()*100:.1f}%")

    # Check for additional demographic variables
    available_demo_vars = []

    # Look for age variable (might have different name)
    age_vars = [col for col in merged_df.columns if 'AGE' in col.upper()]
    if age_vars:
        age_var = age_vars[0]  # Use first age variable found
        print(f"✅ Found age variable: {age_var}")
        merged_df['AGE'] = merged_df[age_var]
        available_demo_vars.append('AGE')
        # Remove minors
        merged_df = merged_df[merged_df['AGE'] >= 18]

    # Look for gender variable
    gender_vars = [col for col in merged_df.columns if 'GEND' in col.upper() or 'SEX' in col.upper()]
    if gender_vars:
        gender_var = gender_vars[0]
        print(f"✅ Found gender variable: {gender_var}")
        merged_df['SEX'] = merged_df[gender_var]
        available_demo_vars.append('SEX')

    # Clean data
    
    # Handle missing values in key variables
    key_vars = ['BPXOSY1', 'BPXODI1', 'HYPERTENSIVE']
    if 'AGE' in merged_df.columns:
        key_vars.append('AGE')

    merged_df = merged_df.dropna(subset=key_vars)

    # Remove extreme BP values
    merged_df = merged_df[
        (merged_df['BPXOSY1'] >= 60) & (merged_df['BPXOSY1'] <= 250) &
        (merged_df['BPXODI1'] >= 30) & (merged_df['BPXODI1'] <= 150)
    ]

    # BMI cleaning if available
    if 'BMXBMI' in merged_df.columns:
        merged_df = merged_df[(merged_df['BMXBMI'] >= 10) & (merged_df['BMXBMI'] <= 80)]

    # Age cleaning if available
    if 'AGE' in merged_df.columns:
        merged_df = merged_df[(merged_df['AGE'] >= 18) & (merged_df['AGE'] <= 100)]

    print(f"Dataset shape after cleaning: {merged_df.shape}")

    # Exploratory Analysis

    # Descriptive statistics for available variables
    analysis_vars = ['BPXOSY1', 'BPXODI1']
    if 'BMXBMI' in merged_df.columns:
        analysis_vars.append('BMXBMI')
    if 'AGE' in merged_df.columns:
        analysis_vars.append('AGE')

    print("Descriptive Statistics:")
    print(merged_df[analysis_vars].describe())

    # Hypertension analysis
    print(f"\nHypertension Rate: {merged_df['HYPERTENSIVE'].mean()*100:.1f}%")

    # Analysis by available demographics
    if 'SEX' in merged_df.columns:
        print("\nHypertension by Gender:")
        gender_analysis = merged_df.groupby('SEX')['HYPERTENSIVE'].agg(['count', 'mean'])
        gender_analysis['mean'] = gender_analysis['mean'] * 100
        print(gender_analysis)

    if 'AGE' in merged_df.columns:
        # Age groups
        merged_df['AGE_GROUP'] = pd.cut(merged_df['AGE'],
                                       bins=[18, 30, 45, 65, 100],
                                       labels=['18-29', '30-44', '45-64', '65+'])
        print("\nHypertension by Age Group:")
        age_analysis = pd.crosstab(merged_df['AGE_GROUP'], merged_df['HYPERTENSIVE'], normalize='index') * 100
        print(age_analysis)

    if 'BMXBMI' in merged_df.columns:
        # BMI categories
        merged_df['BMI_CATEGORY'] = pd.cut(merged_df['BMXBMI'],
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        print("\nHypertension by BMI Category:")
        bmi_analysis = pd.crosstab(merged_df['BMI_CATEGORY'], merged_df['HYPERTENSIVE'], normalize='index') * 100
        print(bmi_analysis)

    # Visualizations

    # 1. Systolic BP distribution
    plt.figure(figsize=(8, 6))
    plt.hist(merged_df['BPXOSY1'], bins=50, alpha=0.7, color='skyblue')
    plt.title('Systolic Blood Pressure Distribution')
    plt.xlabel('SBP (mmHg)')
    plt.savefig('systolic_bp_distribution.png')
    plt.show()

    # 2. Diastolic BP distribution
    plt.figure(figsize=(8, 6))
    plt.hist(merged_df['BPXODI1'], bins=50, alpha=0.7, color='lightcoral')
    plt.title('Diastolic Blood Pressure Distribution')
    plt.xlabel('DBP (mmHg)')
    plt.savefig('diastolic_bp_distribution.png')
    plt.show()

    # 3. Age distribution (if available)
    if 'AGE' in merged_df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(merged_df['AGE'], bins=50, alpha=0.7, color='lightgreen')
        plt.title('Age Distribution')
        plt.xlabel('Age (years)')
        plt.savefig('age_distribution.png')
        plt.show()

    # 4. BMI distribution (if available)
    if 'BMXBMI' in merged_df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(merged_df['BMXBMI'], bins=50, alpha=0.7, color='gold')
        plt.title('BMI Distribution')
        plt.xlabel('BMI')
        plt.savefig('bmi_distribution.png')
        plt.show()

    # 5. BP scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(merged_df['BPXOSY1'], merged_df['BPXODI1'],
                             c=merged_df['HYPERTENSIVE'], alpha=0.6, cmap='coolwarm')
    plt.xlabel('Systolic BP')
    plt.ylabel('Diastolic BP')
    plt.title('BP Scatter Plot by Hypertension Status')
    plt.colorbar(scatter)
    plt.savefig('bp_scatter.png')
    plt.show()

    # 6. Hypertension prevalence visualization
    plt.figure(figsize=(8, 6))
    htn_counts = merged_df['HYPERTENSIVE'].value_counts()
    plt.bar(['No Hypertension', 'Hypertension'], htn_counts.values,
                color=['skyblue', 'salmon'])
    plt.title('Hypertension Prevalence')
    plt.ylabel('Count')
    plt.savefig('hypertension_prevalence.png')
    plt.show()

    # Hypertension by age group
    if 'AGE_GROUP' in merged_df.columns:
        plt.figure(figsize=(8, 6))
        age_htn = merged_df.groupby('AGE_GROUP')['HYPERTENSIVE'].mean() * 100
        plt.bar(age_htn.index.astype(str), age_htn.values, color='lightgreen')
        plt.title('Hypertension Prevalence by Age Group')
        plt.ylabel('Prevalence (%)')
        plt.xticks(rotation=45)
        plt.savefig('hypertension_by_age.png')
        plt.show()

    # Hypertension by gender
    if 'SEX' in merged_df.columns:
        plt.figure(figsize=(8, 6))
        sex_htn = merged_df.groupby('SEX')['HYPERTENSIVE'].mean() * 100
        plt.bar(['Category 0', 'Category 1'], sex_htn.values, color='orange')
        plt.title('Hypertension Prevalence by Gender')
        plt.ylabel('Prevalence (%)')
        plt.savefig('hypertension_by_gender.png')
        plt.show()


    # Correlation analysis for available variables
    corr_vars = ['BPXOSY1', 'BPXODI1']
    if 'AGE' in merged_df.columns:
        corr_vars.append('AGE')
    if 'BMXBMI' in merged_df.columns:
        corr_vars.append('BMXBMI')
    if 'HYPERTENSIVE' in merged_df.columns:
        corr_vars.append('HYPERTENSIVE')

    if len(corr_vars) > 1:
        corr_matrix = merged_df[corr_vars].corr()
        print(corr_matrix.round(3))

        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('Correlation Matrix')
        plt.savefig('correlation_matrix.png')
        plt.show()
    else:
        print("Insufficient variables for correlation analysis")

    # Missing data analysis
    missing_summary = merged_df.isnull().sum()
    missing_percent = (missing_summary / len(merged_df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_summary,
        'Missing Percent': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percent', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.head(10))  # Show top 10
    else:
        print("No missing data in key variables")

    # Outlier detection for key variables
    key_numerical_vars = ['BPXOSY1', 'BPXODI1']
    if 'AGE' in merged_df.columns:
        key_numerical_vars.append('AGE')
    if 'BMXBMI' in merged_df.columns:
        key_numerical_vars.append('BMXBMI')

    for var in key_numerical_vars:
        Q1 = merged_df[var].quantile(0.25)
        Q3 = merged_df[var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = merged_df[(merged_df[var] < lower_bound) | (merged_df[var] > upper_bound)]
        outlier_percent = len(outliers) / len(merged_df) * 100
        print(f"{var}: {len(outliers)} outliers ({outlier_percent:.1f}%) - Bounds: {lower_bound:.1f} to {upper_bound:.1f}")

    # Distribution normality tests
    for var in key_numerical_vars[:3]:  # Test first 3 variables
        # Sample for large datasets
        sample_data = merged_df[var].sample(n=min(5000, len(merged_df)), random_state=42)
        stat, p_value = stats.shapiro(sample_data)
        print(f"{var}: Shapiro-Wilk p-value = {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Not Normal'})")

    # Save final dataset
    final_columns = ['SEQN', 'BPXOSY1', 'BPXODI1', 'HYPERTENSIVE']
    if 'BMXBMI' in merged_df.columns:
        final_columns.append('BMXBMI')
    if 'AGE' in merged_df.columns:
        final_columns.append('AGE')
    if 'SEX' in merged_df.columns:
        final_columns.append('SEX')

    # Add some balance variables that might be relevant
    balance_relevant = ['BAXMSTAT', 'BAXRXNC', 'BAXRXND', 'BAX5STAT']  # Common balance variables
    balance_to_add = [col for col in balance_relevant if col in merged_df.columns]
    final_columns.extend(balance_to_add)

    # Remove duplicates and create final dataset
    final_columns = list(set(final_columns))
    final_df = merged_df[final_columns].copy()

    # Final dataset info
    print(f"INAL DATASET SUMMARY")
    print(f"Variables included: {final_columns}")
    print(f"Dataset shape: {final_df.shape}")
    print(f"Missing values in final dataset:")
    print(final_df.isnull().sum())

    final_df.to_csv('phase1_complete_nhanes_analysis.csv', index=False)
    print(f"\nPHASE 1 COMPLETE!")
    print(f"Final dataset saved as: 'phase1_complete_nhanes_analysis.csv'")

    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total participants: {len(final_df)}")
    print(f"Hypertension prevalence: {final_df['HYPERTENSIVE'].mean()*100:.1f}%")
    if 'AGE' in final_df.columns:
        print(f"Age range: {final_df['AGE'].min():.0f} - {final_df['AGE'].max():.0f} years")
    if 'BMXBMI' in final_df.columns:
        print(f"BMI range: {final_df['BMXBMI'].min():.1f} - {final_df['BMXBMI'].max():.1f}")

else:
    print("CRITICAL: Could not find required blood pressure variables")
    print("Available BP variables:", bp_variables)
    print("Please verify your dataset structure")
