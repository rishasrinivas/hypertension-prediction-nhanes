import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# Load the cleaned dataset from Phase 1
df = pd.read_csv('phase1_complete_nhanes_analysis.csv')
print(f"Dataset loaded with shape: {df.shape}")

# Check available variables
print("\nAvailable variables:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Identify key variables for modeling

# Essential variables
required_vars = ['HYPERTENSIVE']
optional_vars = ['BPXOSY1', 'BPXODI1', 'BMXBMI', 'AGE', 'SEX']

# Check what's available
available_vars = [var for var in required_vars + optional_vars if var in df.columns]
print(f"Available variables for modeling: {available_vars}")

# Prepare data for modeling

# Select modeling variables
modeling_vars = ['HYPERTENSIVE']
if 'AGE' in df.columns:
    modeling_vars.append('AGE')
if 'BMXBMI' in df.columns:
    modeling_vars.append('BMXBMI')
if 'SEX' in df.columns:
    modeling_vars.append('SEX')

print(f"Selected variables: {modeling_vars}")

# Create modeling dataset
model_df = df[modeling_vars].copy()

# Handle missing values
print(f"Dataset shape before cleaning: {model_df.shape}")
model_df = model_df.dropna()
print(f"Dataset shape after removing missing values: {model_df.shape}")

# Check for sufficient variables
if len(modeling_vars) < 2:
    print("Insufficient variables for modeling")
    print("Available variables:", modeling_vars)
    exit()

# Define predictor variables (excluding outcome)
predictor_vars = [var for var in modeling_vars if var != 'HYPERTENSIVE']
print(f"Predictor variables: {predictor_vars}")
print(f"Outcome variable: HYPERTENSIVE")

# Split data into train and test sets
X = model_df[predictor_vars]
y = model_df['HYPERTENSIVE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training prevalence: {y_train.mean()*100:.1f}%")
print(f"Test prevalence: {y_test.mean()*100:.1f}%")

# Standardize continuous variables for sklearn models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with original column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
