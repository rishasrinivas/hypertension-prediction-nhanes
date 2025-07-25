# MODEL A: Age + BMI (if available)
model_a_vars = []
if 'AGE' in predictor_vars:
    model_a_vars.append('AGE')
if 'BMXBMI' in predictor_vars:
    model_a_vars.append('BMXBMI')

if len(model_a_vars) == 0:
    print("Cannot build Model A: No age or BMI variables available")
    model_a_vars = predictor_vars[:2]  # Fallback to first 2 available variables

print(f"Model A predictors: {model_a_vars}")

# MODEL B: Age + BMI + Sex + Race (if available)
model_b_vars = model_a_vars.copy()
if 'SEX' in predictor_vars:
    model_b_vars.append('SEX')
# Note: Race not available in current dataset

print(f"Model B predictors: {model_b_vars}")

# MODEL C: Age + BMI + Additional variables (use all available)
model_c_vars = predictor_vars.copy()
print(f"Model C predictors: {model_c_vars}")

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")

    return {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# Build Model A
print("\n=== BUILDING MODEL A: Age + BMI ===")
if len(model_a_vars) > 0:
    X_train_a = X_train_scaled[model_a_vars]
    X_test_a = X_test_scaled[model_a_vars]

    # Sklearn model for performance metrics
    model_a_sk = LogisticRegression(random_state=42, max_iter=1000)
    model_a_sk.fit(X_train_a, y_train)

    # Statsmodels model for detailed statistics
    X_train_a_sm = sm.add_constant(X_train[model_a_vars])
    model_a_sm = sm.Logit(y_train, X_train_a_sm).fit(disp=0)

    # Evaluate Model A
    results_a = evaluate_model(model_a_sk, X_test_a, y_test, "Model A")
    print(f"  Coefficients: {dict(zip(model_a_vars, model_a_sk.coef_[0]))}")
else:
    print("Skipping Model A due to insufficient variables")

# Build Model B
print("\n=== BUILDING MODEL B: Age + BMI + Sex ===")
if len(model_b_vars) > len(model_a_vars):
    X_train_b = X_train_scaled[model_b_vars]
    X_test_b = X_test_scaled[model_b_vars]

    # Sklearn model
    model_b_sk = LogisticRegression(random_state=42, max_iter=1000)
    model_b_sk.fit(X_train_b, y_train)

    # Statsmodels model
    X_train_b_sm = sm.add_constant(X_train[model_b_vars])
    model_b_sm = sm.Logit(y_train, X_train_b_sm).fit(disp=0)

    # Evaluate Model B
    results_b = evaluate_model(model_b_sk, X_test_b, y_test, "Model B")
    print(f"  Coefficients: {dict(zip(model_b_vars, model_b_sk.coef_[0]))}")
else:
    print("Model B same as Model A, skipping...")
    results_b = results_a
    model_b_sk = model_a_sk
    model_b_vars = model_a_vars

# Build Model C
print("\n=== BUILDING MODEL C: All Available Predictors ===")
X_train_c = X_train_scaled[model_c_vars]
X_test_c = X_test_scaled[model_c_vars]

# Sklearn model
model_c_sk = LogisticRegression(random_state=42, max_iter=1000)
model_c_sk.fit(X_train_c, y_train)

# Statsmodels model
X_train_c_sm = sm.add_constant(X_train[model_c_vars])
model_c_sm = sm.Logit(y_train, X_train_c_sm).fit(disp=0)

# Evaluate Model C
results_c = evaluate_model(model_c_sk, X_test_c, y_test, "Model C")
print(f"  Coefficients: {dict(zip(model_c_vars, model_c_sk.coef_[0]))}")

print("\n=== MODEL COMPARISON ===")

# Create comparison table
comparison_data = {
    'Model': ['Model A', 'Model B', 'Model C'],
    'Predictors': [', '.join(model_a_vars), ', '.join(model_b_vars), ', '.join(model_c_vars)],
    'Accuracy': [results_a['accuracy'], results_b['accuracy'], results_c['accuracy']],
    'AUC': [results_a['auc'], results_b['auc'], results_c['auc']]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Find best model based on AUC
best_model_idx = comparison_df['AUC'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\n Best Model: {best_model_name} (AUC: {comparison_df.loc[best_model_idx, 'AUC']:.4f})")

# Detailed statistics for the best model using statsmodels
print(f"\n=== DETAILED STATISTICS FOR {best_model_name} ===")

if best_model_name == 'Model A':
    print(model_a_sm.summary())
    best_model_sm = model_a_sm
    best_model_vars = model_a_vars
    X_test_best = X_test_scaled[model_a_vars]
    y_pred_best = results_a['y_pred']
    y_pred_proba_best = results_a['y_pred_proba']

elif best_model_name == 'Model B':
    print(model_b_sm.summary())
    best_model_sm = model_b_sm
    best_model_vars = model_b_vars
    X_test_best = X_test_scaled[model_b_vars]
    y_pred_best = results_b['y_pred']
    y_pred_proba_best = results_b['y_pred_proba']

else:  # Model C
    print(model_c_sm.summary())
    best_model_sm = model_c_sm
    best_model_vars = model_c_vars
    X_test_best = X_test_scaled[model_c_vars]
    y_pred_best = results_c['y_pred']
    y_pred_proba_best = results_c['y_pred_proba']

# Calculate odds ratios and confidence intervals

odds_ratios = np.exp(best_model_sm.params)
conf_intervals = np.exp(best_model_sm.conf_int())

or_df = pd.DataFrame({
    'Variable': odds_ratios.index,
    'Odds_Ratio': odds_ratios.values,
    'CI_Lower': conf_intervals[0].values,
    'CI_Upper': conf_intervals[1].values
})

# Format for better readability
or_df['OR (95% CI)'] = or_df.apply(
    lambda row: f"{row['Odds_Ratio']:.3f} ({row['CI_Lower']:.3f} - {row['CI_Upper']:.3f})",
    axis=1
)

print(or_df[['Variable', 'OR (95% CI)']].to_string(index=False))

# Model Diagnostics
print(f"\n=== MODEL DIAGNOSTICS ===")

# 1. Multicollinearity (VIF)
print("1. Multicollinearity Check (VIF):")
X_train_best_sm = sm.add_constant(X_train[best_model_vars])
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train_best_sm.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_best_sm.values, i)
                   for i in range(X_train_best_sm.shape[1])]
print(vif_data.to_string(index=False))

# 2. ROC Curve
print("\n2. ROC Curve Analysis:")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
auc_score = roc_auc_score(y_test, y_pred_proba_best)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {best_model_name}')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')


# 3. Confusion Matrix
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(f"Confusion Matrix:")
print(cm)
print(f"Sensitivity (Recall): {cm[1,1] / (cm[1,0] + cm[1,1]):.3f}")
print(f"Specificity: {cm[0,0] / (cm[0,0] + cm[0,1]):.3f}")
print(f"Precision: {cm[1,1] / (cm[0,1] + cm[1,1]):.3f}")

# 4. Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Save model results
print(f"\n=== SAVING MODEL RESULTS ===")

# Save best model coefficients and statistics
model_results = {
    'best_model': best_model_name,
    'predictors': best_model_vars,
    'coefficients': dict(zip(best_model_sm.params.index, best_model_sm.params.values)),
    'odds_ratios': dict(zip(odds_ratios.index, odds_ratios.values)),
    'p_values': dict(zip(best_model_sm.pvalues.index, best_model_sm.pvalues.values)),
    'performance': {
        'accuracy': accuracy_score(y_test, y_pred_best),
        'auc': auc_score
    }
}

# Save to CSV
results_summary = pd.DataFrame([model_results])
results_summary.to_csv('logistic_regression_model_results.csv', index=False)

# Save detailed odds ratios
or_df.to_csv('odds_ratios_confidence_intervals.csv', index=False)

print("Phase 2 Complete!")
print("Results saved to:")
print("  - logistic_regression_model_results.csv")
print("  - odds_ratios_confidence_intervals.csv")
print(f"  - Best model: {best_model_name}")
print(f"  - AUC: {auc_score:.4f}")
print(f"  - Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

# Final summary
print(f"\n=== FINAL MODEL SUMMARY ===")
print(f"Best Model: {best_model_name}")
print(f"Predictors: {', '.join(best_model_vars)}")
print(f"AUC: {auc_score:.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print("\nKey Findings:")
for idx, row in or_df.iterrows():
    if row['Variable'] != 'const':  # Skip intercept
        print(f"  - {row['Variable']}: OR = {row['Odds_Ratio']:.3f} (95% CI: {row['CI_Lower']:.3f} - {row['CI_Upper']:.3f})")
