# 🩺 Hypertension Prediction Using NHANES Data (L-Cycle)

This project aims to predict hypertension (HTN) using the **National Health and Nutrition Examination Survey (NHANES)** L-Cycle data. We build a data-driven pipeline that includes data preparation, exploratory data analysis (EDA), machine learning modeling, evaluation, and reporting to identify factors associated with hypertension risk.

---

## 📁 Project Structure

```bash
hypertension-prediction-nhanes/
│
├── data/                        # Raw NHANES XPT data files
│   ├── BAX_L.xpt                # Blood pressure & cholesterol
│   ├── BMX_L.xpt                # Body measurements
│   └── BPXO_L.xpt               # Blood pressure examination
│
├── src/                         # Source code for pipeline
│   ├── phase-1.py               # End-to-end pipeline
│   ├── training_splitting.py    # Train/test split utilities
│   └── building_models.py       # Model training and evaluation
│
├── results/                     # Outputs and generated data
│   ├── figures/                 # Visualizations
│   │   ├── bmi_distribution.png
│   │   ├── correlation_matrix.png
│   │   ├── confusion_matrix.png
│   │   ├── bp_scatter.png
│   │   ├── systolic_bp_distribution.png
│   │   └── ... (other figures)
│   ├── logistic_regression_model_results.csv
│   ├── odds_ratios_confidence_intervals.csv
│   └── phase1_complete_nhanes_analysis.csv
│
├── reports/                     # Summary reports and diagrams
│   ├── 3C.pdf                   # Final report
│   └── workflow_diagram.png    # Visual pipeline diagram
│
├── requirements.txt             # List of required Python packages
└── README.md                    # Project overview and usage instructions

text

## 🚀 Getting Started  

### 1. Clone the repository  
```bash
git clone https://github.com/your-username/hypertension-prediction-nhanes.git
cd hypertension-prediction-nhanes
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the pipeline
bash
python src/phase-1.py
# 🔍 Exploring Individual Components

training_splitting.py - Data splitting

building_models.py - Model fitting and results generation

# 📊 Key Outputs
Output File	Description
logistic_regression_model_results.csv	Model performance metrics
odds_ratios_confidence_intervals.csv	OR & 95% CI for interpretability
figures/	Visualizations (EDA & model diagnostics)
3C.pdf	Final analytical report
# 📚 Data Source
All data used are sourced from the official CDC NHANES Portal.

# 🧠 Methodology Summary
Hypertension definition: Using SBP, DBP, and medication status

Primary model: Logistic regression (for interpretability)

Evaluation metrics: Accuracy, ROC-AUC, confusion matrix

Visualization tools: matplotlib, seaborn, pandas

# 🧑‍💻 Authors
Risha Reddy Mukkisa
