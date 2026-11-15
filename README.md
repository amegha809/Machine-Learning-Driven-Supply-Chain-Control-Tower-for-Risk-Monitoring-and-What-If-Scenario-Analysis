# Machine Learning-Driven Supply Chain Control Tower

## Project Overview

The **Machine Learning-Driven Supply Chain Control Tower** is a Python-based desktop application developed to monitor and predict late delivery risks in supply chain operations.

The system uses historical supply chain data and machine learning models to identify potentially delayed orders. It also provides explanations for predictions and allows users to compare different operational scenarios through What-If Analysis.

The main aim is to help users understand delivery risks and make better decisions using data-driven insights.

## Key Features

- **Data Preprocessing and Analysis:** Processes historical supply chain data for machine learning.
- **Machine Learning Model Comparison:** Implements Random Forest, Logistic Regression, and Decision Tree classifiers.
- **Late Delivery Risk Prediction:** Predicts whether an order is likely to experience late delivery.
- **Risk Monitoring Dashboard:** Displays supply chain information and identifies high-risk orders.
- **Explainable AI (SHAP):** Helps users understand the factors influencing individual risk predictions.
- **Feature Importance:** Shows which features have a greater influence on model predictions.
- **What-If Scenario Analysis:** Compares risk probabilities when shipping modes or scheduled shipping days are changed.
- **Desktop GUI:** Provides an interactive interface developed using CustomTkinter.

## Technologies Used

- Python
- CustomTkinter and Tkinter
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SHAP
- Joblib
- Git and GitHub

## Machine Learning Models

The project implements and compares three classification algorithms:

1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier

### Model Evaluation

The models are evaluated using accuracy, precision, recall, and F1-score.

| Metric | Random Forest | Logistic Regression | Decision Tree |
|---|---:|---:|---:|
| Accuracy | 65.12% | 69.68% | 69.45% |
| Precision | 69.56% | 84.93% | 83.49% |
| Recall | 64.72% | 54.36% | 55.20% |
| F1-Score | 67.05% | 66.29% | 66.46% |

Random Forest provides the highest recall and F1-score among the three evaluated models.

## Project Structure

```text
Supply-Chain-Control-Tower/
│
├── app1.py
├── model_training.py
├── predict_risk.py
├── high_risk.py
├── explain_risk.py
├── explain_single_order.py
├── what_if_analysis.py
├── datainspect.py
├── check_model_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── model_comparison.csv
│   ├── model_comparison_chart.png
│   └── shap_summary.png
│
├── processed/
│   └── supply_chain_processed.csv
│
└── reports/
    ├── high_risk_orders.csv
    ├── single_order_explanation.csv
    ├── interactive_order_explanation.csv
    ├── sample_risk_prediction.csv
    ├── shipping_scenario_comparison.csv
    └── other analysis reports and charts
```

## Dataset

The project uses the **DataCo Smart Supply Chain Dataset**, which contains historical order and shipment information.

The original dataset is not included in this repository. The processed dataset used by the project is available in the `processed` folder.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amegha809/Machine-Learning-Driven-Supply-Chain-Control-Tower-for-Risk-Monitoring-and-What-If-Scenario-Analysis.git
```

### 2. Open the Project Folder

```bash
cd Machine-Learning-Driven-Supply-Chain-Control-Tower-for-Risk-Monitoring-and-What-If-Scenario-Analysis
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Models

```bash
python model_training.py
```

This step generates the trained model files required by the application.

### 5. Run the Desktop Application

```bash
python app1.py
```

**Note:** The trained `.pkl` model files are excluded from this GitHub repository because of their large file sizes. Run the training script to generate the required model files locally.

## Main Project Modules

| File | Purpose |
|---|---|
| `app1.py` | Main desktop application and dashboard |
| `model_training.py` | Trains and compares machine learning models |
| `predict_risk.py` | Performs late delivery risk prediction |
| `high_risk.py` | Identifies and prepares high-risk orders |
| `explain_risk.py` | Provides model explainability and feature analysis |
| `explain_single_order.py` | Explains individual order predictions |
| `what_if_analysis.py` | Performs shipping scenario analysis |
| `datainspect.py` | Inspects and analyzes dataset information |
| `check_model_pipeline.py` | Checks the model preprocessing pipeline |

## Future Scope

- Integration with live supply chain data.
- Improved supplier risk analysis.
- More advanced prediction techniques.
- Additional operational scenario simulations.
- Further improvement in model performance and dashboard functionality.

## Conclusion

This project demonstrates how machine learning and explainable AI can be used to support supply chain risk monitoring. By combining risk prediction, high-risk order identification, SHAP explanations, and What-If Analysis in a desktop application, it provides a practical decision-support system for understanding potential delivery problems.

## Repository

[GitHub Repository](https://github.com/amegha809/Machine-Learning-Driven-Supply-Chain-Control-Tower-for-Risk-Monitoring-and-What-If-Scenario-Analysis)
