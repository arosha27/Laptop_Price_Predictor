
# 💻 Laptop Price Predictor

An end-to-end Machine Learning application that predicts laptop prices in **RS** based on hardware specifications. This project follows a structured data science workflow, from raw data cleaning to feature engineering and deployment.

## 🚀 Features

* **Price Estimation:** Precise predictions using an ensemble model.
* **Feature Engineering:** Automated **PPI (Pixels Per Inch)** calculation.
* **Two Data Versions:** Compare model performance on basic vs. engineered datasets.
* **Model Insights:** Visual feature importance directly in the app.

## 📂 Project Workflow & Structure

The project is organized into modular directories for better scalability and deployment:

```text
├── Data/
│   ├── Raw/
│   │   └── laptop_data.csv             # Original raw dataset
│   ├── Processed/
│   │   ├── v1_preprocessed.csv         # Cleaned data (Version 1)
│   │   └── v2_engineered.csv           # Data with engineered features (Version 2)
│   └── Assets/                         # Visualization images and plots
│
├── Experiment/
│   ├── data_preprocessing.ipynb        # Step-by-step cleaning & formatting
│   ├── data_visiualization.ipynb       # EDA and statistical analysis
│   └── model_training.ipynb            # Training 4 models & evaluation
│
├── TrainedModels/
│   ├── laptop_price_best_model.pkl     # Best performing model (Extra Trees)
│   └── model_comparison_results.csv    # Accuracy/Error metrics for all models
│
└── FrontEnd/
    └── app.py                          # Streamlit user interface

```

## 🛠️ Tech Stack

* **Analysis:** Pandas, NumPy, Matplotlib, Seaborn
* **ML Models:** Linear Regression, Random Forest, Gradient Boosting, Extra Trees
* **Deployment:** Streamlit
* **Persistence:** Pickle

## 📈 Model Performance Summary

Based on the results in `TrainedModels/model_comparison_results.csv`:

* **Best Model:** Extra Trees Regressor
* **Accuracy (R² Score):** ~89%
* **Feature Impact:** CPU Brand, RAM, and SSD were found to be the top 3 price drivers.

## 💻 Installation & Usage

### 1. Setup Environment

```bash
git clone https://github.com/yourusername/laptop-price-predictor.git
cd laptop-price-predictor
pip install -r requirements.txt

```

### 2. Run the Application

Navigate to the `FrontEnd` directory and launch the app:

```bash
streamlit run FrontEnd/app.py

```

*Developed for professional portfolio use. If you find this useful, give it a ⭐!*

---
