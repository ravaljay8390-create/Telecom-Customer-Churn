# Telecom Customer Churn Prediction

A beginner-friendly ML capstone project that predicts whether a telecom customer will churn (leave the company), using classification models trained on the IBM Telco Customer Churn dataset.

## Folder structure

```
telco_churn/
├── notebook/
│   ├── Telco_Customer_Churn_Prediction.ipynb   <- Jupyter notebook (EDA + model training)
│   └── data/Telco-Customer-Churn.csv           <- dataset (also used by the notebook)
├── data/Telco-Customer-Churn.csv               <- copy of the dataset
├── models/                                     <- saved model files (created by the notebook)
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── columns.pkl
├── app/
│   └── app.py                                  <- Streamlit web app
├── docs/
│   └── Telecom_Customer_Churn_Prediction_Report.docx   <- project report
└── requirements.txt
```

## How to run it

### 1. Install the requirements
```bash
pip install -r requirements.txt
```

### 2. Run the notebook (optional - the model is already trained and saved)
Open `notebook/Telco_Customer_Churn_Prediction.ipynb` in Jupyter and run all cells top to bottom.
This will regenerate the model files inside `models/`.

```bash
jupyter notebook notebook/Telco_Customer_Churn_Prediction.ipynb
```

### 3. Run the Streamlit app
Run this command from the `telco_churn` project root folder (so it can find the `models/` folder):

```bash
streamlit run app/app.py
```

This will open a web page in your browser where you can enter a customer's details and get a churn prediction.

## Notes
- The model is a Logistic Regression classifier, chosen for its good balance of accuracy (~80%) and simplicity/interpretability.
- Dataset source: IBM Sample Data Sets - Telco Customer Churn (Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
