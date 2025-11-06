
Diabetes Prediction Project

 🩺 Overview

This project aims to predict **the likelihood of diabetes** based on patients’ medical and demographic data using machine learning models.
The final model is deployed through an interactive **Streamlit web app**.


 🎯 Objectives

* Data cleaning and preprocessing
* Feature selection and transformation
* Model training and evaluation
* Hyperparameter tuning
* Model saving for deployment
* Building an interactive Streamlit dashboard


 📊 Dataset Description

The dataset includes the following health indicators:

| Feature  | Description                                     |
| -------- | ----------------------------------------------- |
| RIDAGEYR | Age                                             |
| RIAGENDR | Gender                                          |
| RIDRETH1 | Ethnicity                                       |
| BMXBMI   | Body Mass Index                                 |
| LBXGLU   | Glucose Level                                   |
| LBDHDD   | HDL Cholesterol                                 |
| LBXTR    | Triglycerides                                   |
| BPXOSY1  | Systolic Blood Pressure                         |
| BPXODI1  | Diastolic Blood Pressure                        |
| diabetes | Target variable (0 = No Diabetes, 1 = Diabetes) |


 ⚙️ Data Preprocessing Steps

1. Selected the most relevant health features
2. Converted categorical columns into dummy variables
3. Standardized numeric columns using **StandardScaler**
4. Split data into training and testing sets using **train_test_split** (with stratification)


 🤖 Models Trained

The following models were trained and evaluated:

| Model               | Description                   |
| ------------------- | ----------------------------- |
| Logistic Regression | Baseline linear model         |
| Random Forest       | Ensemble model using bagging  |
| XGBoost             | Gradient boosting-based model |

Performance metrics included **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**.


 🔍 Hyperparameter Tuning

The **Random Forest** model was tuned using **RandomizedSearchCV** with the following parameters:

* `n_estimators`: number of trees
* `max_depth`: tree depth
* `min_samples_split`: minimum split samples
* `min_samples_leaf`: minimum leaf samples

The best model achieved the highest **ROC-AUC** score and was selected for deployment.


 💾 Model Saving

Both the trained model and the scaler were saved using `joblib`:


saved_models/
├── best_rf_model.joblib
└── scaler.joblib


 💻 Streamlit App

An interactive Streamlit app was created for real-time diabetes prediction.
Users input patient data (age, BMI, blood pressure, glucose, HDL, triglycerides, gender), and the model outputs the **probability of having diabetes**.

 🚀 How to Run the App

 1️⃣ Clone the Repository

bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction


 2️⃣ Install Dependencies

pip install -r requirements.txt


3️⃣ Run the App


streamlit run diabetes_dashboard.py

 📁 Project Structure

diabetes-prediction/
│
├── diabetes_dashboard.py        # Streamlit app
├── saved_models/
│   ├── best_rf_model.joblib     # Tuned Random Forest model
│   └── scaler.joblib            # StandardScaler
├── notebook.ipynb               # Jupyter notebook (data prep, training)
├── requirements.txt             # Dependencies
└── README.md                    # Documentation

 🧰 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib, Seaborn



 📈 Results

* Tuned **Random Forest** achieved the best overall performance.
* **ROC-AUC** and **F1** scores showed high predictive accuracy.
* Streamlit app allows users to test predictions easily.

 🔮 Future Work

* Add additional clinical indicators (e.g., insulin, cholesterol ratio)
* Include SHAP or LIME for model interpretability
* Deploy app publicly using Streamlit Cloud or Render


 👩‍💻 Developed By

jana mohamed abozeid
(A machine learning project to predict diabetes using health indicators and deploy it via Streamlit.)

