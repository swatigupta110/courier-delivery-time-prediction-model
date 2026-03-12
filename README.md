# Courier Delivery Time Estimator
A **Machine Learning–based regression model** that predicts the **estimated courier delivery time (in days)** between two locations. This project helps logistics and e-commerce platforms **forecast delivery timelines**, improve **customer experience**, and **optimize courier partner selection**. The model analyzes features such as **distance, courier partner, source pincode, destination state, and geographic coordinates** to estimate delivery time.

# Live Demo
A live interactive demo is available on **Hugging Face Spaces**, where users can input delivery details and receive the predicted delivery time.
https://huggingface.co/spaces/swatigupta110/courier-delivery-days-estimator

# Model Training Notebook
The complete **data analysis, model training, and experimentation notebook** can be viewed here.
https://colab.research.google.com/drive/1dzL6dmk-DurQIYXq1UnI_NsRCDCCgeVu?usp=sharing

# Detailed Explanation of Project
This project builds a **machine learning pipeline** to estimate courier delivery time based on **historical delivery data**.

# Data Preparation
The dataset was loaded using **Pandas** and filtered to ensure that **only successfully delivered orders** were used for model training.

### Steps Performed
- Data loaded from **CSV file** containing courier delivery records
- Only **delivered orders (`status = 40`)** were retained to analyze actual delivery durations
- The **status column was removed** since it contained a constant value after filtering
- Relevant columns such as **courier partner, pincode, geographic coordinates, and delivery time** were preserved
This ensures the model learns **only from successful delivery journeys**.

# Exploratory Data Analysis (EDA)
Basic exploratory analysis was performed to understand the **dataset structure and quality**.

### Key Steps Included
- Inspecting **data types and column information**
- Generating **summary statistics for numerical columns**
- Checking **missing values and null entries**
- Evaluating **unique values in categorical columns**
- Removing **rows with missing data**
This helped ensure that the dataset used for training was **clean and reliable**.

# Outlier Detection and Removal
Delivery time can sometimes contain **abnormal values** due to unexpected delays such as:
- Weather disruptions  
- Operational issues  
- Courier exceptions  
To address this, an **outlier detection mechanism** was implemented.

### Method Used

- Data was grouped by **origin state and destination state**
- A **monotonic regression model ([Isotonic Regression](https://kupas-data.medium.com/isotonic-regression-another-level-of-regression-method-1f22fd03d4cf))** was fitted using:
  - Distance (km)  
  - Delivery time  
- Residual errors between **predicted and actual delivery time** were calculated
- Data points exceeding a defined **residual threshold** were marked as **outliers**
These outliers were removed to ensure the model learns **normal delivery patterns rather than rare anomalies**.

# Feature Engineering
After cleaning the dataset, relevant features were selected for training.

## Input Features
- Courier partner
- Destination state
- Destination latitude
- Destination longitude
- Source pincode
- Distance between locations (in kilometers)

## Target Variable
- Delivery time (in days)
Categorical features were handled directly by **CatBoost**, which efficiently processes categorical data **without extensive preprocessing**.

# Train-Test Split
The dataset was divided into:
- **80% Training Data**
- **20% Testing Data**
This ensures that the model is evaluated on **unseen data** to measure its **generalization performance**.

# CatBoost Regression Model
A **CatBoost Regressor** was used for predicting delivery time.

CatBoost was chosen because it:
- Handles **categorical variables efficiently**
- Provides **strong performance on structured datasets**
- Requires **minimal feature preprocessing**

## Model Parameters
- **Iterations:** 1000  
- **Learning Rate:** 0.1  
- **Depth:** 6  
- **Loss Function:** Regression  
The model was trained using **CatBoost Pool objects** with categorical features specified.

# Model Evaluation
The trained model was evaluated using multiple **regression metrics**.

### Mean Absolute Error (MAE)
Measures the **average absolute difference** between predicted and actual delivery times.

### Mean Squared Error (MSE)
Penalizes **larger errors more heavily**.

### Root Mean Squared Error (RMSE)
Provides **error magnitude in the same unit** as the target variable.

### R² Score
Indicates **how well the model explains variance** in the delivery time.

### Adjusted R²
Adjusts the **R² score based on the number of features used**.

A **scatter plot of Actual vs Predicted values** was also generated to visually evaluate model accuracy.

# Feature Importance
CatBoost's **feature importance** was analyzed to determine which features contribute most to delivery time prediction.

This helps understand how factors like:
- Distance
- Courier partner
- Location information
influence **delivery duration**.

# Prediction Pipeline
A prediction function was implemented to handle **new delivery requests**.

### Steps Performed
1. Load the trained CatBoost model  
2. Prepare input data in the required feature format  
3. Run the prediction model  
4. Return estimated delivery days  

# Deployment
The trained model was integrated into a **Gradio-based web application** and deployed on **Hugging Face Spaces**.
This allows users to:
- Input courier delivery details
- Run the prediction model
- Receive **estimated delivery time instantly**
