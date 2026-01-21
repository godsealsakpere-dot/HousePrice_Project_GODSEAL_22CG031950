import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_openml
import joblib
import os

# 1. Load the Dataset
# We try to load 'train.csv' if it exists (Kaggle version). 
# If not, we fetch the Ames Housing dataset from OpenML which is the same source.
try:
    if os.path.exists('train.csv'):
        df = pd.read_csv('train.csv')
    else:
        print("train.csv not found. Fetching from OpenML...")
        housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
        df = housing.frame
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Feature Selection (Selecting 6 Numerical Features)
# Target: SalePrice
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
target = 'SalePrice'

# Filter dataset
df_selected = df[features + [target]].copy()

# 3. Data Preprocessing
# Handling missing values (Fill with median for numerical stability)
df_selected.fillna(df_selected.median(), inplace=True)

X = df_selected[features]
y = df_selected[target]

# 4. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Pipeline Creation (Scaling + Model)
# We use Random Forest Regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature Scaling
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. Train the Model
print("Training Model...")
pipeline.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = pipeline.predict(X_test)

print("--- Model Evaluation ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 8. Save the Model
# Ensure model directory exists
if not os.path.exists('model'):
    os.makedirs('model')

model_filename = 'model/house_price_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")