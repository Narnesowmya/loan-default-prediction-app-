# loan_default_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load the dataset
csv_path = 'Loan_Default.csv'  # Ensure this file is in the same folder
df = pd.read_csv(csv_path)

print("✅ Dataset loaded successfully!")

# View columns
print("Columns:", df.columns.tolist())

# ----- Step 1: Basic Preprocessing -----
# Let's assume 'Status' is the target column
target_column = 'Status'

# Drop unnecessary or ID columns
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Drop rows with missing target
df.dropna(subset=[target_column], inplace=True)

# Fill or drop other missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != target_column:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target if it's categorical
if df[target_column].dtype == 'object':
    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column])
else:
    target_encoder = None

# ----- Step 2: Train/Test Split -----
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----- Step 3: Train the Model -----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----- Step 4: Evaluate -----
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----- Step 5: Save Model -----
model_dir = os.path.join(os.getcwd(), "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "loan_default_model.pkl"))

# Save encoders (optional if you need them in Streamlit app)
if target_encoder:
    joblib.dump(target_encoder, os.path.join(model_dir, "target_encoder.pkl"))
if label_encoders:
    joblib.dump(label_encoders, os.path.join(model_dir, "feature_encoders.pkl"))

print("✅ Model saved successfully in 'model/loan_default_model.pkl'")
