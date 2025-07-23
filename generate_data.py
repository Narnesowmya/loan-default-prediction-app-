import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Generate dummy dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'loan_amount': np.random.uniform(1000, 50000, n_samples),
    'interest_rate': np.random.uniform(1, 20, n_samples),
    'income': np.random.uniform(20000, 200000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'term': np.random.choice([180, 240, 360], n_samples),
})

# Target: 1 for default, 0 for not
data['default'] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

# Features and target
X = data.drop('default', axis=1)
y = data['default']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('model/loan_default_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model/loan_default_model.pkl")
