import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Step 1: Load the training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
data = pd.read_csv(train_url)

# Step 2: Define features (X) and target (y)
X = data.drop(columns=['meal'])  # Drop the target variable
y = data['meal']  # Define the target variable

# Check and preprocess the data
# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest model
model = RandomForestClassifier(random_state=42)  # Instantiate the model
modelFit = model.fit(X_train, y_train)  # Fit the model to training data

# Calibrate the model for better probability predictions
calibrated_model = CalibratedClassifierCV(modelFit, method='sigmoid')
calibrated_model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred_val_prob = calibrated_model.predict_proba(X_val)[:, 1]  # Probabilities for the positive class
y_pred_val_binary = calibrated_model.predict(X_val)  # Binary predictions for accuracy calculation
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val_binary))  # Print accuracy

# Step 5: Load the test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

# Align test data with training data columns
test_data_aligned = pd.get_dummies(test_data)
test_data_aligned = test_data_aligned.reindex(columns=X_train.columns, fill_value=0)

# Generate probabilistic predictions for the test data
pred_prob = calibrated_model.predict_proba(test_data_aligned)[:, 1]  # Probabilities for the positive class

# Save predictions as a series
pred = pd.Series(pred_prob, name="Predictions")
print(pred)  # Output predictions
