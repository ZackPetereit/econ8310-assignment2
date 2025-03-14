import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Step 1: Load the training data
train_url = ("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data = pd.read_csv(train_url)

# Step 2: Define features (X) and target (y)
X = data.drop(columns=["meal", "DateTime", "id"])  # Drop the target variable
y = data["meal"]  # Define the target variable

# Check and preprocess the data
# Identify categorical columns
categorical_columns = X.select_dtypes(include=["object"]).columns

# Apply one-hot encoding to categorical columns
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Random Forest model
model = RandomForestClassifier(random_state=42)  # Instantiate the model
modelFit = model.fit(X_train, y_train)  # Fit the model to training data

# Step 4: Evaluate the model
y_pred_val  =  modelFit.predict(X_val)  # Generate predictions on validation data
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val ))  # Print accuracy

# Step 5: Load the test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)

test_data_aligned = test_data.reindex(columns=X_train.columns, fill_value=0)

# Generate predictions
predict = modelFit.predict(test_data_aligned)

pred = pd.Series(predict, name="Predictions")
print(pred.head)  # Output predictions