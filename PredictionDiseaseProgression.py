import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data  # Features
y = diabetes.target  # Target (disease progression)

# Create a DataFrame for easier manipulation
diabetes_df = pd.DataFrame(data=X, columns=diabetes.feature_names)
diabetes_df['target'] = y

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a dictionary to store the models and their names
models = {
    'ElasticNet Regression': ElasticNet(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(),
    'MLP Regressor': MLPRegressor(max_iter=1000, random_state=42)
}

# Initialize a dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results[name] = {'MSE': mse, 'R^2': r2}
    
    # Print the results
    print(f"Model: {name}")
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)
    print("\n" + "-"*40 + "\n")

# Summary of results
print("Summary of Model Performance:")
for name, metrics in results.items():
    print(f"{name}: MSE = {metrics['MSE']:.2f}, R^2 = {metrics['R^2']:.2f}")


# Example predictors (feature values for prediction)
# These values are based on the features in the diabetes dataset
example_predictors = np.array([[0.0380759064336729, 0.0506801187398187, 0.0616962065183617, 
                                 0.0218723542451102, 0.0452272829108499, 0.0616962065183617, 
                                 0.0122823713704307, 0.0149176622797431, 0.0222101502407711, 
                                 0.00344524224304873]])
# Make predictions for the example predictors using each model
print("\nPredictions for Example Input Values:")
for name, model in models.items():
    prediction = model.predict(example_predictors)
    print(f"{name}: Predicted disease progression = {prediction[0]:.2f}")