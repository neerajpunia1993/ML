import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and their parameters for tuning
models = {
    'Ridge': {
        'model': Ridge(),
        'params': {'alpha': np.logspace(-4, 4, 10)}
    },
    'Lasso': {
        'model': Lasso(),
        'params': {'alpha': np.logspace(-4, 4, 10)}
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {
            'alpha': np.logspace(-4, 4, 10),
            'l1_ratio': np.linspace(0.1, 1.0, 10)
        }
    }
}

results = {}

# Perform grid search with cross-validation for each model
for name, model_info in models.items():
    grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model and evaluate on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    results[name] = {
        'Best Params': grid_search.best_params_,
        'Test MSE': mse,
        'Grid Search CV Results': grid_search.cv_results_
    }

    # Plotting CV results for alpha
    if name in ['Ridge', 'Lasso']:
        plt.figure(figsize=(10, 5))
        plt.semilogx(grid_search.param_grid['alpha'], -grid_search.cv_results_['mean_test_score'], marker='o', label='CV MSE')
        plt.title(f'{name} - Cross-Validation MSE vs Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Mean Squared Error')
        plt.grid()
        plt.legend()
        plt.show()

# Display the results
for name, result in results.items():
    print(f"{name}:")
    print(f"  Best Params: {result['Best Params']}")
    print(f"  Test MSE: {result['Test MSE']:.4f}\n")

# Example prediction with a new data point
# Let's create a sample input using the feature means from the dataset
sample_input = np.mean(X_train, axis=0).reshape(1, -1)
sample_input_scaled = scaler.transform(sample_input)

# Predicting with the best model (e.g., Ridge)
ridge_best_params = results['Ridge']['Best Params']
ridge_model = Ridge(alpha=ridge_best_params['alpha'])
ridge_model.fit(X_train_scaled, y_train)
predicted_value = ridge_model.predict(sample_input_scaled)

print(f"Predicted diabetes progression for the sample input: {predicted_value[0]:.2f}")
