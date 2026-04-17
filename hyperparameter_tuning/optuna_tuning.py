import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import optuna
from data.fetch import get_spambase
from data.preprocess import preprocess_data
from model.xgboost_classifier import create_xgboost_classifier
from ucimlrepo import fetch_ucirepo 
  
# Fetch raw dataset 
spambase = fetch_ucirepo(id=94) 

dictionary = get_spambase(spambase)
# Extract features and targets from the dictionary
df = dictionary['data']


# Pre-process the data
try:
    X_train, X_test, y_train, y_test = preprocess_data(df)
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Data preprocessing failed. Please check the error messages above.")
except Exception as e:
    exit(1)

# Create a Classifier instance
model = create_xgboost_classifier()

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameters to tune
    params = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    # Train the model with the current hyperparameters
    model.set_params(**params)
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    accuracy = model.score(X_test, y_test)
    
    return accuracy

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize', storage="sqlite:///optuna_study.db", load_if_exists=True)
study.optimize(objective, n_trials=100)

# Print the best trial
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")

# Save the best model
best_params = study.best_params
best_model = create_xgboost_classifier()
best_model.set_params(**best_params)
best_model.fit(X_train, y_train)

import joblib
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
joblib.dump(best_model, model_path)
print(f"Best model saved to {model_path}")