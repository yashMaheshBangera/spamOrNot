Write-Host "Setting up virtual environment and installing dependencies..."
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -q
Write-Host "Running the model training script..."
python hyperparameter_tuning/optuna_tuning.py
Write-Host "Generated the best model and saved it to the model directory."
Write-Host "Starting the Streamlit application..."
streamlit run demo/streamlit.py
Write-Host "Deployment script completed successfully."