echo "Setting up virtual environment and installing dependencies..."
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -q
echo "Running the model training script..."
python hyperparameter_tuning/optuna_tuning.py
echo "Generated the best model and saved it to the model directory."
echo "Starting the Streamlit application..."
streamlit run demo/streamlit.py
echo "Deployment script completed successfully."