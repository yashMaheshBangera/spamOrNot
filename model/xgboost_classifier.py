from xgboost import XGBClassifier

def create_xgboost_classifier():
    """
    Create an XGBoost classifier with default parameters.
    
    Returns:
        XGBClassifier: An instance of the XGBoost classifier.
    """
    return XGBClassifier()
    
def train_xgboost_classifier(model, X_train, y_train):
    """
    Train the XGBoost classifier on the training data.
    
    Args:
        model (XGBClassifier): The XGBoost classifier to be trained.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
    
    Returns:
        XGBClassifier: The trained XGBoost classifier.
    """
    model.fit(X_train, y_train)
    return model