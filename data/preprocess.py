#The following code is for data preprocessing which includes:
# 1. Minority Oversampling of the data
# 2. Rename column with invalid character for XGBoost
# 3. Feature Engineering of a new column
# 4. Data splitting
#This is based on the cleaning.ipynb file which provides more insights into the data pre-processing choices.

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    try:
        # Separate features and target variable
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        print("Starting Random Oversampling to balance the classes...")
        # Apply Random Oversampling to balance the classes
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)


        print(f"Original class distribution: {y.value_counts()}")
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
        
        # Create a new DataFrame with the resampled data
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['Class'] = y_resampled
        df=df_resampled
        print("Oversampling completed successfully.")
    except Exception as e:
        print(f"Error during oversampling: {e}")
    try:
        print("Starting feature engineering...")
        # Rename column with invalid character for XGBoost
        df = df.rename(columns={'char_freq_[': 'char_freq_lb'})
        df['punctuation_count'] = df[['char_freq_;','char_freq_(','char_freq_lb','char_freq_!','char_freq_$','char_freq_#']].sum(axis=1)
        print("Feature engineering completed successfully.")
    except KeyError as e:
        print(f"Error: Column not found - {e}")
    
    try:
        print("Splitting data into features and target variable...")
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Class']), df['Class'], test_size=0.2, random_state=42)
        print("Data splitting completed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error during data splitting: {e}")
        return None,None,None,None