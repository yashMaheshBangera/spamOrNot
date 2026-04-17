import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from ucimlrepo import fetch_ucirepo 
from data.fetch import get_spambase
import re

def preprocess_email(email):
    if isinstance(email,str):
        # Fetch raw dataset 
        spambase = fetch_ucirepo(id=94) 

        dictionary = get_spambase(spambase)
        # Extract features and targets from the dictionary
        df = dictionary['data']

        columns = df.columns
        print(columns)
        print("Preprocessing email input to extract features for the model...")
        words=[]
        characters = [';', '(', '[', '!', '$', '#']
        for i in range(0,len(columns)-4,1): 
            if columns[i].startswith('word_'):
                words.append(columns[i].split('_')[-1])
        email_features=[]
        character_features=[]
        for i in range(0,len(words),1): 
            count = email.split().count(words[i])
            email_features.append(round((count/len(email.split()))*100,2))
        for i in range(0,len(characters),1):
            count = email.count(characters[i])
            character_features.append(count)
        total_characters = sum(character_features)
        for i in range(0,len(character_features),1):
            if total_characters > 0:
               email_features.append(round((character_features[i]/total_characters)*100,2))
            else:
                email_features.append(0)

        runs = re.findall(r'[A-Z]+',email)
        lengths = [len(run) for run in runs]
        email_features.append(round(sum(lengths)/len(lengths),2))
        email_features.append(max(lengths))
        email_features.append(sum(lengths))
        email_features.append(total_characters)
        return email_features
    else:
        raise ValueError("Input email must be a string.")
    