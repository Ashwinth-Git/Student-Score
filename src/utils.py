import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function is used to save the object to the specified file path
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    This function is used to train and evaluate the model with different
    regression algorithms and parameters
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)
            
            # Train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Make prediction using the trained models
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate the R2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Save the model scores
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_model(file_path):
    """
    This function is used to load the object from the specified file path
    """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
        
    except Exception as e:
        raise CustomException(e, sys)
