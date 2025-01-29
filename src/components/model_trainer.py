import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","trained_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def train_model(self, train_array, test_array):
        """
        This function is used to train the models using the train and test data
        """
        try:
            logging.info('Split train and test input data')
            
            # Create the train and test data to train the model
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            
            # List of models to be trained
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K Nearest Neighbors': KNeighborsRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostClassifier()
            }
            
            logging.info('Model training has started')
            # Evaluate the models
            model_report:dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models)
            
            # Get the best performing model's score
            best_model_score = max(sorted(model_report.values()))
            
            # Get the best performing model's name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.75:
                raise CustomException('No best model found')
        
            logging.info('Save the best model')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Make prediction using the best model
            predicted = best_model.predict(X_test)
            
            # Evaluate the model using the R2 score
            r2 = r2_score(y_test, predicted)
            
            logging.info('Model training has been completed successfully')
            return r2, best_model_name
        
        except Exception as e:
            raise CustomException(e, sys)