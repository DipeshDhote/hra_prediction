import os
import sys
import pandas as pd
from src.HeartAttackRiskPrediction.exception import customexception
from src.HeartAttackRiskPrediction.logger import logging
from src.HeartAttackRiskPrediction.utils import load_object


class PredictPipeline:
    def __init__ (self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 Age :int,
                 Cholesterol : int,
                 Heart_Rate :int ,
                 Diabetes : int ,
                 Smoking :int ,
                 Alcohol_Consumption : int,
                 Previous_Heart_Problems : int,
                 Medication_Use : int ,
                 Triglycerides : int,
                 Max_BP : int,
                 Min_BP : int ):
                 
        self.Age = Age
        self.Cholesterol = Cholesterol
        self.Heart_Rate = Heart_Rate
        self.Diabetes = Diabetes
        self.Smoking = Smoking
        self.Alcohol_Consumption = Alcohol_Consumption
        self.Previous_Heart_Problems = Previous_Heart_Problems
        self.Medication_Use = Medication_Use
        self.Triglycerides = Triglycerides
        self.Max_BP = Max_BP
        self.Min_BP = Min_BP      
                         
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'Age':[self.Age],
                    'Cholesterol':[self.Cholesterol],
                    'Heart_Rate':[self.Heart_Rate],
                    'Diabetes':[self.Diabetes],
                    'Smoking':[self.Smoking],
                    'Alcohol Consuption':[self.Alcohol_Consumption],
                    'Privious Heart Problem':[self.Previous_Heart_Problems],
                    'Medication Use':[self.Medication_Use],
                    'Triglycerides':[self.Triglycerides],
                    'Max BP':[self.Max_BP],
                    'Min Bp':[self.Min_BP]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
