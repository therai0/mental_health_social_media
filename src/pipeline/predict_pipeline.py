import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_model_object


class PredictPipeline:
    def __init(self):
        pass

    def predict(self,data_frame):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_model_object(file_path=model_path)
            preprocessor = load_model_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(data_frame)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 Gender:str,
                 Social_Media_Platform:str,
                 Age:int,
                 Daily_Screen_Time:int,
                 Stress_Level:int,
                 Days_Without_Social_Media:int,
                 Exercise_Frequency:int,
                 Sleep_Quality:int):
        
        self.gender =Gender
        
        self.Social_Media_Platform = Social_Media_Platform

        self.age = Age

        self.Daily_Screen_Time = Daily_Screen_Time

        self.Stress_Level = Stress_Level

        self.Days_Without_Social_Media = Days_Without_Social_Media

        self.Exercise_Frequency = Exercise_Frequency

        self.Sleep_Quality = Sleep_Quality
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
            "Gender": [self.gender],
            "Social_Media_Platform": [self.Social_Media_Platform],
            "Age": [self.age],
            "Daily_Screen_Time": [self.Daily_Screen_Time],
            "Stress_Level": [self.Stress_Level],
            "Days_Without_Social_Media": [self.Days_Without_Social_Media],
            "Exercise_Frequency": [self.Exercise_Frequency],
            "Sleep_Quality": [self.Sleep_Quality]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)