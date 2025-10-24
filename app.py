import pickle
from flask import Flask,request,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData


application = Flask(__name__)
app = application


@app.route("/",methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
        Gender=request.form.get('Gender'),
        Social_Media_Platform=request.form.get('Social_Media_Platform'),
        Age=float(request.form.get('Age')),
        Daily_Screen_Time=float(request.form.get('Daily_Screen_Time')),
        Stress_Level=float(request.form.get('Stress_Level')),
        Days_Without_Social_Media=float(request.form.get('Days_Without_Social_Media')),
        Exercise_Frequency=float(request.form.get('Exercise_Frequency')),
        Sleep_Quality=float(request.form.get('Sleep_Quality'))
)
        data_frame = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(data_frame)
        return render_template('home.html',result=result) 

if __name__ == "__main__":
    app.run(debug=True,port='8000')
