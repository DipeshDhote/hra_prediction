from src.HeartAttackRiskPrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods =["GET","POST"])
def predict_risk():
    if request.request=="POST":
        return render_template("form.html")
    
    
    else:
        data=CustomData( 
            Age=int(request.form.get('Age')),
            Sex = int(request.form.get('Sex')),
            Cholesterol= int(request.form.get('Cholesterol')),
            Heart_Rate = int(request.form.get('Heart Rate')),
            Diabetes = int(request.form.get('Diabetes')),
            Smoking = int(request.form.get('Smoking')),
            Alcohol_Consumption = int(request.form.get('Alcohol Consumption')),
            Previous_Heart_Problems = int(request.form.get('Previous Heart Problem')),
            Medication_Use = int(request.form.get('Medication Use')),
            Triglycerides = int(request.form.get('Triglycerides')),
            Max_Bp = int(request.form.get('Max BP')),
            Min_Bp = int(request.form.get('Min BP'))
        )

                                                                                                
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
      
    app.run(host="0.0.0.0",port=8080)