import json
import pickle

import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
# from ipynb.fs.full.spaceship_titanic_experiment import ExperimentalTransformer
import json
import numpy as np
import pandas as pd
from pandas import json_normalize

app=Flask(__name__)
## Load the model

model=joblib.load('model.sav')
scalar=joblib.load('scaler.sav')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    # return str(output[0])
    return jsonify(str(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="Prediction Algerian Forest Fire is {}".format(output))
   
    # return data

if __name__=="__main__":
    app.run(debug=True)
   
     