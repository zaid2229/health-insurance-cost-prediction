import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import sklearn
import joblib

app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def results():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = float(request.form['smoker'])
    region = float(request.form['region'])
    
    X = np.array([[age,sex,bmi,children,smoker,region]])
    model = joblib.load('model_joblib_gr')
    Y_predict = model.predict(X)
    return jsonify({'Prediction': float(Y_predict)})


if __name__ == '__main__':
    app.run(debug = True, port = 1010)