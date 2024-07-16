from flask import Flask, render_template, request
import random
import predict
import pickle
import numpy as np

app = Flask(__name__)   # Flask constructor

@app.route('/') 
@app.route('/predict',methods=['GET','POST'])
def predict_result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        print(request.form)
        print(to_predict_list)
        to_predict_list = list(to_predict_list.values())
        print(to_predict_list)
        result = predict.predict([to_predict_list])
        if result == "Normal":
            result = f'{result}'

        else:
            result = f'{result}'

        return render_template('prediction.html',prediction=result)
        # return render_template('/prediction.html',prediction = result,pie_chart=pie_chart)
        
    return render_template('/index.html')


if __name__=='__main__':
   app.run(debug=False)
