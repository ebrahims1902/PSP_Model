from flask import Flask, render_template, request
import pickle
import numpy as np
from numpy.core.numeric import outer

app = Flask(__name__)

model = pickle.load(open("PSP_MODEL.pkll", 'rb')) 

@app.route('/')
def helloworld():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    float_features = [np.float64(x) for x in request.form.values()]
    final = [np.array(float_features)]
    prediction = model.predict(final)
    return render_template('predict.html', pred=prediction)

if __name__ == '__main__':
    app.run(debug=True)
