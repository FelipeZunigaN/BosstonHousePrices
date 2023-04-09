import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

## load model
regmodel = pickle.load(open('regmodel.pkl','rb') )
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print( np.array(list(data.values())).reshape(1,-1) )
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])

    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)


# print(regmodel.predict(np.array([[-0.46808082, -0.49218391, -1.79054859, -1.36081455, -6.19407736,
#         -8.3243657 , -2.49509073, -1.70191083, -1.20724766, -2.45184812,
#         -8.69691627, -4.1390015 , -1.89639069]])))

