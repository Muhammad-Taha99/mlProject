import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('./models/Model1.pkl', 'rb'))
model2 = pickle.load(open('./models/Model2.pkl', 'rb'))
model3 = pickle.load(open('./models/Model3.pkl', 'rb'))
model4 = pickle.load(open('./models/Model4.pkl', 'rb'))
model5 = pickle.load(open('./models/Model5.pkl', 'rb'))
model6 = pickle.load(open('./models/Model6.pkl', 'rb'))
model7 = pickle.load(open('./models/Model7.pkl', 'rb'))
model8 = pickle.load(open('./models/Model8.pkl', 'rb'))
model9 = pickle.load(open('./models/Model9.pkl', 'rb'))
model10 = pickle.load(open('./models/Model10.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # request.form['open']
    vr = model1.predict(final_features)
    rfr = model2.predict(final_features)
    brr = model3.predict(final_features)
    dtr = model4.predict(final_features)
    gbr = model5.predict(final_features)
    krr = model6.predict(final_features)
    lr = model7.predict(final_features)
    lss = model8.predict(final_features)
    mlp = model9.predict(final_features)
    svr = model10.predict(final_features)

    svr = round((svr[0]), 2)
    vr = round((vr[0]), 2)
    rfr = round((rfr[0]), 2)
    brr = round((brr[0]), 2)
    dtr = round((dtr[0]), 2)
    gbr = round((gbr[0]), 2)
    krr = round((krr[0]), 2)
    lr = round((lr[0]), 2)
    mlp = round((mlp[0]), 2)
    lss = round((lss[0]), 2)

    prediction = [["Support Vector Machine", svr], ["Ridge", lr],
                  ["Tweedie Regressor", rfr], ["Nearest Centroid", brr],
                  ["Decision Tree Regressor", krr], ["Lasso Lars", dtr],
                  ["Lasso Regressor", gbr], ["Random Forest Regressor", lss],
                  ["Gradient Boosting Regressor", mlp], ["Linear Regression", vr]]

    return render_template('index.html', prediction = prediction)
    
if __name__ == "__main__":
    app.run(debug=True)
