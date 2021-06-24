import numpy as np
from flask  import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('./model.pkl','rb'))

@app.route('/')
def root():
    return render_template("predict.html")

@app.route("/predict",methods=['POST','GET'])
def predict():
    # rendering results on HTML GUI
    feat=[int(x) for x in request.form.values()]
    final_features=[np.array(feat)]
    prediction=model.predict(final_features)

    output=round(prediction[0],2)
    return render_template('predict.html',prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True,port=5000)