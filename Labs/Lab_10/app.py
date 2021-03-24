#This is Heroku Deployment Lectre
from flask import Flask, request, render_template
import os
import pickle

print(os.getcwd())
path = os.getcwd()

with open('Models/SVM.pkl', 'rb') as f:
    svm = pickle.load(f)

with open('Models/DecisionTree.pkl', 'rb') as f:
    decision_tree = pickle.load(f)

with open('Models/RandomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)


def get_predictions(age, sex, chol, req_model):
    mylist = [age, sex, chol]
    mylist = [float(i) for i in mylist]
    vals = [mylist]

    if req_model == 'SVM':
        #print(req_model)
        return svm.predict(vals)[0]

    elif req_model == 'DecisionTree':
        #print(req_model)
        return decision_tree.predict(vals)[0]

    elif req_model == 'RandomForest':
        #print(req_model)
        return random_forest.predict(vals)[0]
    else:
        return "Cannot Predict"


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def my_form_post():
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        chol = request.form['chol']
        req_model = request.form['req_model']

        target = get_predictions(age, sex, chol, req_model)

        if target == 1:
            heart_disease = 'Customer has heart disease.'
        else:
            heart_disease = 'Customer does not have heart disease.'

        return render_template('home.html', target=target, heart_disease=heart_disease)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)