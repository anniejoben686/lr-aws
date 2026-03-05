from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        income = float(request.form["income"])
        credit = float(request.form["credit"])

        # Validation rules
        if income < 1000 or income > 1000000:
            return render_template("index.html",
            prediction_text="Monthly income must be between 1,000 and 1,000,000")

        if credit < 300 or credit > 900:
            return render_template("index.html",
            prediction_text="Credit score must be between 300 and 900")

        features = np.array([[income, credit]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("index.html",
                               prediction_text=result)

    except:
        return render_template("index.html",
                               prediction_text="Invalid input. Please enter numbers.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)