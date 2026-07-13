from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        features = []

        for value in request.form.values():
            features.append(float(value))

        features = np.array(features).reshape(1, -1)

        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Credit Card Rejected ❌"
        else:
            result = "Credit Card Approved ✅"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return str(e)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)