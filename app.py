from flask import Flask, request, jsonify
import pandas as pd, joblib

app = Flask(__name__)
MODEL = joblib.load("rf_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = pd.DataFrame([data])
    yhat = int(MODEL.predict(X)[0])
    proba = float(MODEL.predict_proba(X)[:,1][0])
    return jsonify({"prediction": yhat, "probability": proba})

if __name__ == "__main__":
    app.run(port=5000)
