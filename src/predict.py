import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open("models/1.0-xgboost.bin", 'rb') as f:
    model, transformer = pickle.load(f)

app = Flask('bank-marketing')

@app.route('/predict', methods=['POST'])
def predict():

    client = request.get_json()
    client_id = client['client_id']
    del client['client_id']

    X = pd.DataFrame(client, index=[client_id])
    X_transformed = transformer.transform(transformer.add_custom_features(X))
    
    y_proba = model.predict_proba(X_transformed)[0, 1]

    response = {
        'client_id': int(client_id),
        'subscribe_proba': round(float(y_proba), 4)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
