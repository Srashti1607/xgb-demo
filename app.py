from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from xgboost import XGBClassifier

# Load the trained model and scaler (assuming you saved the scaler during training)
model_path = 'xgboost.pkl'

# Load the model and scaler
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    start_time = time.time()

    # Extract data from form
    int_features = [float(x) for x in request.form.values()]  # Cast to float for compatibility with StandardScaler
    final_features = [np.array(int_features)]

    # Scale the input features using the same scaler used during training
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(final_features)

    # Make prediction
    prediction = model.predict(scaled_features)
    output = 'Healthy' if prediction[0] == 1 else 'Unhealthy'

    end_time = time.time()
    time_taken = end_time - start_time  # Time taken to process the request

    # Display the result
    prediction_text = f"Prediction: {output}. Time taken: {time_taken:.4f} seconds."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
