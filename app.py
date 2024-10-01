from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model_filename = 'car_price_prediction_model.pkl'
model = joblib.load(model_filename)

# Function to prepare the input data
def prepare_input(data):
    # Ensure all necessary fields are present
    try:
        input_features = {
            'Make': [data['Make']],  # Categorical feature
            'Year': [int(data['Year'])],
            'Engine Cylinders': [float(data['Engine Cylinders'])],
            'Number of Doors': [float(data['Number of Doors'])],
            'highway MPG': [float(data['highway MPG'])],
            'Transmission Type': [data.get('Transmission Type', 'AUTOMATIC')]  # Default to AUTOMATIC if not provided
        }
        
        # Convert the input features dictionary to a DataFrame
        input_df = pd.DataFrame(input_features)
        
    except KeyError as e:
        raise ValueError(f"Missing input field: {e}")

    return input_df
  
@app.route('/predict', methods=['POST'])
def predict_value():
    # Access form data
    data = request.form.to_dict()
    # Strip whitespace from keys
    data = {k.strip(): v for k, v in data.items()}
    
    print("Received data:", data)  # Log incoming data for debugging 

    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    try:
        input_data = prepare_input(data)
        prediction = model.predict(input_data)
        return jsonify({"predicted_value": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
