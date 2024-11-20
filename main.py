from flask import Flask, jsonify, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)
# Load the trained model
pipe = pickle.load(open("models/finalHousePriceXGBModel.pkl", 'rb'))

# Sample location data (for demonstration)
locations_data = {
    "data_columns": [
        "Hindon Vihar", "Noida Extensions", "Sector 100", "Sector 104",
        "Sector 107", "Sector 108", "Sector 110", "Sector 115",
        "Sector 117", "Sector 118", "Sector 119", "Sector 120",
        "Sector 121", "Sector 122", "Sector 128", "Sector 129",
        "Sector 131", "Sector 133", "Sector 134", "Sector 135",
        "Sector 137", "Sector 143", "Sector 143B", "Sector 144",
        "Sector 146", "Sector 150", "Sector 151", "Sector 152",
        "Sector 16", "Sector 168", "Sector 19", "Sector 22",
        "Sector 25 Yamuna Express Way", "Sector 29", "Sector 32",
        "Sector 34", "Sector 37", "Sector 43", "Sector 44", "Sector 45",
        "Sector 46", "Sector 49", "Sector 50", "Sector 51", "Sector 52",
        "Sector 53", "Sector 61", "Sector 62", "Sector 70", "Sector 71",
        "Sector 72", "Sector 73", "Sector 74", "Sector 75", "Sector 76",
        "Sector 77", "Sector 78", "Sector 79", "Sector 82", "Sector 89",
        "Sector 93", "Sector 94", "Sikandarpur Village", "other"
    ]
}

# Encode locations
label_encoder = LabelEncoder()
label_encoder.fit(locations_data["data_columns"])  # Ensure encoder matches training

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_location_names', methods=['GET'])
def get_location_names():
    return jsonify(locations_data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        loc = request.form.get('location')
        bhk = int(request.form.get('uiBHK'))
        bathrooms = int(request.form.get('uiBATH'))
        size = int(request.form.get('size'))

        # Validate input location
        if loc not in locations_data["data_columns"]:
            loc = "other"  # Default to 'other' for unknown locations

        # Encode the location
        loc_encoded = label_encoder.transform([loc])[0]

        # Create input dataframe
        test = pd.DataFrame([{
            'loc_encoded': loc_encoded,
            'bhk': bhk,
            'size': size,
            'bathrooms': bathrooms,
            'bathrooms_per_bhk': bathrooms / bhk,
            'bathrooms_to_size_ratio': bathrooms / size
        }])

        # Make a prediction
        prediction = pipe.predict(test)[0]
        return str(int(prediction/1000)/100) + " Lakh"

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app on the local server
    app.run(debug=True)
