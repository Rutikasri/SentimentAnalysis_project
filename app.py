from flask import Flask, request, render_template, jsonify
import joblib

# Load the model and label encoder
model = joblib.load('sentiment_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the user (in JSON format)
        input_data = request.get_json()
        input_text = input_data.get('text', '').strip()  # Ensure text is not None

        if not input_text:
            return jsonify({'error': 'Input text is empty'}), 400

        # Make prediction using the model
        prediction_encoded = model.predict([input_text])[0]

        # Decode the predicted sentiment label
        sentiment = label_encoder.inverse_transform([prediction_encoded])[0]

        # Return the prediction result as JSON
        return jsonify({'prediction': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
